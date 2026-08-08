"""
Step 5.3: Trial-aligned evaluation with continuous filtering,
trial-level aggregation, and optional standardization.
"""
import argparse
import threading
import time
from collections import deque

import numpy as np
import onnxruntime as ort
from pylsl import resolve_streams, StreamInlet
from scipy.signal import butter, lfilter, lfilter_zi

MODEL_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]
VOLTS_TO_MICROVOLTS = 1e6
LABEL_MAP = {"feet": 0, "left_hand": 1, "right_hand": 2, "tongue": 3}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

FILTER_LOW_HZ = 4.0
FILTER_HIGH_HZ = 38.0
FILTER_ORDER = 4
TRIAL_DURATION_SEC = 4.0


class RollingBuffer:
    def __init__(self, window_seconds, sfreq):
        self.maxlen = int(window_seconds * sfreq)
        self.buffer = deque(maxlen=self.maxlen)
        self.lock = threading.Lock()
        self.n_received = 0

    def append(self, sample):
        with self.lock:
            self.buffer.append(sample)
            self.n_received += 1

    def get_window(self):
        with self.lock:
            if len(self.buffer) < self.maxlen:
                return None
            snapshot = np.array(self.buffer, dtype=np.float32).T
        return snapshot


def eeg_receiver_loop(inlet, channel_indices, buffer, b, a, zi, stop_event):
    while not stop_event.is_set():
        sample, ts = inlet.pull_sample(timeout=0.5)
        if sample is None:
            continue
        raw = np.array(sample, dtype=np.float32)[channel_indices] * VOLTS_TO_MICROVOLTS
        filtered, zi[:] = lfilter(b, a, [raw], axis=0, zi=zi)
        buffer.append(filtered[0])


def marker_receiver_loop(inlet, latest_marker_holder, stop_event):
    while not stop_event.is_set():
        sample, ts = inlet.pull_sample(timeout=0.5)
        if sample is not None:
            latest_marker_holder["label"] = sample[0]
            latest_marker_holder["onset_time"] = time.time()  # wall-clock


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--no_norm", action="store_true", help="Disable z-score standardization")
    parser.add_argument("--debug", action="store_true", help="Print raw logits")
    args = parser.parse_args()

    effective_trial_duration = TRIAL_DURATION_SEC / args.speed
    print(f"Playback speed: {args.speed}x -> effective trial window: {effective_trial_duration:.2f}s")
    print(f"Standardization: {'OFF' if args.no_norm else 'ON (per-window)'}")
    print(f"Debug raw logits: {'ON' if args.debug else 'OFF'}")

    print("Resolving LSL streams...")
    streams = resolve_streams(wait_time=5.0)
    eeg_info = marker_info = None
    for s in streams:
        if s.type() == "EEG" and s.name() == "SimulatedEEG":
            eeg_info = s
        elif s.type() == "Markers":
            marker_info = s
    if eeg_info is None or marker_info is None:
        print("Streams not found.")
        return

    eeg_inlet = StreamInlet(eeg_info)
    marker_inlet = StreamInlet(marker_info)

    # Get channel metadata
    all_channels, sfreq = [], None
    for attempt in range(10):
        full_info = eeg_inlet.info()
        sfreq = full_info.nominal_srate()
        ch = full_info.desc().child("channels").child("channel")
        while not ch.empty():
            all_channels.append(ch.child_value("label"))
            ch = ch.next_sibling("channel")
        if all_channels:
            break
        time.sleep(0.5)
    if not all_channels:
        print("Failed to retrieve channel metadata.")
        return

    print(f"Stream channels ({len(all_channels)}): {all_channels[:5]}...")
    print(f"Model expects channels: {MODEL_CHANNELS[:5]}...")

    # Check that all MODEL_CHANNELS exist in the stream
    missing = [c for c in MODEL_CHANNELS if c not in all_channels]
    if missing:
        print(f"WARNING: Missing channel(s): {missing}")

    channel_indices = [all_channels.index(c) for c in MODEL_CHANNELS]
    print(f"Using channel indices: {channel_indices[:5]}...")

    # Design stateful filter
    nyquist = sfreq / 2.0
    b, a = butter(FILTER_ORDER, [FILTER_LOW_HZ/nyquist, FILTER_HIGH_HZ/nyquist], btype="band")
    zi = lfilter_zi(b, a)[:, np.newaxis] * np.ones((1, len(MODEL_CHANNELS)))
    print(f"Filter: {FILTER_LOW_HZ}-{FILTER_HIGH_HZ}Hz")

    # Load ONNX
    onnx_path = "../../models/eegnet_subject1.onnx"
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print(f"ONNX input shape: {session.get_inputs()[0].shape}")

    buffer = RollingBuffer(window_seconds=4.0, sfreq=sfreq)
    latest_marker = {"label": None, "onset_time": None}
    stop_event = threading.Event()

    threading.Thread(target=eeg_receiver_loop,
                     args=(eeg_inlet, channel_indices, buffer, b, a, zi, stop_event),
                     daemon=True).start()
    threading.Thread(target=marker_receiver_loop,
                     args=(marker_inlet, latest_marker, stop_event),
                     daemon=True).start()

    print("Threads started. Waiting for buffer fill...\n")
    print(f"{'Time':<8}{'GroundTruth':<14}{'Predicted':<14}{'Confidence':<12}{'Status':<10}")

    # Trial aggregation
    current_gt = None
    current_onset = None
    trial_probs = []
    trial_active = False
    trial_finalized = False

    n_trials = 0
    n_correct = 0
    frame_counter = 0

    try:
        while True:
            time.sleep(0.25)
            window = buffer.get_window()
            if window is None:
                continue
            frame_counter += 1

            # --- Optional standardization ---
            if args.no_norm:
                model_input = window[np.newaxis, np.newaxis, :, :].astype(np.float32)
            else:
                mean = np.mean(window, axis=1, keepdims=True)
                std = np.std(window, axis=1, keepdims=True)
                std = np.where(std < 1e-6, 1e-6, std)
                window_norm = (window - mean) / std
                model_input = window_norm[np.newaxis, np.newaxis, :, :].astype(np.float32)

            # Ensure 1001 time points
            if model_input.shape[-1] != 1001:
                pad_width = 1001 - model_input.shape[-1]
                model_input = np.pad(model_input, ((0,0),(0,0),(0,0),(0,pad_width)), mode="edge")

            logits = session.run(None, {input_name: model_input})[0][0]
            if args.debug:
                print(f"Raw logits: {logits}")

            probs = softmax(logits)
            pred_idx = int(np.argmax(probs))
            pred_label = IDX_TO_LABEL[pred_idx]
            confidence = probs[pred_idx]

            gt_label = latest_marker["label"]
            onset_time = latest_marker["onset_time"]
            now = time.time()

            # New trial detection
            if gt_label is not None and gt_label != current_gt:
                if trial_active and not trial_finalized and trial_probs:
                    avg_probs = np.mean(trial_probs, axis=0)
                    final_pred = IDX_TO_LABEL[int(np.argmax(avg_probs))]
                    is_correct = (final_pred == current_gt)
                    n_trials += 1
                    n_correct += is_correct
                    print(f">>> TRIAL {n_trials} END: GT={current_gt}, Pred={final_pred}, Correct={is_correct}")
                current_gt = gt_label
                current_onset = onset_time
                trial_probs = []
                trial_active = True
                trial_finalized = False

            # Score window: entire trial duration
            in_valid_window = (
                trial_active
                and current_onset is not None
                and current_onset <= now <= current_onset + effective_trial_duration
            )

            if in_valid_window:
                trial_probs.append(probs)
                status = "accum"
            else:
                status = "rest"
                if trial_active and not trial_finalized and current_onset is not None and now > current_onset + effective_trial_duration:
                    if trial_probs:
                        avg_probs = np.mean(trial_probs, axis=0)
                        final_pred = IDX_TO_LABEL[int(np.argmax(avg_probs))]
                        is_correct = (final_pred == current_gt)
                        n_trials += 1
                        n_correct += is_correct
                        print(f">>> TRIAL {n_trials} END (timeout): GT={current_gt}, Pred={final_pred}, Correct={is_correct}")
                    trial_finalized = True
                    trial_active = False

            if frame_counter % 50 == 0:
                print(f"[DIAG] Window mean: {np.mean(mean):.3f} µV, std: {np.mean(std):.3f} µV")

            display_gt = gt_label if gt_label else "—"
            print(f"{now % 1000:<8.1f}{display_gt:<14}{pred_label:<14}{confidence:<12.3f}{status:<10}")

    except KeyboardInterrupt:
        stop_event.set()
        if trial_active and not trial_finalized and trial_probs:
            avg_probs = np.mean(trial_probs, axis=0)
            final_pred = IDX_TO_LABEL[int(np.argmax(avg_probs))]
            is_correct = (final_pred == current_gt)
            n_trials += 1
            n_correct += is_correct
            print(f">>> FINAL TRIAL: GT={current_gt}, Pred={final_pred}, Correct={is_correct}")

        print(f"\n\n--- FINAL TRIAL-ALIGNED RESULTS ---")
        print(f"Total trials scored: {n_trials}")
        print(f"Correct matches: {n_correct}")
        if n_trials > 0:
            print(f"Accuracy: {n_correct/n_trials:.3f}")


if __name__ == "__main__":
    main()
