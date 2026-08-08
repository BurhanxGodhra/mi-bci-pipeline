"""
Step 5.3: Trial-aligned evaluation - only scores predictions made during
the actual valid imagery window [cue_onset, cue_onset+4s], not during rest.
This gives a fair, apples-to-apples comparison against the offline 74.3% benchmark.
"""
import threading
import time
from collections import deque

import numpy as np
from scipy.signal import butter, filtfilt
import onnxruntime as ort
from pylsl import resolve_streams, StreamInlet

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
TRIAL_DURATION_SEC = 4.0  # matches the dataset's known cue+imagery duration


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
            snapshot = np.array(self.buffer, dtype=np.float32)
        return snapshot.T


def eeg_receiver_loop(inlet, channel_indices, buffer, stop_event):
    while not stop_event.is_set():
        sample, ts = inlet.pull_sample(timeout=0.5)
        if sample is None:
            continue
        arr = np.array(sample, dtype=np.float32)
        buffer.append(arr[channel_indices] * VOLTS_TO_MICROVOLTS)


def marker_receiver_loop(inlet, latest_marker_holder, stop_event):
    while not stop_event.is_set():
        sample, ts = inlet.pull_sample(timeout=0.5)
        if sample is not None:
            latest_marker_holder["label"] = sample[0]
            latest_marker_holder["onset_time"] = time.time()  # record WHEN this cue started


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def main():
    print("Resolving LSL streams (5s timeout)...")
    streams = resolve_streams(wait_time=5.0)

    eeg_info, marker_info = None, None
    for s in streams:
        if s.type() == "EEG" and s.name() == "SimulatedEEG":
            eeg_info = s
        elif s.type() == "Markers":
            marker_info = s

    if eeg_info is None or marker_info is None:
        print("Streams not found. Is lsl_outlet.py running?")
        return

    eeg_inlet = StreamInlet(eeg_info)
    marker_inlet = StreamInlet(marker_info)

    all_channels = []
    sfreq = None
    for attempt in range(10):
        full_info = eeg_inlet.info()
        sfreq = full_info.nominal_srate()
        ch = full_info.desc().child("channels").child("channel")
        candidate_channels = []
        while not ch.empty():
            candidate_channels.append(ch.child_value("label"))
            ch = ch.next_sibling("channel")
        if len(candidate_channels) > 0:
            all_channels = candidate_channels
            break
        print(f"  Channel metadata not yet available, retrying ({attempt+1}/10)...")
        time.sleep(0.5)

    if not all_channels:
        print("Failed to retrieve channel metadata after retries. Aborting.")
        return

    print(f"Stream channels ({len(all_channels)}): {all_channels}")
    channel_indices = [all_channels.index(c) for c in MODEL_CHANNELS]

    nyquist = sfreq / 2.0
    low = FILTER_LOW_HZ / nyquist
    high = FILTER_HIGH_HZ / nyquist
    b, a = butter(FILTER_ORDER, [low, high], btype="band")
    print(f"Bandpass filter designed: {FILTER_LOW_HZ}-{FILTER_HIGH_HZ}Hz, order={FILTER_ORDER}")

    onnx_path = "../../models/eegnet_subject1.onnx"
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    buffer = RollingBuffer(window_seconds=4.0, sfreq=sfreq)
    latest_marker = {"label": None, "onset_time": None}
    stop_event = threading.Event()

    eeg_thread = threading.Thread(target=eeg_receiver_loop, args=(eeg_inlet, channel_indices, buffer, stop_event), daemon=True)
    marker_thread = threading.Thread(target=marker_receiver_loop, args=(marker_inlet, latest_marker, stop_event), daemon=True)
    eeg_thread.start()
    marker_thread.start()

    print("Threads started. Waiting for buffer to fill (4s)...\n")
    print(f"{'Time':<8}{'GroundTruth':<14}{'Predicted':<14}{'Confidence':<12}{'Scored':<8}{'Match':<6}")
    print("-" * 63)

    n_predictions, n_correct = 0, 0
    n_skipped_rest = 0

    try:
        while True:
            time.sleep(0.25)
            window = buffer.get_window()
            if window is None:
                continue

            filtered_window = filtfilt(b, a, window, axis=1)
            model_input = filtered_window[np.newaxis, np.newaxis, :, :].astype(np.float32)

            if model_input.shape[-1] != 1001:
                pad_width = 1001 - model_input.shape[-1]
                model_input = np.pad(model_input, ((0,0),(0,0),(0,0),(0,pad_width)), mode="edge")

            logits = session.run(None, {input_name: model_input})[0][0]
            probs = softmax(logits)
            pred_idx = int(np.argmax(probs))
            pred_label = IDX_TO_LABEL[pred_idx]
            confidence = probs[pred_idx]

            gt_label = latest_marker["label"]
            onset_time = latest_marker["onset_time"]
            now = time.time()

            # Only score if we're within the valid [onset, onset+4s] imagery window
            in_valid_window = (
                gt_label is not None
                and onset_time is not None
                and onset_time <= now <= onset_time + TRIAL_DURATION_SEC
            )

            if in_valid_window:
                n_predictions += 1
                is_correct = (gt_label == pred_label)
                if is_correct:
                    n_correct += 1
                scored_str = "YES"
                is_match = "✓" if is_correct else " "
            else:
                n_skipped_rest += 1
                scored_str = "rest"
                is_match = " "

            display_gt = gt_label if gt_label else "—"
            elapsed = now
            print(f"{elapsed % 1000:<8.1f}{display_gt:<14}{pred_label:<14}{confidence:<12.3f}{scored_str:<8}{is_match:<6}")

    except KeyboardInterrupt:
        stop_event.set()
        print(f"\n\n--- TRIAL-ALIGNED FILTERED RESULTS ---")
        print(f"Predictions scored (within valid 4s imagery window): {n_predictions}")
        print(f"Predictions skipped (rest period / stale marker)   : {n_skipped_rest}")
        print(f"Correct matches: {n_correct}")
        if n_predictions > 0:
            print(f"Trial-aligned accuracy: {n_correct/n_predictions:.3f}")


if __name__ == "__main__":
    main()
