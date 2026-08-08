"""
Step 5.6 (final, v3): Trial-aligned evaluation with leading-margin filter fix.

- Uses CONFIRMED dataset interval (2, 6) instead of the wrongly-assumed (0, 4)
- Buffer holds an extra BUFFER_LEAD_MARGIN_SEC of history so filtfilt has real
  (non-padded) data to settle on before the region we actually feed to the model
- Does NOT use a trailing margin (tested in v4, inconclusive/noisy, not adopted)

Final documented results (subject 1, --speed 1):
  Full window [2-6s]: 61.3% (152/248)
  Near-end (last 1s):  65.1% (41/63)
"""
import argparse
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

# CONFIRMED from dataset.interval (BNCI2014_001)
INTERVAL_START_SEC = 2.0
INTERVAL_END_SEC = 6.0
NEAR_END_WINDOW_SEC = 1.0  # last 1s of the interval = tightest trailing-buffer alignment

BUFFER_LEAD_MARGIN_SEC = 1.0  # extra history for filter settling, cropped after filtering


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
        return snapshot.T  # (channels, samples)


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
            latest_marker_holder["onset_time"] = time.time()


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=1.0,
                         help="Must match the --speed used in lsl_outlet.py")
    args = parser.parse_args()

    effective_interval_start = INTERVAL_START_SEC / args.speed
    effective_interval_end = INTERVAL_END_SEC / args.speed
    effective_near_end_start = effective_interval_end - (NEAR_END_WINDOW_SEC / args.speed)

    print(f"Playback speed: {args.speed}x")
    print(f"Valid scoring window (wall-clock, relative to onset): "
          f"[{effective_interval_start:.2f}s, {effective_interval_end:.2f}s]")
    print(f"Near-end tight window: [{effective_near_end_start:.2f}s, {effective_interval_end:.2f}s]\n")

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

    # Retry fetching channel metadata (race-condition-safe pattern)
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
    print(f"Selected {len(channel_indices)} model channels at indices: {channel_indices}")

    nyquist = sfreq / 2.0
    low = FILTER_LOW_HZ / nyquist
    high = FILTER_HIGH_HZ / nyquist
    b, a = butter(FILTER_ORDER, [low, high], btype="band")
    print(f"Bandpass filter designed: {FILTER_LOW_HZ}-{FILTER_HIGH_HZ}Hz, order={FILTER_ORDER}")
    print(f"Leading buffer margin for filter settling: {BUFFER_LEAD_MARGIN_SEC}s")

    onnx_path = "../../models/eegnet_subject3.onnx"
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Buffer holds 4s target window + lead margin for filter settling
    buffer = RollingBuffer(window_seconds=4.0 + BUFFER_LEAD_MARGIN_SEC, sfreq=sfreq)
    latest_marker = {"label": None, "onset_time": None}
    stop_event = threading.Event()

    eeg_thread = threading.Thread(target=eeg_receiver_loop, args=(eeg_inlet, channel_indices, buffer, stop_event), daemon=True)
    marker_thread = threading.Thread(target=marker_receiver_loop, args=(marker_inlet, latest_marker, stop_event), daemon=True)
    eeg_thread.start()
    marker_thread.start()

    print("Threads started. Waiting for buffer to fill...\n")
    print(f"{'Time':<8}{'GroundTruth':<14}{'Predicted':<14}{'Conf':<8}{'Zone':<10}{'Match':<6}")
    print("-" * 55)

    n_full, n_full_correct = 0, 0
    n_near_end, n_near_end_correct = 0, 0

    try:
        while True:
            time.sleep(0.25)
            window = buffer.get_window()
            if window is None:
                continue

            # Filter the FULL buffer (target window + lead margin), then crop
            # off the margin -- this gives filtfilt real, unpadded data to
            # settle on before the region we actually feed to the model.
            filtered_full = filtfilt(b, a, window, axis=1)
            lead_samples = int(BUFFER_LEAD_MARGIN_SEC * sfreq)
            filtered_window = filtered_full[:, lead_samples:]  # drop distorted front edge

            model_input = filtered_window[np.newaxis, np.newaxis, :, :].astype(np.float32)

            if model_input.shape[-1] != 1001:
                pad_width = 1001 - model_input.shape[-1]
                model_input = np.pad(model_input, ((0, 0), (0, 0), (0, 0), (0, pad_width)), mode="edge")

            logits = session.run(None, {input_name: model_input})[0][0]
            probs = softmax(logits)
            pred_idx = int(np.argmax(probs))
            pred_label = IDX_TO_LABEL[pred_idx]
            confidence = probs[pred_idx]

            gt_label = latest_marker["label"]
            onset_time = latest_marker["onset_time"]
            now = time.time()

            zone = "—"
            is_match = " "

            if gt_label is not None and onset_time is not None:
                elapsed_since_onset = now - onset_time

                in_full_window = effective_interval_start <= elapsed_since_onset <= effective_interval_end
                in_near_end = effective_near_end_start <= elapsed_since_onset <= effective_interval_end

                if in_full_window:
                    n_full += 1
                    is_correct = (gt_label == pred_label)
                    if is_correct:
                        n_full_correct += 1
                    is_match = "✓" if is_correct else " "
                    zone = "full"

                    if in_near_end:
                        n_near_end += 1
                        if is_correct:
                            n_near_end_correct += 1
                        zone = "NEAR-END"
                else:
                    zone = "rest/early"

            display_gt = gt_label if gt_label else "—"
            print(f"{now % 1000:<8.1f}{display_gt:<14}{pred_label:<14}{confidence:<8.3f}{zone:<10}{is_match:<6}")

    except KeyboardInterrupt:
        stop_event.set()
        print(f"\n\n--- CORRECTED TRIAL-ALIGNED RESULTS (interval={INTERVAL_START_SEC}-{INTERVAL_END_SEC}s) ---")
        if n_full > 0:
            print(f"Full window [{INTERVAL_START_SEC}-{INTERVAL_END_SEC}s]: {n_full_correct}/{n_full} = {n_full_correct/n_full:.3f}")
        else:
            print("Full window: no samples")
        if n_near_end > 0:
            print(f"Near-end window (tightest alignment, last {NEAR_END_WINDOW_SEC}s): "
                  f"{n_near_end_correct}/{n_near_end} = {n_near_end_correct/n_near_end:.3f}")
        else:
            print("Near-end: no samples")


if __name__ == "__main__":
    main()
