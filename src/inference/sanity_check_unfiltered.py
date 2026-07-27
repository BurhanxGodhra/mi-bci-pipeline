"""
Step 5.1b: Sanity check - unfiltered rolling window fed directly to ONNX model.
Compares predictions against known ground-truth markers to establish a
BEFORE-filtering baseline accuracy number.
"""
import threading
import time
from collections import deque

import numpy as np
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

    # Retry fetching channel metadata — inlet.info() can occasionally return
    # before the outlet's desc() metadata has fully propagated over the network
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

    # Load ONNX model
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
    print(f"{'Time':<8}{'GroundTruth':<14}{'Predicted':<14}{'Confidence':<12}{'Match':<6}")
    print("-" * 55)

    n_predictions, n_correct = 0, 0

    try:
        while True:
            time.sleep(0.25)
            window = buffer.get_window()
            if window is None:
                continue

            # NO FILTERING - raw window straight to model
            model_input = window[np.newaxis, np.newaxis, :, :].astype(np.float32)  # (1,1,22,1000)

            # Pad/trim to 1001 timepoints to match training shape exactly
            if model_input.shape[-1] != 1001:
                pad_width = 1001 - model_input.shape[-1]
                model_input = np.pad(model_input, ((0,0),(0,0),(0,0),(0,pad_width)), mode="edge")

            logits = session.run(None, {input_name: model_input})[0][0]
            probs = softmax(logits)
            pred_idx = int(np.argmax(probs))
            pred_label = IDX_TO_LABEL[pred_idx]
            confidence = probs[pred_idx]

            gt_label = latest_marker["label"] if latest_marker["label"] else "—"
            is_match = "✓" if gt_label == pred_label else " "

            if gt_label != "—":
                n_predictions += 1
                if gt_label == pred_label:
                    n_correct += 1

            elapsed = time.time()
            print(f"{elapsed % 1000:<8.1f}{gt_label:<14}{pred_label:<14}{confidence:<12.3f}{is_match:<6}")

    except KeyboardInterrupt:
        stop_event.set()
        print(f"\n\n--- UNFILTERED BASELINE RESULTS ---")
        print(f"Total predictions vs ground truth: {n_predictions}")
        print(f"Correct matches: {n_correct}")
        if n_predictions > 0:
            print(f"Running accuracy: {n_correct/n_predictions:.3f}")


if __name__ == "__main__":
    main()
