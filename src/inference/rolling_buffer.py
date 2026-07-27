"""
Step 5.1: Thread-safe rolling buffer fed by a background LSL receiver thread.
This does NOT do filtering or inference yet — just proves reliable continuous
buffering of the last N seconds of EEG.
"""
import threading
import time
from collections import deque

import numpy as np
from pylsl import resolve_streams, StreamInlet


# The 22 EEG channels our model expects (excludes EOG1-3 and STI from the 26-channel stream)
MODEL_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]

VOLTS_TO_MICROVOLTS = 1e6  # MNE gives Volts; models/filters conventionally expect µV


class RollingBuffer:
    def __init__(self, window_seconds: float, sfreq: float, n_channels: int):
        self.maxlen = int(window_seconds * sfreq)
        self.sfreq = sfreq
        self.n_channels = n_channels
        self.buffer = deque(maxlen=self.maxlen)
        self.lock = threading.Lock()
        self.n_samples_received = 0

    def append(self, sample: np.ndarray):
        with self.lock:
            self.buffer.append(sample)
            self.n_samples_received += 1

    def is_full(self) -> bool:
        with self.lock:
            return len(self.buffer) == self.maxlen

    def get_window(self) -> np.ndarray | None:
        """Returns shape (n_channels, n_samples) snapshot, or None if not yet full."""
        with self.lock:
            if len(self.buffer) < self.maxlen:
                return None
            # Copy while holding the lock so the snapshot is consistent
            snapshot = np.array(self.buffer, dtype=np.float32)  # (n_samples, n_channels)
        return snapshot.T  # -> (n_channels, n_samples)


def find_channel_indices(all_channel_labels: list[str], wanted_labels: list[str]) -> list[int]:
    return [all_channel_labels.index(ch) for ch in wanted_labels]


def receiver_loop(inlet: StreamInlet, channel_indices: list[int], buffer: RollingBuffer, stop_event: threading.Event):
    """Background thread: continuously pull samples, select channels, scale, buffer them."""
    while not stop_event.is_set():
        sample, ts = inlet.pull_sample(timeout=0.5)
        if sample is None:
            continue
        sample_arr = np.array(sample, dtype=np.float32)
        selected = sample_arr[channel_indices] * VOLTS_TO_MICROVOLTS
        buffer.append(selected)


def main():
    print("Resolving LSL streams (5s timeout)...")
    streams = resolve_streams(wait_time=5.0)

    eeg_stream_info = None
    for s in streams:
        if s.type() == "EEG" and s.name() == "SimulatedEEG":
            eeg_stream_info = s
            break

    if eeg_stream_info is None:
        print("SimulatedEEG stream not found. Is lsl_outlet.py running?")
        return

    inlet = StreamInlet(eeg_stream_info)
    sfreq = eeg_stream_info.nominal_srate()

    # Pull channel labels from the stream's embedded metadata
    info = inlet.info()
    all_channels = []
    ch = info.desc().child("channels").child("channel")
    while not ch.empty():
        all_channels.append(ch.child_value("label"))
        ch = ch.next_sibling("channel")

    print(f"Stream channels ({len(all_channels)}): {all_channels}")
    channel_indices = find_channel_indices(all_channels, MODEL_CHANNELS)
    print(f"Selected {len(channel_indices)} model channels at indices: {channel_indices}")

    buffer = RollingBuffer(window_seconds=4.0, sfreq=sfreq, n_channels=len(MODEL_CHANNELS))
    stop_event = threading.Event()

    receiver_thread = threading.Thread(
        target=receiver_loop, args=(inlet, channel_indices, buffer, stop_event), daemon=True
    )
    receiver_thread.start()
    print("\nBackground receiver thread started. Waiting for buffer to fill (4s)...")

    try:
        # Poll every 250ms, print window stats once buffer is full
        while True:
            time.sleep(0.25)
            window = buffer.get_window()
            if window is None:
                print(f"  Buffering... {buffer.n_samples_received} samples received so far")
                continue

            print(f"Window ready | shape={window.shape} | "
                  f"mean={window.mean():.2f}µV | std={window.std():.2f}µV | "
                  f"total_received={buffer.n_samples_received}")

    except KeyboardInterrupt:
        print("\nStopping receiver thread...")
        stop_event.set()
        receiver_thread.join(timeout=2)
        print("Stopped cleanly.")


if __name__ == "__main__":
    main()
