"""
Record a real-time (1x speed, always) EEG + markers session to .npy files.
Manages its own lsl_outlet.py subprocess -- no manual terminal needed.
Speed is intentionally fixed at 1x: a recording should represent what a
real headset would actually capture, and marker/sample timestamps only
share a common time scale when playback is real-time.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from pylsl import resolve_streams, StreamInlet

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.common.bci_utils import MODEL_CHANNELS, VOLTS_TO_MICROVOLTS, DATA_PROCESSED_DIR, STREAMING_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()

    eeg_stream_name = f"SimulatedEEG_S{args.subject}"
    marker_stream_name = f"SimulatedMarkers_S{args.subject}"

    print(f"Launching outlet for subject {args.subject} at 1x (real-time) speed...")
    outlet_proc = subprocess.Popen(
        [sys.executable, "lsl_outlet.py", "--subject", str(args.subject), "--speed", "1"],
        cwd=str(STREAMING_DIR),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    try:
        time.sleep(3.0)  # let the outlet finish loading data before we search for it
        print(f"Resolving '{eeg_stream_name}' / '{marker_stream_name}'...")
        streams = resolve_streams(wait_time=8.0)
        eeg_info, marker_info = None, None
        for s in streams:
            if s.type() == "EEG" and s.name() == eeg_stream_name:
                eeg_info = s
            elif s.type() == "Markers" and s.name() == marker_stream_name:
                marker_info = s

        if eeg_info is None or marker_info is None:
            print("Could not find the expected streams. Aborting.")
            return

        eeg_inlet = StreamInlet(eeg_info)
        marker_inlet = StreamInlet(marker_info)

        full_info = eeg_inlet.info()
        all_ch = []
        ch = full_info.desc().child("channels").child("channel")
        while not ch.empty():
            all_ch.append(ch.child_value("label"))
            ch = ch.next_sibling("channel")
        ch_indices = [all_ch.index(c) for c in MODEL_CHANNELS]

        print(f"Recording subject {args.subject} for {args.duration:.0f}s (real-time)...")
        recorded_eeg, recorded_markers = [], []
        recording_start_time = None

        start_wallclock = time.time()
        while time.time() - start_wallclock < args.duration:
            sample, ts = eeg_inlet.pull_sample(timeout=0.1)
            if sample is not None:
                if recording_start_time is None:
                    recording_start_time = ts
                arr = np.array(sample, dtype=np.float32)
                recorded_eeg.append(arr[ch_indices] * VOLTS_TO_MICROVOLTS)

            marker_sample, marker_ts = marker_inlet.pull_sample(timeout=0.0)
            if marker_sample is not None and recording_start_time is not None:
                relative_ts = marker_ts - recording_start_time
                recorded_markers.append((relative_ts, marker_sample[0]))
                print(f"Marker: {marker_sample[0]} at t={relative_ts:.2f}s")

    finally:
        print("Stopping outlet subprocess...")
        outlet_proc.terminate()
        try:
            outlet_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            outlet_proc.kill()

    if not recorded_eeg:
        print("No EEG samples collected.")
        return

    eeg_data = np.array(recorded_eeg, dtype=np.float32).T
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    session_id = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"replay_S{args.subject}_{session_id}"
    np.save(DATA_PROCESSED_DIR / f"{base_name}.npy", eeg_data)
    print(f"Saved {eeg_data.shape[1]} EEG samples ({base_name}.npy)")

    if recorded_markers:
        markers = np.array(recorded_markers, dtype=[('timestamp', 'f8'), ('label', 'U20')])
        np.save(DATA_PROCESSED_DIR / f"{base_name}_markers.npy", markers)
        print(f"Saved {len(markers)} markers ({base_name}_markers.npy)")


if __name__ == "__main__":
    main()
