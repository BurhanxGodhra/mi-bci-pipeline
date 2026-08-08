"""
Step 4.3: Minimal LSL inlet — verifies the outlet is actually receivable.
Run this WHILE lsl_outlet.py is streaming, in a separate terminal.
"""
import time
from pylsl import resolve_streams, StreamInlet


def main():
    print("Resolving available LSL streams (5s timeout)...")
    streams = resolve_streams(wait_time=5.0)

    if not streams:
        print("No LSL streams found. Is lsl_outlet.py running?")
        return

    print(f"\nFound {len(streams)} stream(s):")
    eeg_inlet, marker_inlet = None, None
    for s in streams:
        print(f"  - name='{s.name()}' type='{s.type()}' channels={s.channel_count()} "
              f"srate={s.nominal_srate()}")
        if s.type() == "EEG":
            eeg_inlet = StreamInlet(s)
        elif s.type() == "Markers":
            marker_inlet = StreamInlet(s)

    if eeg_inlet is None or marker_inlet is None:
        print("Could not find both EEG and Markers streams.")
        return

    print("\nPulling samples for 15 seconds (Ctrl+C to stop early)...\n")
    n_eeg_samples = 0

    start = time.time()
    try:
        while time.time() - start < 15:
            eeg_sample, eeg_ts = eeg_inlet.pull_sample(timeout=0.1)
            if eeg_sample is not None:
                n_eeg_samples += 1
                if n_eeg_samples % 250 == 0:
                    print(f"  EEG sample #{n_eeg_samples} | ts={eeg_ts:.2f} | "
                          f"first 3 channels: {[round(v, 4) for v in eeg_sample[:3]]}")

            marker_sample, marker_ts = marker_inlet.pull_sample(timeout=0.0)
            if marker_sample is not None:
                print(f"  >>> MARKER RECEIVED: '{marker_sample[0]}' at ts={marker_ts:.2f}")
    except KeyboardInterrupt:
        pass

    print(f"\nTotal EEG samples received: {n_eeg_samples}")
    print(f"Effective received rate: {n_eeg_samples/15:.1f} Hz (expected ~250Hz at 1x speed)")


if __name__ == "__main__":
    main()
