"""
Step 4.2 & 4.3: Stream continuous EEG over LSL with playback speed control.
"""
import argparse
import time
import numpy as np
import mne
from moabb.datasets import BNCI2014_001
from pylsl import StreamInfo, StreamOutlet


def load_continuous_data(subject_id: int = 1):
    dataset = BNCI2014_001()
    data = dataset.get_data(subjects=[subject_id])
    subject_data = data[subject_id]

    all_raws = []
    for session_id, runs in subject_data.items():
        for run_id, raw in runs.items():
            all_raws.append(raw)

    raw_concat = mne.concatenate_raws(all_raws)

    events = [
        (ann["onset"], ann["description"])
        for ann in raw_concat.annotations
        if "BAD" not in ann["description"] and "boundary" not in ann["description"].lower()
    ]

    eeg_array = raw_concat.get_data()
    ch_names = raw_concat.ch_names
    sfreq = raw_concat.info["sfreq"]

    return eeg_array, ch_names, sfreq, events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1, help="Subject ID (1-9) whose recording to stream")

    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (e.g. 10 = 10x faster than real-time)")
    args = parser.parse_args()

    print(f"Streaming subject {args.subject} at {args.speed}x speed")
 
    print("Loading and concatenating continuous EEG data...")
    eeg_array, ch_names, sfreq, events = load_continuous_data(subject_id=args.subject)
    n_channels, n_samples = eeg_array.shape

    print(f"Loaded: {n_channels} channels, {n_samples} samples @ {sfreq}Hz "
          f"({n_samples/sfreq/60:.1f} min), {len(events)} clean events")
    print(f"Playback speed: {args.speed}x")

    eeg_info = StreamInfo(
        name=f"SimulatedEEG_S{args.subject}",
        type="EEG",
        channel_count=n_channels,
        nominal_srate=sfreq,
        channel_format="float32",
        source_id=f"bci_iv_2a_subject{args.subject}_sim",
    )
    chns = eeg_info.desc().append_child("channels")
    for ch in ch_names:
        chns.append_child("channel").append_child_value("label", ch)
    eeg_outlet = StreamOutlet(eeg_info)

    marker_info = StreamInfo(
        name=f"SimulatedMarkers_S{args.subject}",
        type="Markers",
        channel_count=1,
        nominal_srate=0,
        channel_format="string",
        source_id=f"bci_iv_2a_subject{args.subject}_sim_markers",
    )
    marker_outlet = StreamOutlet(marker_info)

    print("\nLSL outlets created: 'SimulatedEEG' and 'SimulatedMarkers'")
    print("Waiting 3s for consumers to connect...")
    time.sleep(3)

    print("Streaming started. Press Ctrl+C to stop.\n")

    sample_period = 1.0 / sfreq / args.speed
    event_idx = 0
    start_time = time.perf_counter()

    try:
        for i in range(n_samples):
            target_time = start_time + i * sample_period
            now = time.perf_counter()
            if target_time > now:
                time.sleep(target_time - now)

            sample = eeg_array[:, i].tolist()
            eeg_outlet.push_sample(sample)

            current_t = i / sfreq
            if event_idx < len(events) and events[event_idx][0] <= current_t:
                onset, label = events[event_idx]
                marker_outlet.push_sample([label])
                print(f"[t={current_t:7.2f}s] Marker pushed: '{label}'")
                event_idx += 1

            if i % (int(sfreq) * 30) == 0 and i > 0:
                print(f"  ...streamed {i/sfreq:.0f}s / {n_samples/sfreq:.0f}s")

    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")

    print("Streaming complete.")


if __name__ == "__main__":
    main()
