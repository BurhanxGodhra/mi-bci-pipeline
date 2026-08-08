"""
Step 4.1: Load continuous (non-epoched) raw EEG for LSL streaming prep.
Inspects structure — does NOT stream anything yet.
"""
from moabb.datasets import BNCI2014_001


def main():
    dataset = BNCI2014_001()

    print("Loading raw continuous data for subject 1 (all sessions/runs)...")
    # get_data returns: {subject_id: {session_id: {run_id: raw_object}}}
    data = dataset.get_data(subjects=[1])

    subject_data = data[1]  # subject 1's data
    print(f"\nSessions found: {list(subject_data.keys())}")

    all_raws = []
    for session_id, runs in subject_data.items():
        print(f"\nSession '{session_id}':")
        for run_id, raw in runs.items():
            duration_sec = raw.n_times / raw.info["sfreq"]
            n_events = len(raw.annotations)
            print(f"  Run '{run_id}': {raw.n_times} samples @ {raw.info['sfreq']}Hz "
                  f"= {duration_sec:.1f}s | {n_events} annotated events | "
                  f"{len(raw.ch_names)} channels")
            all_raws.append(raw)

    print(f"\nTotal runs collected: {len(all_raws)}")

    # Inspect one run's annotations closely (event markers = trial onsets)
    sample_raw = all_raws[0]
    print(f"\n--- Sample annotations from first run ---")
    print(f"Channel names: {sample_raw.ch_names}")
    for ann in sample_raw.annotations[:8]:
        print(f"  onset={ann['onset']:.2f}s  duration={ann['duration']:.2f}s  "
              f"description='{ann['description']}'")

    # Concatenate ALL runs across ALL sessions into one continuous recording
    import mne
    raw_concat = mne.concatenate_raws(all_raws)
    total_duration_min = raw_concat.n_times / raw_concat.info["sfreq"] / 60

    print(f"\n--- Concatenated Continuous Recording ---")
    print(f"Total samples : {raw_concat.n_times}")
    print(f"Total duration: {total_duration_min:.1f} minutes")
    print(f"Total events  : {len(raw_concat.annotations)}")
    print(f"Sampling rate : {raw_concat.info['sfreq']} Hz")
    print(f"Channels      : {len(raw_concat.ch_names)}")


if __name__ == "__main__":
    main()
