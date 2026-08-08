"""
Step 2.1: Download BCI Competition IV-2a and inspect its structure.
This does NOT train anything yet — just fetches data and prints shapes.
"""
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

def main():
    # BNCI2014_001 is MOABB's identifier for BCI Competition IV-2a
    dataset = BNCI2014_001()

    # MotorImagery paradigm handles epoch extraction, filtering band, resampling
    paradigm = MotorImagery(
        n_classes=4,        # left_hand, right_hand, feet, tongue
        fmin=4, fmax=38,    # standard mu/beta band for motor imagery
        resample=250        # dataset's native sampling rate
    )

    print("Downloading/loading Subject 1 data (this may take a few minutes on first run)...")
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[1])

    print("\n--- Dataset Shape Report ---")
    print(f"X (EEG epochs) shape : {X.shape}   # (n_trials, n_channels, n_timepoints)")
    print(f"y (labels) shape     : {y.shape}")
    print(f"Unique classes       : {set(y)}")
    print(f"Metadata columns     : {list(metadata.columns)}")
    print(f"Sessions in metadata : {metadata['session'].unique()}")
    print(f"Runs in metadata     : {metadata['run'].unique()}")

if __name__ == "__main__":
    main()


