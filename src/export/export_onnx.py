"""
Step 3.1: Export trained EEGNet checkpoint to ONNX format.
"""
import sys
import os
import torch

# Allow importing EEGNet from src/training
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
from eegnet import EEGNet


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject",type=int, default=1)
    args = parser.parse_args()

    checkpoint_path = f"../../models/eegnet_subject{args.subject}.pt"
    onnx_output_path = f"../../models/eegnet_subject{args.subject}.onnx"

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    n_channels = checkpoint["n_channels"]
    n_timepoints = checkpoint["n_timepoints"]
    label_map = checkpoint["label_map"]
    n_classes = len(label_map)

    reported_acc = checkpoint.get('final_test_acc', checkpoint.get('val_acc'))
    if reported_acc is not None:
        print(f"Loaded checkpoint: reported_acc={reported_acc:.3f}")
    else:
        print("Loaded checkpoint (no accuracy field found in this checkpoint)")
    print(f"n_channels={n_channels}, n_timepoints={n_timepoints}, n_classes={n_classes}")

    # Rebuild model architecture and load trained weights
    model = EEGNet(n_classes=n_classes, n_channels=n_channels, n_timepoints=n_timepoints)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # critical: disables dropout, freezes batchnorm running stats

    # Dummy input matching real inference shape: (batch=1, channels=1, electrodes, time)
    dummy_input = torch.randn(1, 1, n_channels, n_timepoints)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        input_names=["eeg_input"],
        output_names=["class_logits"],
        dynamic_axes={
            "eeg_input": {0: "batch_size"},   # allow variable batch size at inference
            "class_logits": {0: "batch_size"},
        },
        opset_version=18,
        do_constant_folding=True,
    )

    print(f"\nONNX model exported to: {onnx_output_path}")

    # Sanity check: verify the exported graph is structurally valid
    import onnx
    onnx_model = onnx.load(onnx_output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure validated successfully.")

    file_size_kb = os.path.getsize(onnx_output_path) / 1024
    print(f"File size: {file_size_kb:.1f} KB")


if __name__ == "__main__":
    main()
