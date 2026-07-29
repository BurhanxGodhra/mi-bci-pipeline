"""
Step 3.2: Verify ONNX Runtime outputs match PyTorch outputs numerically.
"""
import sys
import os
import numpy as np
import torch
import onnxruntime as ort

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
from eegnet import EEGNet
from train import load_split_data  # reuse the same data loader from Phase 2


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    args = parser.parse_args()

    checkpoint_path = f"../../models/eegnet_subject{args.subject}.pt"
    onnx_path = f"../../models/eegnet_subject{args.subject}.onnx"

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    n_channels = checkpoint["n_channels"]
    n_timepoints = checkpoint["n_timepoints"]
    n_classes = len(checkpoint["label_map"])

    # --- Load PyTorch model ---
    torch_model = EEGNet(n_classes=n_classes, n_channels=n_channels, n_timepoints=n_timepoints)
    torch_model.load_state_dict(checkpoint["model_state_dict"])
    torch_model.eval()

    # --- Load ONNX Runtime session ---
    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    # --- Get real test data (subject 1's held-out session) ---
    print("Loading real test trials for verification...")
    _, (X_test, y_test), _ = load_split_data(subject_id=args.subject)

    n_samples_to_check = 20
    X_sample = X_test[:n_samples_to_check]  # (20, 22, 1001)
    X_sample_torch = torch.tensor(X_sample, dtype=torch.float32).unsqueeze(1)  # (20, 1, 22, 1001)

    # --- Run PyTorch inference ---
    with torch.no_grad():
        torch_logits = torch_model(X_sample_torch).numpy()

    # --- Run ONNX Runtime inference ---
    onnx_input = X_sample_torch.numpy().astype(np.float32)
    onnx_logits = ort_session.run([output_name], {input_name: onnx_input})[0]

    # --- Compare ---
    max_abs_diff = np.max(np.abs(torch_logits - onnx_logits))
    all_close = np.allclose(torch_logits, onnx_logits, atol=1e-4, rtol=1e-3)

    torch_preds = torch_logits.argmax(axis=1)
    onnx_preds = onnx_logits.argmax(axis=1)
    preds_match = np.array_equal(torch_preds, onnx_preds)

    print(f"\n--- Verification on {n_samples_to_check} real test trials ---")
    print(f"Max absolute logit difference : {max_abs_diff:.8f}")
    print(f"Logits numerically close (atol=1e-4, rtol=1e-3): {all_close}")
    print(f"Predicted classes match exactly: {preds_match}")
    print(f"PyTorch predictions: {torch_preds}")
    print(f"ONNX predictions   : {onnx_preds}")

    if all_close and preds_match:
        print("\n✅ VERIFICATION PASSED — ONNX model is numerically equivalent to PyTorch model.")
    else:
        print("\n❌ VERIFICATION FAILED — investigate before proceeding.")


if __name__ == "__main__":
    main()
