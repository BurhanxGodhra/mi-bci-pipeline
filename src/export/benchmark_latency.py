"""
Step 3.3: Benchmark single-trial inference latency across backends.
"""
import sys
import os
import time
import numpy as np
import torch
import onnxruntime as ort

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
from eegnet import EEGNet

N_WARMUP = 20
N_ITERS = 200


def summarize(times_ms: np.ndarray, label: str):
    print(f"\n--- {label} ---")
    print(f"Mean   : {times_ms.mean():.3f} ms")
    print(f"Median : {np.median(times_ms):.3f} ms")
    print(f"P95    : {np.percentile(times_ms, 95):.3f} ms")
    print(f"P99    : {np.percentile(times_ms, 99):.3f} ms")
    print(f"Min/Max: {times_ms.min():.3f} / {times_ms.max():.3f} ms")


def benchmark_pytorch(model, dummy_input, device):
    model = model.to(device)
    x = dummy_input.to(device)
    model.eval()

    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = model(x)
        if device.type == "mps":
            torch.mps.synchronize()

        times = []
        for _ in range(N_ITERS):
            start = time.perf_counter()
            _ = model(x)
            if device.type == "mps":
                torch.mps.synchronize()  # MPS is async — must sync before stopping the clock
            end = time.perf_counter()
            times.append((end - start) * 1000)

    return np.array(times)


def benchmark_onnx(onnx_path, dummy_input_np, provider):
    session = ort.InferenceSession(onnx_path, providers=[provider])
    input_name = session.get_inputs()[0].name

    for _ in range(N_WARMUP):
        _ = session.run(None, {input_name: dummy_input_np})

    times = []
    for _ in range(N_ITERS):
        start = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input_np})
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return np.array(times)


def main():
    checkpoint_path = "../../models/eegnet_subject1.pt"
    onnx_path = "../../models/eegnet_subject1.onnx"

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    n_channels = checkpoint["n_channels"]
    n_timepoints = checkpoint["n_timepoints"]
    n_classes = len(checkpoint["label_map"])

    model = EEGNet(n_classes=n_classes, n_channels=n_channels, n_timepoints=n_timepoints)
    model.load_state_dict(checkpoint["model_state_dict"])

    dummy_input = torch.randn(1, 1, n_channels, n_timepoints)  # single trial, batch=1
    dummy_input_np = dummy_input.numpy().astype(np.float32)

    print(f"Benchmarking single-trial inference ({N_ITERS} iterations, {N_WARMUP} warm-up)")
    print(f"Input shape: {dummy_input.shape}")

    # --- PyTorch CPU ---
    cpu_times = benchmark_pytorch(model, dummy_input, torch.device("cpu"))
    summarize(cpu_times, "PyTorch (CPU)")

    # --- PyTorch MPS ---
    if torch.backends.mps.is_available():
        mps_times = benchmark_pytorch(model, dummy_input, torch.device("mps"))
        summarize(mps_times, "PyTorch (MPS)")

    # --- ONNX Runtime CPU ---
    onnx_cpu_times = benchmark_onnx(onnx_path, dummy_input_np, "CPUExecutionProvider")
    summarize(onnx_cpu_times, "ONNX Runtime (CPU)")

    # --- Summary table ---
    print("\n" + "=" * 55)
    print(f"{'Backend':<25}{'Mean (ms)':<12}{'P95 (ms)':<12}")
    print("=" * 55)
    print(f"{'PyTorch (CPU)':<25}{cpu_times.mean():<12.3f}{np.percentile(cpu_times, 95):<12.3f}")
    if torch.backends.mps.is_available():
        print(f"{'PyTorch (MPS)':<25}{mps_times.mean():<12.3f}{np.percentile(mps_times, 95):<12.3f}")
    print(f"{'ONNX Runtime (CPU)':<25}{onnx_cpu_times.mean():<12.3f}{np.percentile(onnx_cpu_times, 95):<12.3f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
