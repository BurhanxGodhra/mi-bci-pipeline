"""
Single source of truth for constants and helpers shared across the live
dashboard, the forensics replay tool, and the underlying inference pipeline.
"""
from pathlib import Path

import numpy as np
from scipy.signal import butter

COMMON_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = COMMON_DIR.parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
TRAINING_DIR = PROJECT_ROOT / "src" / "training"
STREAMING_DIR = PROJECT_ROOT / "src" / "streaming"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOGS_DIR = PROJECT_ROOT / "logs"

MODEL_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]
VOLTS_TO_MICROVOLTS = 1e6
LABEL_MAP = {"feet": 0, "left_hand": 1, "right_hand": 2, "tongue": 3}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

FILTER_LOW_HZ = 4.0
FILTER_HIGH_HZ = 38.0
FILTER_ORDER = 4
BUFFER_LEAD_MARGIN_SEC = 1.0

INTERVAL_START_SEC = 2.0
INTERVAL_END_SEC = 6.0
NEAR_END_WINDOW_SEC = 1.0

WAVEFORM_CHANNELS = ["C3", "Cz", "C4"]
WAVEFORM_CHANNEL_INDICES = [MODEL_CHANNELS.index(c) for c in WAVEFORM_CHANNELS]

INTENT_STYLE = {
    "left_hand":  {"label": "LEFT HAND",  "color": "#4F8DFF"},
    "right_hand": {"label": "RIGHT HAND", "color": "#B07CFF"},
    "feet":       {"label": "FEET",       "color": "#37D399"},
    "tongue":     {"label": "TONGUE",     "color": "#FFB454"},
    None:         {"label": "—",          "color": "#5A6270"},
}


def design_bandpass_filter(sfreq: float):
    nyquist = sfreq / 2.0
    return butter(FILTER_ORDER, [FILTER_LOW_HZ / nyquist, FILTER_HIGH_HZ / nyquist], btype="band")


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def ensure_onnx_model(subject_id: int):
    import sys
    import torch

    onnx_path = MODELS_DIR / f"eegnet_subject{subject_id}.onnx"
    if onnx_path.exists():
        return str(onnx_path)

    checkpoint_path = MODELS_DIR / f"eegnet_subject{subject_id}.pt"
    if not checkpoint_path.exists():
        return None

    sys.path.append(str(TRAINING_DIR))
    from eegnet import EEGNet

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    n_channels = checkpoint["n_channels"]
    n_timepoints = checkpoint["n_timepoints"]
    n_classes = len(checkpoint["label_map"])

    model = EEGNet(n_classes=n_classes, n_channels=n_channels, n_timepoints=n_timepoints)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 1, n_channels, n_timepoints)
    torch.onnx.export(
        model, dummy_input, str(onnx_path),
        input_names=["eeg_input"], output_names=["class_logits"],
        dynamic_axes={"eeg_input": {0: "batch_size"}, "class_logits": {0: "batch_size"}},
        opset_version=18, do_constant_folding=True,
    )
    return str(onnx_path)
