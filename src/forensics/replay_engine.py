import re
from pathlib import Path

import numpy as np
from scipy.signal import filtfilt
import onnxruntime as ort

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.common.bci_utils import BUFFER_LEAD_MARGIN_SEC, design_bandpass_filter, softmax, ensure_onnx_model, MODELS_DIR

LABEL_MAP_REVERSE = {0: 'feet', 1: 'left_hand', 2: 'right_hand', 3: 'tongue'}


def parse_subject_from_filename(path):
    match = re.search(r"_S(\d+)_", Path(path).name)
    return int(match.group(1)) if match else None


class ReplayEngine:
    def __init__(self, data_path, sfreq=250):
        self.data = np.load(data_path)
        self.sfreq = sfreq
        self.total_samples = self.data.shape[1]
        self.total_seconds = self.total_samples / sfreq
        self.b, self.a = design_bandpass_filter(sfreq)

        subject_id = parse_subject_from_filename(data_path)
        model_path = ensure_onnx_model(subject_id) if subject_id is not None else None
        if model_path is None:
            model_path = str(MODELS_DIR / "eegnet_subject3.onnx")
            print(f"WARNING: could not detect subject from '{data_path}', using fallback model {model_path}")

        self.subject_id = subject_id
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def get_window_at_index(self, index):
        window_len = int((4.0 + BUFFER_LEAD_MARGIN_SEC) * self.sfreq)
        start = index - window_len
        return None if start < 0 else self.data[:, start:index]

    def step(self, index):
        window = self.get_window_at_index(index)
        if window is None or window.shape[1] < 1001:
            return None, None, None, index / self.sfreq
        filtered_full = filtfilt(self.b, self.a, window, axis=1)
        lead_samples = int(BUFFER_LEAD_MARGIN_SEC * self.sfreq)
        filt_win = filtered_full[:, lead_samples:]
        if filt_win.shape[1] < 1001:
            filt_win = np.pad(filt_win, ((0, 0), (0, 1001 - filt_win.shape[1])), mode='edge')
        elif filt_win.shape[1] > 1001:
            filt_win = filt_win[:, :1001]
        model_input = filt_win[np.newaxis, np.newaxis, :, :].astype(np.float32)
        logits = self.session.run(None, {self.input_name: model_input})[0][0]
        probs = softmax(logits)
        pred_idx = int(np.argmax(probs))
        return LABEL_MAP_REVERSE[pred_idx], probs[pred_idx], filt_win, index / self.sfreq
