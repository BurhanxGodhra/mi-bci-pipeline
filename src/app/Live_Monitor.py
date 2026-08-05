"""
Live BCI dashboard - subject switcher + prediction smoothing (EMA over last
5 cycles' probability vectors) for stable, device-ready decisions.
"""
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch
from scipy.signal import filtfilt
import onnxruntime as ort
from pylsl import resolve_streams, StreamInlet

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.common.bci_utils import (
    MODEL_CHANNELS, VOLTS_TO_MICROVOLTS, LABEL_MAP, IDX_TO_LABEL,
    INTERVAL_START_SEC, INTERVAL_END_SEC, NEAR_END_WINDOW_SEC,
    WAVEFORM_CHANNELS, WAVEFORM_CHANNEL_INDICES, INTENT_STYLE,
    design_bandpass_filter, softmax, ensure_onnx_model,
    MODELS_DIR, LOGS_DIR, STREAMING_DIR, BUFFER_LEAD_MARGIN_SEC,
)

WAVEFORM_OFFSET_UV = 60
LATENCY_HISTORY_LEN = 50
SMOOTHING_WINDOW = 5  # NEW: number of recent cycles averaged into the displayed decision

st.set_page_config(page_title="Neural Intent Monitor", layout="wide", page_icon="◆")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0B0D12; }
    #MainMenu, footer, header { visibility: hidden; }
    .panel { background: #12151C; border: 1px solid #1F232D; border-radius: 14px;
             padding: 22px 26px; margin-bottom: 16px; }
    .panel-label { font-size: 12px; font-weight: 600; letter-spacing: 0.08em;
                   color: #6B7280; text-transform: uppercase; margin-bottom: 10px; }
    .hero-value { font-size: 52px; font-weight: 800; letter-spacing: -0.01em; line-height: 1.1; }
    .hero-sub { font-size: 13px; color: #8B92A3; margin-top: 6px; }
    .badge { display: inline-block; padding: 3px 12px; border-radius: 999px;
             font-size: 12px; font-weight: 600; letter-spacing: 0.03em; }
    .stat-value { font-size: 28px; font-weight: 700; color: #E9EBF0; }
    .stat-label { font-size: 12px; color: #6B7280; font-weight: 500; margin-top: 2px; }
    .bar-track { width: 100%; height: 8px; border-radius: 999px; background: #1F232D;
                 margin-top: 10px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 999px; transition: width 0.2s ease; }
    .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                  background: #37D399; margin-right: 8px; box-shadow: 0 0 8px #37D399; }
    .status-dot-off { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                       background: #5A6270; margin-right: 8px; }
    .raw-note { font-size: 12px; color: #5A6270; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)


class RollingBuffer:
    def __init__(self, window_seconds, sfreq):
        self.maxlen = int(window_seconds * sfreq)
        self.buffer = deque(maxlen=self.maxlen)
        self.lock = threading.Lock()
        self.n_received = 0

    def append(self, sample):
        with self.lock:
            self.buffer.append(sample)
            self.n_received += 1

    def get_window(self):
        with self.lock:
            if len(self.buffer) < self.maxlen:
                return None
            snapshot = np.array(self.buffer, dtype=np.float32)
        return snapshot.T


def eeg_receiver_loop(inlet, channel_indices, buffer, stop_event):
    while not stop_event.is_set():
        sample, ts = inlet.pull_sample(timeout=0.5)
        if sample is None:
            continue
        arr = np.array(sample, dtype=np.float32)
        buffer.append(arr[channel_indices] * VOLTS_TO_MICROVOLTS)


def marker_receiver_loop(inlet, latest_marker_holder, stop_event):
    while not stop_event.is_set():
        sample, ts = inlet.pull_sample(timeout=0.5)
        if sample is not None:
            latest_marker_holder["label"] = sample[0]
            latest_marker_holder["onset_time"] = time.time()


def plot_signal_pipeline(raw_window, filtered_window, sfreq):
    n_samples = filtered_window.shape[1]
    time_axis = np.linspace(0, n_samples / sfreq, n_samples)
    raw_aligned = raw_window[:, -n_samples:]

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.6), sharex=True)
    for fig_ax in axes:
        fig_ax.set_facecolor("#12151C")
    fig.patch.set_facecolor("#12151C")

    for i, ch_idx in enumerate(WAVEFORM_CHANNEL_INDICES):
        axes[0].plot(time_axis, raw_aligned[ch_idx, :] + i * WAVEFORM_OFFSET_UV * 1.5,
                     linewidth=0.7, color="#5A6270")
        axes[1].plot(time_axis, filtered_window[ch_idx, :] + i * WAVEFORM_OFFSET_UV,
                     linewidth=0.9, color="#37D399")

    for ax, title, offset in [(axes[0], "RAW SIGNAL", WAVEFORM_OFFSET_UV * 1.5),
                               (axes[1], "CLEANED (4–38Hz BANDPASS)", WAVEFORM_OFFSET_UV)]:
        ax.set_yticks([i * offset for i in range(len(WAVEFORM_CHANNELS))])
        ax.set_yticklabels(WAVEFORM_CHANNELS, color="#6B7280", fontsize=9)
        ax.set_title(title, fontsize=10, color="#8B92A3", loc="left", fontweight=600, pad=8)
        ax.tick_params(colors="#6B7280")
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[1].set_xlabel("Time in window (s)", color="#6B7280", fontsize=9)
    fig.tight_layout()
    return fig


def start_outlet_process(subject_id, speed):
    return subprocess.Popen(
        [sys.executable, "lsl_outlet.py", "--subject", str(subject_id), "--speed", str(speed)],
        cwd=str(STREAMING_DIR),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def stop_outlet_process(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


def initialize_pipeline(subject_id, speed):
    onnx_path = ensure_onnx_model(subject_id)
    if onnx_path is None:
        return None, f"No trained checkpoint found for subject {subject_id}."

    time.sleep(2.0)
    streams = resolve_streams(wait_time=8.0)

    eeg_info, marker_info = None, None
    target_eeg = f"SimulatedEEG_S{subject_id}"
    target_marker = f"SimulatedMarkers_S{subject_id}"
    for s in streams:
        if s.name() == target_eeg and s.type() == "EEG":
            eeg_info = s
        elif s.name() == target_marker and s.type() == "Markers":
            marker_info = s

    if eeg_info is None or marker_info is None:
        return None, f"Could not find streams '{target_eeg}' / '{target_marker}'. Outlet may still be starting."

    eeg_inlet = StreamInlet(eeg_info)
    marker_inlet = StreamInlet(marker_info)

    all_channels = []
    sfreq = None
    for attempt in range(10):
        full_info = eeg_inlet.info()
        sfreq = full_info.nominal_srate()
        ch = full_info.desc().child("channels").child("channel")
        candidate = []
        while not ch.empty():
            candidate.append(ch.child_value("label"))
            ch = ch.next_sibling("channel")
        if candidate:
            all_channels = candidate
            break
        time.sleep(0.5)

    if not all_channels:
        return None, "Failed to retrieve channel metadata."

    channel_indices = [all_channels.index(c) for c in MODEL_CHANNELS]
    b, a = design_bandpass_filter(sfreq)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    buffer = RollingBuffer(window_seconds=4.0 + BUFFER_LEAD_MARGIN_SEC, sfreq=sfreq)
    latest_marker = {"label": None, "onset_time": None}
    stop_event = threading.Event()

    eeg_thread = threading.Thread(
        target=eeg_receiver_loop, args=(eeg_inlet, channel_indices, buffer, stop_event), daemon=True
    )
    marker_thread = threading.Thread(
        target=marker_receiver_loop, args=(marker_inlet, latest_marker, stop_event), daemon=True
    )
    eeg_thread.start()
    marker_thread.start()

    return {
        "buffer": buffer, "latest_marker": latest_marker, "stop_event": stop_event,
        "b": b, "a": a, "sfreq": sfreq,
        "session": session, "input_name": input_name, "subject_id": subject_id,
    }, None


for key, default in [
    ("pipeline", None), ("outlet_process", None), ("current_subject", None),
    ("accuracy", {"n_full": 0, "n_full_correct": 0, "n_near_end": 0, "n_near_end_correct": 0}),
    ("latency_history", deque(maxlen=LATENCY_HISTORY_LEN)),
    ("prob_history", deque(maxlen=SMOOTHING_WINDOW)),  # NEW
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.sidebar.markdown("### ◆ Neural Intent Monitor")
st.sidebar.caption("Real-time motor imagery decoding")
st.sidebar.divider()

selected_subject = st.sidebar.selectbox("Subject", options=list(range(1, 10)),
                                         index=(st.session_state.current_subject or 3) - 1)
speed = st.sidebar.number_input("Playback speed", min_value=0.1, value=1.0, step=0.5)
st.sidebar.caption(f"Prediction smoothing: last {SMOOTHING_WINDOW} cycles (EMA)")  # NEW

connect_clicked = st.sidebar.button("🔌 Connect / Switch Subject", use_container_width=True)
disconnect_clicked = st.sidebar.button("⏹ Disconnect", use_container_width=True)

if connect_clicked:
    if st.session_state.pipeline is not None:
        st.session_state.pipeline["stop_event"].set()
    stop_outlet_process(st.session_state.outlet_process)

    with st.spinner(f"Starting stream for subject {selected_subject}..."):
        proc = start_outlet_process(selected_subject, speed)
        pipeline, error = initialize_pipeline(selected_subject, speed)

    if pipeline is None:
        st.sidebar.error(error)
        stop_outlet_process(proc)
        st.session_state.pipeline = None
        st.session_state.outlet_process = None
        st.session_state.current_subject = None
    else:
        st.session_state.pipeline = pipeline
        st.session_state.outlet_process = proc
        st.session_state.current_subject = selected_subject
        st.session_state.accuracy = {"n_full": 0, "n_full_correct": 0, "n_near_end": 0, "n_near_end_correct": 0}
        st.session_state.latency_history = deque(maxlen=LATENCY_HISTORY_LEN)
        st.session_state.prob_history = deque(maxlen=SMOOTHING_WINDOW)  # NEW: reset on subject switch

if disconnect_clicked:
    if st.session_state.pipeline is not None:
        st.session_state.pipeline["stop_event"].set()
    stop_outlet_process(st.session_state.outlet_process)
    st.session_state.pipeline = None
    st.session_state.outlet_process = None
    st.session_state.current_subject = None

st.sidebar.divider()
proc = st.session_state.outlet_process
proc_status = "running" if (proc is not None and proc.poll() is None) else "stopped"
st.sidebar.caption(f"Outlet subprocess: **{proc_status}**" + (f" (pid {proc.pid})" if proc and proc_status == "running" else ""))

pipeline = st.session_state.pipeline
acc = st.session_state.accuracy
latency_history = st.session_state.latency_history
prob_history = st.session_state.prob_history  # NEW

if pipeline is None:
    st.markdown(
        '<div style="margin-bottom:18px;"><span class="status-dot-off"></span>'
        '<span style="color:#8B92A3; font-size:13px; font-weight:500;">NOT CONNECTED</span></div>',
        unsafe_allow_html=True
    )
    st.info("Select a subject in the sidebar and click **Connect / Switch Subject** to begin streaming.")
    st.stop()

st.markdown(
    f'<div style="margin-bottom:18px;"><span class="status-dot"></span>'
    f'<span style="color:#8B92A3; font-size:13px; font-weight:500;">LIVE — Subject {pipeline["subject_id"]}</span></div>',
    unsafe_allow_html=True
)

hero_col, signal_col = st.columns([1, 1.4])
with hero_col:
    hero_slot = st.empty()
    conf_slot = st.empty()
    stats_slot = st.empty()
with signal_col:
    waveform_slot = st.empty()
    latency_slot = st.empty()

buffer = pipeline["buffer"]
latest_marker = pipeline["latest_marker"]
b, a = pipeline["b"], pipeline["a"]
sfreq = pipeline["sfreq"]
session = pipeline["session"]
input_name = pipeline["input_name"]
lead_samples = int(BUFFER_LEAD_MARGIN_SEC * sfreq)

effective_interval_start = INTERVAL_START_SEC / speed
effective_interval_end = INTERVAL_END_SEC / speed
effective_near_end_start = effective_interval_end - (NEAR_END_WINDOW_SEC / speed)

for _ in range(200):
    if st.session_state.pipeline is not pipeline:
        break

    window = buffer.get_window()
    if window is None:
        hero_slot.markdown(
            f'<div class="panel"><div class="panel-label">Status</div>'
            f'<div class="hero-sub">Buffering signal — {buffer.n_received} samples received…</div></div>',
            unsafe_allow_html=True
        )
        time.sleep(0.25)
        continue

    t_start = time.perf_counter()
    filtered_full = filtfilt(b, a, window, axis=1)
    filtered_window = filtered_full[:, lead_samples:]
    model_input = filtered_window[np.newaxis, np.newaxis, :, :].astype(np.float32)
    if model_input.shape[-1] != 1001:
        pad_width = 1001 - model_input.shape[-1]
        model_input = np.pad(model_input, ((0, 0), (0, 0), (0, 0), (0, pad_width)), mode="edge")

    logits = session.run(None, {input_name: model_input})[0][0]
    raw_probs = softmax(logits)
    raw_pred_idx = int(np.argmax(raw_probs))
    raw_pred_label = IDX_TO_LABEL[raw_pred_idx]
    raw_confidence = raw_probs[raw_pred_idx]
    t_end = time.perf_counter()
    latency_history.append((t_end - t_start) * 1000)

    # --- NEW: prediction smoothing (average probability vectors over recent cycles) ---
    prob_history.append(raw_probs)
    smoothed_probs = np.mean(np.array(prob_history), axis=0)
    pred_idx = int(np.argmax(smoothed_probs))
    pred_label = IDX_TO_LABEL[pred_idx]
    confidence = smoothed_probs[pred_idx]

    gt_label = latest_marker["label"]
    onset_time = latest_marker["onset_time"]
    now = time.time()
    zone_text = "Not yet scored"
    is_match = None

    if gt_label is not None and onset_time is not None:
        elapsed_since_onset = now - onset_time
        in_full = effective_interval_start <= elapsed_since_onset <= effective_interval_end
        in_near_end = effective_near_end_start <= elapsed_since_onset <= effective_interval_end
        if in_full:
            is_match = (gt_label == pred_label)  # scored against the SMOOTHED (displayed) decision
            acc["n_full"] += 1
            if is_match:
                acc["n_full_correct"] += 1
            zone_text = "Precision window" if in_near_end else "Scoring window"
            if in_near_end:
                acc["n_near_end"] += 1
                if is_match:
                    acc["n_near_end_correct"] += 1
        else:
            zone_text = "Rest period"

    pred_style = INTENT_STYLE.get(pred_label, INTENT_STYLE[None])
    gt_style = INTENT_STYLE.get(gt_label, INTENT_STYLE[None])
    match_note = ""
    if is_match is True:
        match_note = '<span class="badge" style="background:#37D39922; color:#37D399;">MATCH</span>'
    elif is_match is False:
        match_note = '<span class="badge" style="background:#FF4F4F22; color:#FF6B6B;">MISMATCH</span>'

    raw_note = ""
    if raw_pred_label != pred_label:
        raw_style = INTENT_STYLE.get(raw_pred_label, INTENT_STYLE[None])
        raw_note = (f'<div class="raw-note">Raw single-cycle read: '
                    f'<span style="color:{raw_style["color"]};">{raw_style["label"]}</span> '
                    f'({raw_confidence:.0%}) — smoothed over by recent history</div>')

    hero_slot.markdown(f"""
        <div class="panel">
            <div class="panel-label">Detected Intent — Subject {pipeline['subject_id']} (smoothed, n={len(prob_history)})</div>
            <div class="hero-value" style="color:{pred_style['color']};">{pred_style['label']}</div>
            <div class="hero-sub">
                Reference cue: <b style="color:{gt_style['color']};">{gt_style['label']}</b>
                &nbsp;·&nbsp; {zone_text} &nbsp; {match_note}
            </div>
            {raw_note}
        </div>
    """, unsafe_allow_html=True)

    conf_pct = confidence * 100
    conf_slot.markdown(f"""
        <div class="panel">
            <div class="panel-label">Smoothed Confidence</div>
            <div class="stat-value">{conf_pct:.1f}%</div>
            <div class="bar-track"><div class="bar-fill" style="width:{conf_pct}%; background:{pred_style['color']};"></div></div>
        </div>
    """, unsafe_allow_html=True)

    full_acc = acc["n_full_correct"] / acc["n_full"] * 100 if acc["n_full"] > 0 else 0.0
    near_acc = acc["n_near_end_correct"] / acc["n_near_end"] * 100 if acc["n_near_end"] > 0 else 0.0
    avg_latency = np.mean(latency_history) if latency_history else 0.0

    stats_slot.markdown(f"""
        <div class="panel">
            <div class="panel-label">Session Statistics</div>
            <div style="display:flex; justify-content:space-between; margin-top:6px;">
                <div><div class="stat-value">{avg_latency:.2f} ms</div><div class="stat-label">Avg. processing latency</div></div>
                <div><div class="stat-value">{full_acc:.1f}%</div><div class="stat-label">Full-window accuracy</div></div>
                <div><div class="stat-value">{near_acc:.1f}%</div><div class="stat-label">Precision-window accuracy</div></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    fig = plot_signal_pipeline(window, filtered_window, sfreq)
    waveform_slot.pyplot(fig)
    plt.close(fig)

    if len(latency_history) > 1:
        latency_slot.markdown('<div class="panel-label" style="margin-top:4px;">Latency Trend (last 50 cycles)</div>', unsafe_allow_html=True)
        latency_slot.line_chart(list(latency_history), height=140, use_container_width=True)

    time.sleep(0.25)

st.rerun()
