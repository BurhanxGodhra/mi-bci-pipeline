"""
Professional, minimalist offline replay dashboard for BCI forensics.
Allows selection of recording sessions, auto-play, and live confusion matrix.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import welch
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import time
import sys
import os
import subprocess
import glob
import io
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.forensics.replay_engine import ReplayEngine
from src.common.bci_utils import (
    INTERVAL_START_SEC, INTERVAL_END_SEC, NEAR_END_WINDOW_SEC,
    DATA_PROCESSED_DIR, STREAMING_DIR,
)


# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
MODEL_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz"
]

INTENT_COLORS = {
    'feet': '#37D399',
    'left_hand': '#4F8DFF',
    'right_hand': '#B07CFF',
    'tongue': '#FFB454',
    None: '#5A6270'
}
LABEL_MAP = {'feet': 0, 'left_hand': 1, 'right_hand': 2, 'tongue': 3}

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(page_title="Neural Replay Forensics", layout="wide")

# Custom CSS
st.markdown("""
<style>
    body, .stApp {
        background-color: #0E1117;
        color: #E6E6E6;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .main-header {
        font-size: 28px;
        font-weight: 500;
        letter-spacing: -0.5px;
        color: #FFFFFF;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 14px;
        color: #888888;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1A1D24;
        border-radius: 8px;
        padding: 16px 20px;
        border: 1px solid #2A2D35;
    }
    .metric-label {
        font-size: 12px;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 500;
        color: #FFFFFF;
        line-height: 1.3;
    }
    .metric-delta {
        font-size: 14px;
        color: #37D399;
    }
    .match-true {
        color: #37D399;
    }
    .match-false {
        color: #FF6B6B;
    }
    .section-title {
        font-size: 16px;
        font-weight: 500;
        color: #CCCCCC;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid #2A2D35;
        padding-bottom: 0.5rem;
    }
    .stat-item {
        background-color: #1A1D24;
        border-radius: 6px;
        padding: 10px 16px;
        border: 1px solid #2A2D35;
        min-width: 120px;
    }
    .stat-label {
        font-size: 11px;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .stat-number {
        font-size: 18px;
        font-weight: 500;
        color: #FFFFFF;
    }
    .progress-bar {
        width: 100%;
        height: 4px;
        background-color: #2A2D35;
        border-radius: 2px;
        margin: 8px 0;
    }
    .progress-fill {
        height: 100%;
        background-color: #37D399;
        border-radius: 2px;
        transition: width 0.1s ease;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Neural Replay Forensics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Offline analysis of pre-recorded EEG sessions</div>', unsafe_allow_html=True)

# -------------------------------------------------------------------
# Session state initialization (MUST come before sidebar access)
# -------------------------------------------------------------------
if 'history' not in st.session_state:
    st.session_state.history = []
if 'all_preds' not in st.session_state:
    st.session_state.all_preds = []
if 'all_truths' not in st.session_state:
    st.session_state.all_truths = []
if 'current_time' not in st.session_state:
    st.session_state.current_time = 5.0
if 'last_file' not in st.session_state:
    st.session_state.last_file = None
if 'auto_play_active' not in st.session_state:
    st.session_state.auto_play_active = False

# -------------------------------------------------------------------
# Session management
# -------------------------------------------------------------------
data_dir = str(DATA_PROCESSED_DIR)
eeg_files = glob.glob(os.path.join(data_dir, "replay_*.npy"))
eeg_files = [f for f in eeg_files if "_markers" not in f]

if not eeg_files:
    st.warning("No recording sessions found. Please record a session first.")
    if st.button("Record new session (60s)"):
        with st.spinner("Recording..."):
            script_path = str(STREAMING_DIR / "record_replay_data_with_markers.py")
            active_subject = st.session_state.get("current_subject")
            if active_subject is None:
                st.error("No subject is currently connected in Live Monitor. Go there and click "
                         "'Connect / Switch Subject' first, then come back here to record.")
                st.stop()
            subprocess.run(
                ["python", script_path, "--subject", str(active_subject), "--duration", "60"],
                cwd=str(STREAMING_DIR), check=True,
            )
        st.rerun()
    st.stop()

file_names = [os.path.basename(f) for f in eeg_files]
selected_file = st.sidebar.selectbox("Session", file_names)
selected_path = os.path.join(data_dir, selected_file)

# Detect session change and reset state
if st.session_state.last_file != selected_file:
    st.session_state.last_file = selected_file
    st.session_state.all_preds = []
    st.session_state.all_truths = []
    st.session_state.history = []
    st.session_state.current_time = 5.0
    st.session_state.auto_play_active = False
    st.cache_resource.clear()

@st.cache_resource
def load_engine_for_file(path):
    return ReplayEngine(path)

engine = load_engine_for_file(selected_path)
total_sec = engine.total_seconds

base_name = selected_file.replace(".npy", "")
marker_path = os.path.join(data_dir, f"{base_name}_markers.npy")
if os.path.exists(marker_path):
    markers = np.load(marker_path, allow_pickle=True)
    st.sidebar.success(f"Markers: {len(markers)} events")
else:
    markers = None
    st.sidebar.warning("No markers file for this session")

# -------------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### Controls")

# Auto-play checkbox
auto_play = st.sidebar.checkbox("Auto-play", value=st.session_state.auto_play_active)
st.session_state.auto_play_active = auto_play

if auto_play:
    speed = st.sidebar.select_slider("Speed", options=[0.5, 1.0, 2.0, 5.0, 10.0], value=1.0)
    loop = st.sidebar.checkbox("Loop (repeat)", value=False)
    reset_on_play = st.sidebar.checkbox("Reset matrix on play", value=True)
else:
    speed = 1.0
    loop = False
    reset_on_play = False

# If auto-play was just turned on, reset if requested
if auto_play and reset_on_play and st.session_state.get('last_auto_play_state', False) == False:
    st.session_state.all_preds = []
    st.session_state.all_truths = []
    st.session_state.history = []
    st.session_state.current_time = 5.0
st.session_state.last_auto_play_state = auto_play

reset = st.sidebar.button("Reset confusion matrix")
if reset:
    st.session_state.all_preds = []
    st.session_state.all_truths = []
    st.session_state.history = []

if st.sidebar.button("Record new session"):
    with st.spinner("Recording 60s session..."):
        script_path = str(STREAMING_DIR / "record_replay_data_with_markers.py")
        active_subject = st.session_state.get("current_subject")
        if active_subject is None:
            st.error("No subject is currently connected in Live Monitor. Go there and click "
                     "'Connect / Switch Subject' first, then come back here to record.")
            st.stop()
        subprocess.run(
            ["python", script_path, "--subject", str(active_subject), "--duration", "60"],
            cwd=str(STREAMING_DIR), check=True,
        )
    st.rerun()

# Display session info
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Session duration:** {total_sec:.1f}s")
st.sidebar.markdown(f"**Predictions:** {len(st.session_state.all_preds)}")

# -------------------------------------------------------------------
# Main slider
# -------------------------------------------------------------------
if auto_play:
    t = st.session_state.current_time
    t += 0.05 * speed
    if t >= total_sec:
        if loop:
            t = 0.0
        else:
            # Stop auto-play at end
            t = total_sec
            st.session_state.auto_play_active = False
            st.info("Session ended. Auto-play stopped.")
            # We'll let the rerun handle the checkbox update
    st.session_state.current_time = t
    actual_t = t
    # Show slider disabled when auto-playing
    st.slider("Time (s)", 0.0, total_sec, t, step=0.05, disabled=True, key="time_slider")
else:
    t = st.slider("Time (s)", 0.0, total_sec, st.session_state.current_time, step=0.05)
    st.session_state.current_time = t
    actual_t = t

# Progress bar
progress = actual_t / total_sec if total_sec > 0 else 0
st.markdown(f"""
<div style="margin-top: -0.5rem; margin-bottom: 1rem;">
    <div class="progress-bar">
        <div class="progress-fill" style="width: {progress*100:.1f}%;"></div>
    </div>
    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #888888;">
        <span>0s</span>
        <span>{actual_t:.1f}s / {total_sec:.1f}s</span>
        <span>{total_sec:.1f}s</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Run inference
# -------------------------------------------------------------------
sample_rate = engine.sfreq
index = int(actual_t * sample_rate)
if index < int((4.0 + 1.0) * sample_rate):
    st.info("Buffering... (need at least 5s of data)")
    st.stop()

pred_label, conf, filt_win, actual_t = engine.step(index)
if pred_label is None:
    st.info("Not enough data.")
    st.stop()

# Ground truth
gt_label = None
if markers is not None and len(markers) > 0:
    preceding = markers[markers['timestamp'] <= actual_t]
    if len(preceding) > 0:
        onset_time = preceding['timestamp'][-1]
        candidate_label = preceding['label'][-1]
        elapsed = actual_t - onset_time
        if INTERVAL_START_SEC <= elapsed <= INTERVAL_END_SEC:
            gt_label = candidate_label

# Update history and confusion matrix (only if auto-play is not stopped)
st.session_state.history.append(pred_label)
if len(st.session_state.history) > 50:
    st.session_state.history.pop(0)

if gt_label is not None:
    st.session_state.all_preds.append(pred_label)
    st.session_state.all_truths.append(gt_label)

# -------------------------------------------------------------------
# Metric cards
# -------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Predicted</div>
        <div class="metric-value">{pred_label.upper()}</div>
        <div class="metric-delta">{conf:.1%} confidence</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    gt_display = gt_label.upper() if gt_label else "—"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Ground Truth</div>
        <div class="metric-value">{gt_display}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    if gt_label:
        match = (pred_label == gt_label)
        match_text = "MATCH" if match else "MISMATCH"
        match_class = "match-true" if match else "match-false"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Status</div>
            <div class="metric-value {match_class}">{match_text}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Status</div>
            <div class="metric-value" style="color:#888;">—</div>
        </div>
        """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Time</div>
        <div class="metric-value">{actual_t:.1f}s</div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------
# Waveform plot
# -------------------------------------------------------------------
st.markdown('<div class="section-title">Signal Waveforms</div>', unsafe_allow_html=True)
fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
raw_win = engine.get_window_at_index(index)
if raw_win is not None:
    raw_segment = raw_win[:, -filt_win.shape[1]:]
    time_axis = np.linspace(0, filt_win.shape[1] / sample_rate, filt_win.shape[1])
    chs = ['C3', 'Cz', 'C4']
    ch_idx = [MODEL_CHANNELS.index(c) for c in chs]
    offsets = [0, 50, 100]
    for i, ci in enumerate(ch_idx):
        axes[0].plot(time_axis, raw_segment[ci] + offsets[i], label=f'Raw {chs[i]}', linewidth=0.8)
        axes[1].plot(time_axis, filt_win[ci] + offsets[i], label=f'Filtered {chs[i]}', linewidth=0.8)
    axes[0].set_title("Raw EEG", fontsize=12, color='#AAAAAA')
    axes[1].set_title("Filtered (4-38 Hz)", fontsize=12, color='#AAAAAA')
    axes[1].set_xlabel("Time (s)")
    for ax in axes:
        ax.set_facecolor('#0E1117')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#2A2D35')
        ax.spines['left'].set_color('#2A2D35')
        ax.tick_params(colors='#888888')
        ax.legend(loc='upper right', frameon=False, labelcolor='#AAAAAA')
        ax.grid(True, alpha=0.15)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# -------------------------------------------------------------------
# Topographic map and Confusion matrix side by side
# -------------------------------------------------------------------
st.markdown('<div class="section-title">Spatial & Performance Metrics</div>', unsafe_allow_html=True)

col_top, col_cm = st.columns(2)

# --- Topographic map ---
with col_top:
    freqs, psd = welch(filt_win, fs=sample_rate, nperseg=min(256, filt_win.shape[1]))
    alpha_mask = (freqs >= 8) & (freqs <= 12)
    alpha_power = np.mean(psd[:, alpha_mask], axis=1)

    info = mne.create_info(ch_names=MODEL_CHANNELS, sfreq=sample_rate, ch_types='eeg')
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        info.set_montage(montage)
    except:
        montage = None

    fig2, ax2 = plt.subplots(figsize=(3.5, 3.5))
    if montage is not None:
        try:
            mne.viz.plot_topomap(alpha_power, info, axes=ax2, show=False, cmap='RdBu_r')
        except:
            ax2.text(0.5, 0.5, "Unavailable", ha='center', va='center', color='#888888')
    else:
        ax2.text(0.5, 0.5, "No montage", ha='center', va='center', color='#888888')
    ax2.set_facecolor('#0E1117')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# --- Confusion matrix ---
with col_cm:
    if len(st.session_state.all_truths) > 0:
        labels = list(LABEL_MAP.keys())
        cm = confusion_matrix(st.session_state.all_truths, st.session_state.all_preds, labels=labels)
        fig_cm, ax_cm = plt.subplots(figsize=(3.5, 3.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax_cm,
                    cbar=False, square=True)
        ax_cm.set_xlabel('Predicted', color='#AAAAAA')
        ax_cm.set_ylabel('True', color='#AAAAAA')
        ax_cm.set_title('')
        ax_cm.tick_params(colors='#888888')
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)
    else:
        st.info("No ground truth data yet.")

# -------------------------------------------------------------------
# Statistics & Report
# -------------------------------------------------------------------
st.markdown('<div class="section-title">Session Statistics</div>', unsafe_allow_html=True)

if len(st.session_state.all_truths) > 0:
    labels = list(LABEL_MAP.keys())
    total = len(st.session_state.all_truths)
    correct = sum(1 for p, t in zip(st.session_state.all_preds, st.session_state.all_truths) if p == t)
    acc = correct / total if total > 0 else 0

    report = classification_report(st.session_state.all_truths, st.session_state.all_preds, labels=labels, output_dict=True, zero_division=0)

    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    col_stat1.metric("Total Predictions", total)
    col_stat2.metric("Overall Accuracy", f"{acc:.1%}")
    col_stat3.metric("Correct", correct)
    col_stat4.metric("Incorrect", total - correct)

    st.markdown("**Per-class Performance**")
    class_stats = []
    for label in labels:
        if label in report:
            stats = report[label]
            class_stats.append({
                "Class": label,
                "Precision": stats.get('precision', 0),
                "Recall": stats.get('recall', 0),
                "F1-score": stats.get('f1-score', 0),
                "Support": stats.get('support', 0)
            })
    df = pd.DataFrame(class_stats).round(3)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Download Report
    def generate_report():
        buf = io.StringIO()
        buf.write("NEURAL REPLAY FORENSICS – SESSION REPORT\n")
        buf.write("="*50 + "\n")
        buf.write(f"Session: {selected_file}\n")
        buf.write(f"Total duration: {total_sec:.1f}s\n")
        buf.write(f"Predictions made: {total}\n")
        buf.write(f"Overall accuracy: {acc:.1%}\n\n")
        buf.write("Per-class performance:\n")
        for label in labels:
            if label in report:
                s = report[label]
                buf.write(f"  {label:12} | Precision: {s.get('precision',0):.3f} | Recall: {s.get('recall',0):.3f} | F1: {s.get('f1-score',0):.3f} | Support: {s.get('support',0)}\n")
        buf.write("\nConfusion Matrix:\n")
        cm = confusion_matrix(st.session_state.all_truths, st.session_state.all_preds, labels=labels)
        buf.write(" " + " ".join(f"{l:>8}" for l in labels) + "\n")
        for i, row in enumerate(cm):
            buf.write(f"{labels[i]:8} " + " ".join(f"{v:>8}" for v in row) + "\n")
        buf.write("\nGenerated: " + time.strftime("%Y-%m-%d %H:%M:%S"))
        return buf.getvalue()

    if st.button("📥 Download Report"):
        report_text = generate_report()
        st.download_button(
            label="Download as .txt",
            data=report_text,
            file_name=f"forensics_report_{base_name}.txt",
            mime="text/plain"
        )
else:
    st.info("Not enough ground truth data to generate statistics. Keep scrubbing or use a session with markers.")

# -------------------------------------------------------------------
# Prediction history
# -------------------------------------------------------------------
st.markdown('<div class="section-title">Prediction History (last 50)</div>', unsafe_allow_html=True)
history_str = " ".join([
    f"<span style='background:{INTENT_COLORS.get(l, '#5A6270')}; padding:2px 6px; border-radius:4px; font-size:12px;'>{l}</span>"
    for l in st.session_state.history
])
st.markdown(f'<div style="padding:8px 0;">{history_str}</div>', unsafe_allow_html=True)

# -------------------------------------------------------------------
# Auto-play loop (re-run)
# -------------------------------------------------------------------
if auto_play and st.session_state.auto_play_active:
    time.sleep(0.05)
    st.rerun()