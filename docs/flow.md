# Execution Flow

Documents how execution actually moves between modules — what calls what,
in what order — for the system's non-obvious runtime paths (threading,
subprocess management, Streamlit's rerun model). Not a full API reference;
a map of the paths that were non-trivial to get right.

---

## Flow 1: Live prediction cycle (`src/app/Live_Monitor.py`)

Triggered every 250ms by `@st.fragment(run_every=0.25)` on `live_fragment()`,
independent of full-page Streamlit reruns (decisions.md D-010).

1. `buffer.get_window()` — `RollingBuffer.get_window()`. Returns `None`
   if the buffer hasn't filled yet; otherwise returns a `(22, 1250)`
   snapshot under `threading.Lock`, filled continuously by a background
   thread (`eeg_receiver_loop`) pulling from the LSL inlet.
2. `filtfilt(b, a, window, axis=1)` — coefficients from
   `src/common/bci_utils.py: design_bandpass_filter()`. Filters the full
   5s buffer (4s target + 1s leading margin), then crops the margin
   (decisions.md D-005, D-007).
3. `session.run(...)` — ONNX Runtime inference on the cropped window.
4. `softmax()` — `src/common/bci_utils.py`.
5. `prob_history.append(raw_probs)` → `np.mean(prob_history, axis=0)` —
   EMA smoothing over the last `SMOOTHING_WINDOW=5` cycles (decisions.md D-012).
6. Confidence check against `rejection_threshold` — below threshold,
   `display_label = None`, cycle excluded from scoring (decisions.md D-012).
7. Ground-truth scoring: compares `display_label` against
   `latest_marker["label"]`, gated by `[onset+2s, onset+6s]` window
   (decisions.md D-006), scaled by `speed` for playback-rate correctness.
8. Render: hero panel (intent + command schematic), confidence bar,
   session stats, waveform (raw + filtered), signal quality, confusion
   matrix + per-class table, session log append.

**Background threads** (started once, on "Connect / Switch Subject"):
- `eeg_receiver_loop` — pulls from LSL EEG inlet, appends to `RollingBuffer`.
- `marker_receiver_loop` — pulls from LSL marker inlet, updates
  `latest_marker` dict (label + onset timestamp).

**Subprocess managed by this page:**
- `start_outlet_process(subject_id, speed)` launches
  `lsl_outlet.py --subject N --speed S` via `subprocess.Popen`, killed
  and relaunched on every "Connect / Switch Subject" click.

---

## Flow 2: Subject switch

1. If a pipeline already exists: `pipeline["stop_event"].set()` +
   `stop_outlet_process(old_proc)`.
2. `start_outlet_process(new_subject, speed)` — launches new outlet subprocess.
3. `ensure_onnx_model(new_subject)` — checks `models/eegnet_subject{N}.onnx`;
   if missing, rebuilds the checkpoint into ONNX on the fly (shared logic,
   used identically by the standalone `export_onnx.py` script and the
   forensics tool — decisions.md D-011).
4. `initialize_pipeline()` — resolves `SimulatedEEG_S{N}` /
   `SimulatedMarkers_S{N}`, starts fresh receiver threads, resets
   `prob_history`, `accuracy`, `latency_history`, `session_log`.

---

## Flow 3: Forensics replay (`src/app/pages/`)

1. `glob(DATA_PROCESSED_DIR / "replay_*.npy")` — lists recorded sessions.
2. On select: `ReplayEngine(path)` —
   `parse_subject_from_filename()` regex-extracts the subject ID from
   `replay_S{N}_{timestamp}.npy`, so the correct trained model always
   loads, never assumed; `ensure_onnx_model(subject_id)` shares the same
   export logic as Flow 2.
3. On scrub/auto-play tick: `engine.step(index)` slices the recorded
   array, applies the same filter+crop as Flow 1 step 2, runs inference,
   returns `(pred_label, confidence, filtered_window, actual_t)`.
4. Ground truth: loads the companion `*_markers.npy`, applies the same
   `[onset+2s, onset+6s]` logic as Flow 1 step 7 via the shared constants
   — this consistency is exactly what decisions.md D-011 protects against
   silently drifting apart again.

**"Record new session":**
1. Reads `st.session_state["current_subject"]` (cross-page shared state)
   so a recording always matches whatever is live in Live Monitor.
2. Runs `record_replay_data_with_markers.py --subject N --duration 60`,
   which manages its own outlet subprocess, always at `--speed 1`
   (real-time) — recordings must be real-time so marker wall-clock
   timestamps and sample-index-derived time share one consistent scale.

---

## Flow 4: Offline training (`src/training/train_all_subjects_refit.py`)

Two-stage per subject (decisions.md D-013):

1. **Stage 1 (probe):** stratified 80/20 split of session `0train` →
   train probe model, track best epoch by inner-validation accuracy only
   (session `1test` never touched here).
2. **Stage 2 (refit):** fresh model trained on the *full* `0train`
   session, for exactly the epoch count found in Stage 1 → evaluated
   exactly once against `1test` → checkpoint saved with `final_test_acc`,
   appended to `logs/training_results.csv`.
3. Downstream: `export_onnx.py --subject N` loads the checkpoint, exports
   ONNX; `verify_onnx.py` confirms numerical equivalence against PyTorch
   before it's trusted anywhere else.
