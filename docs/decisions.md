# Engineering Decision Log

Each entry records a meaningful technical decision made during this project, why it was made, what alternatives existed, and what its known consequences are. This document answers *why*, not *what changed* — see git history and `docs/PROJECT_NARRATIVE.md` for the latter. Entries are numbered chronologically by when the decision was made, not by importance.

---

## D-001: Session-based train/test split over random split

**Phase:** 2 (Offline Training)
**Decision:** Split each subject's trials by the dataset's native session labels (`0train` / `1test`) rather than a random shuffle-split.
**Reasoning:** The BCI Competition IV-2a protocol was designed to be evaluated this way — `0train` and `1test` are genuinely separate recording sessions (different day/setup), so this split tests generalization across sessions, not just across trials within one sitting. A random split would leak session-specific noise characteristics between train and test.
**Consequences:** More honest evaluation than a random split would have given, at the cost of a smaller, fixed test set (288 trials) rather than a tunable split ratio.

## D-002: EEGNet over a larger/custom CNN architecture

**Phase:** 2
**Decision:** Implement EEGNet (Lawhern et al., 2018) rather than a larger custom convolutional network.
**Reasoning:** EEGNet's inductive bias — temporal convolution, then depthwise spatial convolution per temporal filter, then separable convolution — mirrors the structure of classical EEG analysis pipelines (bandpass filter, then spatial filter). At 3,444 parameters, it is small enough to train reliably on ~230–288 trials per subject without severe overfitting, which a larger architecture would risk given the limited per-subject data available.
**Consequences:** Good offline accuracy (see Results) without needing data augmentation or transfer learning to make training tractable. Ceiling on accuracy versus larger architectures was not explored.

## D-003: ONNX Runtime (CPU) over PyTorch as the real-time inference backend

**Phase:** 3 (Export & Benchmarking)
**Decision:** Deploy inference via ONNX Runtime CPU, not PyTorch (CPU or MPS).
**Reasoning:** Direct benchmark (n=200, single-trial): PyTorch CPU 1.62ms mean, PyTorch MPS 0.93ms mean, ONNX Runtime CPU 0.29ms mean. For a model this small, MPS dispatch overhead outweighs its compute advantage; ONNX Runtime's static-graph optimizations minimize overhead more effectively than either PyTorch backend. This was measured, not assumed.
**Consequences:** Sub-millisecond inference, leaving effectively the entire real-time latency budget to signal acquisition and windowing rather than model compute. Requires maintaining an ONNX export step and numerical-equivalence verification alongside the PyTorch training code.

## D-004: Continuous streaming simulation over pre-cut trial replay

**Phase:** 4 (Streaming Simulation)
**Decision:** The LSL outlet streams continuous, unsegmented EEG (with realistic inter-trial gaps) rather than replaying pre-cut 4-second trial epochs.
**Reasoning:** A real headset never hands a system pre-segmented trials — the system has to detect and window relevant signal itself. Streaming continuous data forces the downstream buffering/scoring logic to solve the same problem a real deployment would face, rather than testing against an artificially easier setup.
**Consequences:** Directly caused the need for, and surfaced the value of, the trial-alignment debugging work in Phase 5 (D-006). A pre-cut-trial approach would have hidden the scoring-window bug entirely.

## D-005: Zero-phase (`filtfilt`) bandpass filtering, accepted as a real-time limitation

**Phase:** 5 (Real-Time Debugging)
**Decision:** Use `scipy.signal.filtfilt` for the 4–38Hz bandpass filter in the real-time pipeline, matching the zero-phase filtering MOABB applies offline during training-epoch extraction.
**Reasoning:** Matching the training-time filter characteristic as closely as possible was prioritized, since a preprocessing mismatch between training and inference was already identified as a major accuracy driver (see D-006, D-007). `filtfilt` requires access to data on both sides of the region being filtered, which a real-time system does not have for the newest edge of an incoming window.
**Consequences:** A documented, quantified real-time/offline accuracy gap (see README Results) attributable partly to this. The correct production fix — a causal, stateful filter (`sosfilt` with persistent `zi`) — is specified in the README's Engineering Roadmap but was not implemented, a deliberate scope decision (see D-014).

## D-006: Correcting the trial-scoring window using `dataset.interval`

**Phase:** 5
**Decision:** After discovering that real-time accuracy was implausibly low (34.5%, near chance) even after adding correct bandpass filtering (46.9%), the trial-scoring window was corrected from an assumed `[onset, onset+4s]` to the dataset's actual, code-confirmed `interval = (2, 6)` seconds.
**Reasoning:** MOABB's `MotorImagery` paradigm defines `tmin`/`tmax` relative to each dataset's own task interval, not the raw event marker timestamp. The BCI IV-2a protocol places a 2-second preparatory period before motor imagery actually begins. This was confirmed by directly querying `dataset.interval` in code, not inferred from documentation prose or memory.
**Consequences:** Raised real-time accuracy from ~47% to ~53–54%. This is the single highest-leverage bug fix in the project's real-time debugging history, and a direct lesson in verifying framework assumptions in code rather than assuming a plausible default.

## D-007: Leading-margin buffer extension to fix filter edge distortion

**Phase:** 5
**Decision:** Extend the rolling buffer by an extra ~1 second of real history beyond the 4-second target window, filter the full extended window, then crop off the distorted leading margin before inference.
**Reasoning:** Individual real-time trials showed a signature of systematic, not random, error — entire trials confidently misclassified into the same wrong class throughout. This pointed to `filtfilt`'s handling of window edges: offline, the entire continuous recording is filtered once, so every training epoch has genuine signal on both sides; the real-time pipeline was filtering a freshly-cut window each cycle, forcing `filtfilt` to pad the leading edge with reflected data rather than real history.
**Consequences:** Raised real-time accuracy from ~53% to 61.3% (full-window) / 65.1% (near-end) under the (later found to be leakage-affected) original model; comparable relative improvement confirmed against the corrected model (~64% full-window). Left the trailing edge of each window still non-causally filtered — the remaining, accepted structural limitation (see D-005, D-014).

## D-008: Trailing-margin filtering experiment — tested and rejected

**Phase:** 5
**Decision:** Tested whether adding a symmetric trailing margin (accepting ~0.5s additional latency so `filtfilt` would have genuine data on both edges) would further close the real-time/offline gap. Result: inconclusive to mildly negative on the tightest-alignment metric. Not adopted.
**Reasoning:** The added latency was not repaid by a reproducible accuracy gain, within the noise expected from the small number of scored predictions in a short test run.
**Consequences:** Kept the leading-margin-only configuration as final. Documented as a tested-and-rejected alternative rather than an unexplored option, consistent with reporting negative results rather than only positive ones.

## D-009: Live-demo subject selection — revised after the leakage fix

**Phase:** 6, revised in Phase 8
**Decision:** Subject 3 was originally selected for the live dashboard demo (highest leakage-affected offline accuracy, 81.9%). After the validation-methodology fix (D-013), Subject 9 was selected instead (77.3% mean, corrected methodology's best performer).
**Reasoning:** Once the original selection criterion's number was shown to be unreliable, the subject selection was redone against the corrected, trustworthy numbers rather than left pointing at a now-outdated justification.
**Consequences:** ONNX export and forensics-tool defaults were updated accordingly; Subject 1 was retained as the subject used for real-time-vs-offline comparison, since the entire Phase 5 debugging narrative and its specific numbers are documented against it.

## D-010: `st.fragment` over manual loop + `sleep` + `rerun` for live updates

**Phase:** 6
**Decision:** Migrated the dashboard's live-update mechanism from a manual `for` loop with `time.sleep()` and `st.rerun()` to Streamlit's `@st.fragment(run_every=...)` decorator.
**Reasoning:** The manual loop pattern caused a visible bug — toggling a sidebar control (accessibility mode) while the loop was mid-cycle produced duplicated, ghosted UI elements that did not clear on subsequent toggles. `st.fragment` is purpose-built for auto-refreshing part of a page independently of full-page reruns, and does not fight against widget-triggered reruns the way the manual pattern did.
**Consequences:** Eliminated the ghosting bug entirely and simplified the codebase (removed manual placeholder/loop bookkeeping). Required a full-file rewrite rather than a patch, since the update mechanism is structural, not incidental.

## D-011: Shared constants module (`src/common/bci_utils.py`)

**Phase:** 6 (codebase merge)
**Decision:** Extract every value shared across the live dashboard, forensics tool, and inference pipeline — channel lists, filter parameters, the validated `(2, 6)` scoring interval, file paths — into one module, imported everywhere rather than redefined per file.
**Reasoning:** The forensics tool, developed independently, was found to have re-introduced two bugs already fixed elsewhere in the codebase (the scoring-window bug from D-006, and a separate marker-timestamp clock mismatch) — a direct, observed cost of constants and logic drifting out of sync across files that should have shared one source of truth.
**Consequences:** A fix made once now applies everywhere it's needed. Required touching every file in the project during the merge, a one-time cost traded against ongoing duplication risk.

## D-012: Prediction smoothing (EMA) and confidence-gated rejection

**Phase:** 6 (feature additions)
**Decision:** Smooth per-cycle predictions using an exponential moving average over the last 5 cycles' softmax probability vectors (not hard-label majority voting), and withhold a decision entirely ("NO ACTION") when smoothed confidence falls below a tunable threshold.
**Reasoning:** Averaging probabilities rather than voting on hard labels lets a brief, low-confidence flicker into the wrong class be outvoted by several moderately-confident correct cycles nearby, rather than counting every cycle equally regardless of confidence. Confidence-gated rejection is a direct response to the project's assistive-device framing: a system that might drive a physical device should not act on an uncertain signal.
**Consequences:** Directly improved subjective and measured prediction stability. Introduces two new tunable parameters (smoothing window size, rejection threshold) with no principled method yet for selecting their values beyond manual tuning during development.

## D-013: Test-set leakage discovered and corrected (three-way split + full-data refit)

**Phase:** 8 (Documentation / External Review Response)
**Decision:** An independent external technical review identified that the original training loop selected its best checkpoint by repeatedly evaluating against the same session (`1test`) subsequently used for final reported accuracy — a form of test-set leakage through model selection. This was corrected with a strict three-way procedure: (1) an inner train/validation split, carved entirely from session `0train`, determines the training stopping point via honest validation; (2) a fresh model is trained from scratch on the *full* `0train` session, stopping at the epoch determined in step 1; (3) this final model is evaluated exactly once against the true held-out `1test` session, and that number is reported.
**Reasoning:** This is a genuine methodological bug, not an accepted structural tradeoff — unlike the filtering limitation (D-005), it was fixable without any conceptual compromise, and fixing it was treated as non-negotiable rather than something to document around. A proposal to selectively keep whichever of two independent training runs' numbers happened to be higher, per subject, was considered and explicitly rejected — that would reintroduce the identical leakage mechanism one level up (selecting outcomes based on test-set performance across runs, rather than across epochs).
**Consequences:** Overall mean offline accuracy dropped from 62.9% (leakage-affected) to 59.5% (corrected, mean of 2 independent runs) — not a uniform drop: strong-signal subjects moved modestly, already-near-chance subjects barely moved, and two previously "mid-tier" subjects (2, 7) diverged sharply, revealing that one (7) was genuinely data-starved (recovered with more training data) while the other (2) had no real signal to begin with (did not recover). This is treated as one of the project's more valuable findings, not merely a correction, and is discussed at length in `docs/PROJECT_NARRATIVE.md`. Every downstream artifact — ONNX exports, the live-demo subject choice (D-009), and all reported numbers — was regenerated and re-verified against the corrected models, not left partially updated.

## D-014: Causal filtering scoped but deliberately deferred

**Phase:** 8
**Decision:** The concrete fix for the real-time filtering limitation (D-005) — a causal, stateful filter using `scipy.signal.sosfilt` with persistent `zi` — was specified in detail (see README Engineering Roadmap) but not implemented before finalizing the project for portfolio submission.
**Reasoning:** This is a genuine design tradeoff, explained and quantified, not an invalid result — unlike D-013, deferring it does not make any reported number incorrect or misleading. Implementing and properly validating it (including checking whether the resulting phase delay creates a *new* training/inference mismatch, requiring its own empirical test) is a substantial, open-ended piece of work, weighed against a fixed timeline for internship application submission. A partial or rushed implementation, adopted without the same empirical rigor applied to every other change in this project, was judged worse than a clearly scoped, honestly deferred item.
**Consequences:** The real-time/offline accuracy gap remains at its currently measured, explained magnitude. This is the leading item in the Engineering Roadmap specifically because it is well-understood and shovel-ready, not because it is the easiest item to list.
