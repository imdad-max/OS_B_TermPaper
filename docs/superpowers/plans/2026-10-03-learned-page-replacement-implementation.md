# Learned Page Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a small terminal-based Python experiment comparing FIFO, LRU, Optimal, and a leakage-free decision-tree page-eviction policy across an explicit midpoint workload shift.

**Architecture:** Four focused modules separate classical simulation, workload generation, learned-policy training/prediction, and experiment reporting. `main.py` supplies validated interactive or command-line configuration; every policy consumes the same held-out trace and emits aligned hit/fault events that are aggregated around one shift index.

**Tech Stack:** Python 3.10+, NumPy, pandas, matplotlib, scikit-learn, and Python's built-in `unittest`.

**Spec:** `docs/superpowers/specs/2026-10-03-learned-page-replacement-design.md`

## Global Constraints

- Run on Windows/VS Code with `python main.py`; do not require Linux, Docker, a VM, GPU, database, API, GUI, web app, or deep learning.
- Standard defaults are 10 pages, 4 frames, seed 42, and 1,000 references per phase.
- Valid interactive configurations require `pages >= 3` and `2 <= frames < pages`.
- Phase 1 uses an 85% sequential working-set choice and 15% whole-universe noise.
- Phase 2 uses 70% whole-universe random choice and 30% temporary-hot-set choice; the hot set changes every 25 references.
- The workload shifts exactly halfway; policy memory is not reset at the shift.
- Training uses three traces with seeds `seed + 1`, `seed + 2`, and `seed + 3`; evaluation uses only `seed`.
- Learned features are exactly recency, total frequency, previous-20-reference frequency, and residency time; page identity and future-use information are excluded.
- The decision tree uses `max_depth=5`, `min_samples_leaf=5`, `class_weight="balanced"`, and `random_state=seed`.
- Belady future knowledge is permitted only in Optimal simulation and offline training-label generation.
- Do not embed result values or result-dependent conclusions in source code or documentation.
- Use built-in `unittest`; tests must be observed failing before their production behavior is added.

## Review Focus

- Minimum meaningful configuration (`pages=3`, `frames=2`) must complete training and evaluation instead of producing a one-class model.
- `frames >= pages`, `frames < 2`, and `pages < 3` must be rejected before experiment execution through both programmatic and terminal entry points.
- Blank terminal input must select defaults, while non-integer input must print a clear message and retry without terminating.
- Learned eviction must locate the probability column by `model.classes_` rather than assuming label `1` is always column index 1.
- An output directory whose path contains spaces must work, and a second run must replace the current CSV/figures cleanly without depending on stale files.

---

## File Map

- `src/page_replacement.py`: shared simulation result type, trace validation, Optimal victim selection, FIFO/LRU/Optimal implementations.
- `src/workload.py`: deterministic two-phase workload type and generator.
- `src/ml_model.py`: past-only features, Belady-labelled training data, decision-tree training, learned-policy simulation.
- `src/experiment.py`: common-trace policy evaluation, phase metrics, CSV generation, and plotting.
- `main.py`: argument parsing, terminal prompts, validation, execution, and concise console summary.
- `tests/test_page_replacement.py`: hand-derived classical algorithm tests.
- `tests/test_workload.py`: reproducibility, range, validation, and shift-characteristic tests.
- `tests/test_ml_model.py`: leakage boundary, labels, training, scoring, tie-breaking, and learned simulation tests.
- `tests/test_experiment.py`: metric aggregation, fair common-trace evaluation, CSV, figures, and rerun tests.
- `tests/test_main.py`: terminal defaults, retry behavior, CLI configuration, and end-to-end entry-point tests.
- `README.md`: assignment explanation, manual examples, setup, reproduction, output, limitations, disclosure.
- `docs/demo_and_viva.md`: 3–5 minute demo outline and 15 concise viva answers.
- `requirements.txt`: only NumPy, pandas, matplotlib, and scikit-learn.
- `.gitignore`: virtual environments, bytecode, caches, and editor-local files; generated assignment results remain tracked.

### Task 1: Classical Page-Replacement Simulators

**Files:**
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `src/page_replacement.py`
- Create: `tests/test_page_replacement.py`
- Create: `.gitignore`

**Interfaces:**
- Consumes: no project interfaces.
- Produces: `SimulationResult(algorithm: str, hits: int, faults: int, events: tuple[bool, ...], evictions: tuple[int, ...])`; `choose_optimal_victim(resident_pages: Sequence[int], future_trace: Sequence[int]) -> int`; `simulate_fifo(trace: Sequence[int], frame_count: int) -> SimulationResult`; `simulate_lru(...) -> SimulationResult`; `simulate_optimal(...) -> SimulationResult`.

- [ ] **Step 1: Write failing validation and hand-calculated algorithm tests**

Add tests asserting:

```python
trace = [8, 1, 2, 3, 1, 4, 1, 5, 3, 4, 1, 4, 3]
self.assertEqual((simulate_fifo(trace, 3).hits, simulate_fifo(trace, 3).faults), (3, 10))
self.assertEqual((simulate_lru(trace, 3).hits, simulate_lru(trace, 3).faults), (4, 9))
self.assertEqual((simulate_optimal(trace, 3).hits, simulate_optimal(trace, 3).faults), (6, 7))
```

Also assert `len(events) == len(trace)`, `hits + faults == len(trace)`, exact victim choices on a short trace, lowest-page tie-breaking in `choose_optimal_victim`, and `ValueError` for an empty trace, non-positive page identifiers, or `frame_count < 1`.

- [ ] **Step 2: Run the classical test module and observe the expected import failure**

Run: `python -m unittest tests.test_page_replacement -v`

Expected: FAIL because `src.page_replacement` does not exist.

- [ ] **Step 3: Implement shared validation, result recording, and FIFO**

Create the exact interfaces above. FIFO keeps a `deque` of insertion order; hits never reorder it. Record `True` for a hit and `False` for a fault.

- [ ] **Step 4: Run the FIFO/validation tests and verify they pass while LRU/Optimal remain failing**

Run: `python -m unittest tests.test_page_replacement -v`

Expected: FIFO and validation assertions PASS; unimplemented LRU/Optimal assertions FAIL.

- [ ] **Step 5: Implement LRU and Optimal**

LRU evicts the resident page with the smallest last-access index. Optimal calls `choose_optimal_victim` with the suffix strictly after the current reference; a page never used again has infinite next-use distance, with lowest page identifier breaking equal distances.

- [ ] **Step 6: Run the module and full suite**

Run: `python -m unittest tests.test_page_replacement -v`

Expected: all classical tests PASS.

Run: `python -m unittest discover -s tests -v`

Expected: all currently present tests PASS.

- [ ] **Step 7: Commit Task 1**

```text
git add .gitignore src tests/test_page_replacement.py
git commit -m "feat: implement classical page replacement policies"
```

### Task 2: Reproducible Two-Phase Workload

**Files:**
- Create: `src/workload.py`
- Create: `tests/test_workload.py`

**Interfaces:**
- Consumes: no earlier project interfaces.
- Produces: `Workload(trace: tuple[int, ...], shift_index: int)` and `generate_workload(page_count: int, frame_count: int, references_per_phase: int = 1000, seed: int = 42) -> Workload`.

- [ ] **Step 1: Write failing workload tests**

Test exact contracts:

```python
workload = generate_workload(10, 4, references_per_phase=1000, seed=42)
self.assertEqual(len(workload.trace), 2000)
self.assertEqual(workload.shift_index, 1000)
self.assertTrue(all(1 <= page <= 10 for page in workload.trace))
self.assertEqual(workload, generate_workload(10, 4, 1000, 42))
self.assertNotEqual(workload.trace, generate_workload(10, 4, 1000, 43).trace)
```

For seed 42, compute the share of transitions that follow the configured Phase-1 cycle and assert it is at least 0.20 greater before the shift than after it. Add the Review Focus minimum-configuration test and `ValueError` tests for `pages < 3`, `frames < 2`, `frames >= pages`, and `references_per_phase < 25`.

- [ ] **Step 2: Run workload tests and observe the expected import failure**

Run: `python -m unittest tests.test_workload -v`

Expected: FAIL because `src.workload` does not exist.

- [ ] **Step 3: Implement the workload generator**

Use `numpy.random.default_rng(seed)`. Phase 1 uses working-set size `min(page_count, max(2, min(frame_count, 4)))`, one sequential cursor, and the specified 85/15 choice. Phase 2 resamples `min(3, page_count)` unique hot pages every 25 positions and uses the specified 70/30 choice. Return immutable integer tuples and an exact midpoint.

- [ ] **Step 4: Run workload tests and the full suite**

Run: `python -m unittest tests.test_workload -v`

Expected: all workload tests PASS.

Run: `python -m unittest discover -s tests -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit Task 2**

```text
git add src/workload.py tests/test_workload.py
git commit -m "feat: generate reproducible workload shifts"
```

### Task 3: Leakage-Free Learned Eviction Policy

**Files:**
- Create: `src/ml_model.py`
- Create: `tests/test_ml_model.py`
- Create: `requirements.txt`

**Interfaces:**
- Consumes: `SimulationResult` and `choose_optimal_victim` from Task 1; trace tuples from Task 2.
- Produces: `FEATURE_NAMES`; `TrainingData(features: np.ndarray, labels: np.ndarray, decision_ids: np.ndarray)`; `build_candidate_features(page: int, time_index: int, last_access: Mapping[int, int], access_counts: Mapping[int, int], recent_window: Sequence[int], loaded_at: Mapping[int, int]) -> tuple[int, int, int, int]`; `generate_training_data(traces: Sequence[Sequence[int]], frame_count: int) -> TrainingData`; `train_eviction_model(training_data: TrainingData, random_state: int) -> DecisionTreeClassifier`; `choose_learned_victim(candidates: Sequence[int], feature_rows: Sequence[Sequence[float]], eviction_probabilities: Sequence[float]) -> int`; `simulate_learned(trace: Sequence[int], frame_count: int, model: DecisionTreeClassifier) -> SimulationResult`.

- [ ] **Step 1: Write failing feature and label-generation tests**

Use a hand-built history to assert the exact feature tuple `(recency, total_frequency, recent_frequency, residency_time)` and assert `FEATURE_NAMES` contains no page identity or future-use field. For a small trace, group rows by `decision_ids` and assert every decision has exactly one positive label and `frame_count - 1` negative labels.

- [ ] **Step 2: Run focused tests and observe the expected import failure**

Run: `python -m unittest tests.test_ml_model.LearnedFeatureTests tests.test_ml_model.TrainingLabelTests -v`

Expected: FAIL because `src.ml_model` does not exist.

- [ ] **Step 3: Implement past-only features and Belady label generation**

Update history only after features and labels for the current fault have been captured. Use a `deque(maxlen=20)` for recent history. Use `choose_optimal_victim` only inside offline label generation, never inside learned simulation.

- [ ] **Step 4: Verify feature and label tests pass**

Run: `python -m unittest tests.test_ml_model.LearnedFeatureTests tests.test_ml_model.TrainingLabelTests -v`

Expected: all selected tests PASS.

- [ ] **Step 5: Write failing classifier, probability-column, tie-break, and learned-simulation tests**

Assert tree parameters and both learned classes. Train a real tree on literal samples whose `classes_` ordering is inspected, then assert the label-1 column is used. Test `choose_learned_victim` with equal probabilities: greater recency wins, followed by lower page identifier. Simulate a short trace and assert aligned events plus `hits + faults == len(trace)`.

- [ ] **Step 6: Run the new tests and observe expected missing-behavior failures**

Run: `python -m unittest tests.test_ml_model -v`

Expected: feature/label tests PASS; classifier and simulation tests FAIL because behavior is not implemented.

- [ ] **Step 7: Implement model training and learned simulation**

Create the exact configured `DecisionTreeClassifier`. Reject empty data or data without labels 0 and 1. In `simulate_learned`, obtain `positive_column = list(model.classes_).index(1)` and never assume a fixed probability-column index. Use the pure victim chooser for deterministic ties.

- [ ] **Step 8: Run ML tests and the full suite**

Run: `python -m unittest tests.test_ml_model -v`

Expected: all ML tests PASS.

Run: `python -m unittest discover -s tests -v`

Expected: all tests PASS.

- [ ] **Step 9: Add the minimal dependency manifest**

Write `requirements.txt` with only these compatible ranges:
`numpy>=1.24,<3`, `pandas>=2.0,<3`, `matplotlib>=3.7,<4`, and
`scikit-learn>=1.3,<2`.

- [ ] **Step 10: Commit Task 3**

```text
git add requirements.txt src/ml_model.py tests/test_ml_model.py
git commit -m "feat: add learned eviction classifier"
```

### Task 4: Experiment Metrics, CSV, and Figures

**Files:**
- Create: `src/experiment.py`
- Create: `tests/test_experiment.py`

**Interfaces:**
- Consumes: all simulators, `generate_workload`, `generate_training_data`, and `train_eviction_model`.
- Produces: `ExperimentConfig(page_count: int = 10, frame_count: int = 4, seed: int = 42, references_per_phase: int = 1000, output_dir: Path = Path("results"))`; `ExperimentReport(results: pd.DataFrame, training_accuracy: float, evaluation_seed: int, training_seeds: tuple[int, int, int], shift_index: int, trace_length: int, csv_path: Path, figure_paths: tuple[Path, ...])`; `aggregate_metrics(result: SimulationResult, shift_index: int, config: ExperimentConfig) -> list[dict[str, object]]`; `evaluate_policies(trace: Sequence[int], frame_count: int, learned_model: DecisionTreeClassifier) -> dict[str, SimulationResult]`; `run_experiment(config: ExperimentConfig) -> ExperimentReport`.

- [ ] **Step 1: Write failing metric and common-trace evaluation tests**

Create a literal `SimulationResult` and assert exact Before Shift, After Shift, and Overall references/hits/page-faults/hit-ratio rows, including that the two phase totals sum to Overall. Train a small real model and assert `evaluate_policies` returns exactly `FIFO`, `LRU`, `Optimal`, and `Learned`, each with `len(events) == len(trace)`. Assert `ExperimentConfig` rejects the same invalid page/frame/reference bounds as the workload generator.

- [ ] **Step 2: Run focused tests and observe the expected import failure**

Run: `python -m unittest tests.test_experiment.ExperimentMetricTests -v`

Expected: FAIL because `src.experiment` does not exist.

- [ ] **Step 3: Implement configuration validation, aggregation, and common-trace policy evaluation**

Validate the same bounds as `generate_workload`. The CSV row schema is exactly: `algorithm`, `period`, `references`, `hits`, `page_faults`, `hit_ratio`, `pages`, `frames`, `seed`, `shift_index`, `trace_length`.

- [ ] **Step 4: Verify metric tests pass**

Run: `python -m unittest tests.test_experiment.ExperimentMetricTests -v`

Expected: all selected tests PASS.

- [ ] **Step 5: Write failing artifact integration tests**

Run `run_experiment` with 50 references per phase and a temporary output folder containing a space. Assert a 12-row CSV with the exact schema, three non-empty PNG files with exact names, finite ratios in `[0, 1]`, `evaluation_seed == 42`, `training_seeds == (43, 44, 45)`, and successful replacement on a second run. Run the same integration path with the minimum `pages=3, frames=2` configuration and assert it trains and evaluates successfully.

- [ ] **Step 6: Run integration tests and observe missing artifact failures**

Run: `python -m unittest tests.test_experiment -v`

Expected: metric tests PASS; artifact tests FAIL because `run_experiment` and plotting are incomplete.

- [ ] **Step 7: Implement experiment orchestration and three charts**

Generate evaluation seed `seed` and three training workloads at `seed + 1..3`; concatenate their candidate samples, train once, and evaluate all policies on the same evaluation tuple. Use matplotlib's non-interactive `Agg` backend. Create `workload_shift.png`, `hit_ratio_comparison.png`, and `page_faults_by_phase.png` with titles, axis labels, readable legends, and tight layout; close every figure after saving.

- [ ] **Step 8: Run experiment tests and the full suite**

Run: `python -m unittest tests.test_experiment -v`

Expected: all experiment tests PASS.

Run: `python -m unittest discover -s tests -v`

Expected: all tests PASS.

- [ ] **Step 9: Commit Task 4**

```text
git add src/experiment.py tests/test_experiment.py
git commit -m "feat: generate experiment metrics and figures"
```

### Task 5: Terminal Interface and End-to-End Entry Point

**Files:**
- Create: `main.py`
- Create: `tests/test_main.py`

**Interfaces:**
- Consumes: `ExperimentConfig` and `run_experiment` from Task 4.
- Produces: `parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace`; `prompt_for_int(label: str, default: int, predicate: Callable[[int], bool], error_message: str, input_fn: Callable[[str], str], output_fn: Callable[[str], None]) -> int`; `resolve_config(args: argparse.Namespace, input_fn: Callable[[str], str] = input, output_fn: Callable[[str], None] = print) -> ExperimentConfig`; `main(argv: Sequence[str] | None = None) -> int`.

- [ ] **Step 1: Write failing prompt, CLI, and validation tests**

Inject iterator-backed input and list-append output functions. Assert blank responses produce defaults; `"abc"` and out-of-range values print errors and retry; minimum `pages=3, frames=2` succeeds; and CLI values `--pages 8 --frames 3 --seed 7 --references-per-phase 50 --output-dir <temp>` produce the matching config without prompting. Assert argparse rejects invalid fully supplied combinations before `run_experiment` is called.

- [ ] **Step 2: Run main tests and observe the expected import failure**

Run: `python -m unittest tests.test_main -v`

Expected: FAIL because `main.py` does not exist.

- [ ] **Step 3: Implement parsing, prompts, validation, and console reporting**

With no page/frame/seed flags, show the approved three prompts. Support `--pages`, `--frames`, `--seed`, `--references-per-phase`, and `--output-dir`. Print configuration, shift index, training-sample diagnostic accuracy labelled `Training accuracy (diagnostic only)`, a pandas results table, and absolute output paths. Return exit code 0 on success.

- [ ] **Step 4: Run main tests and the full suite**

Run: `python -m unittest tests.test_main -v`

Expected: all terminal tests PASS.

Run: `python -m unittest discover -s tests -v`

Expected: all tests PASS.

- [ ] **Step 5: Run a short non-interactive smoke experiment**

Run: `python main.py --pages 8 --frames 3 --seed 7 --references-per-phase 50 --output-dir work/smoke-results`

Expected: exit code 0, a 12-row printed table, one CSV, and three named figures.

- [ ] **Step 6: Commit Task 5**

```text
git add main.py tests/test_main.py
git commit -m "feat: add interactive experiment runner"
```

### Task 6: Student Documentation, Default Artifacts, and Final Verification

**Files:**
- Create: `README.md`
- Create: `docs/demo_and_viva.md`
- Generate: `results/raw_results.csv`
- Generate: `results/figures/workload_shift.png`
- Generate: `results/figures/hit_ratio_comparison.png`
- Generate: `results/figures/page_faults_by_phase.png`

**Interfaces:**
- Consumes: the completed command-line program and all prior public interfaces.
- Produces: reproducible student-facing instructions, demonstration material, and default-run artifacts generated solely by `python main.py`.

- [ ] **Step 1: Write README with every assignment-required section**

Include title, course, track, problem, simple FIFO/LRU/Optimal explanations, one hand-worked example per policy, learned sample/features/label/training/prediction/leakage explanation, exact workload probabilities and shift diagram, metrics, project tree, VS Code setup, test/run/reproduction commands, output locations, limitations, and the required disclosure verbatim. State that generated artifacts are a reproducibility run and that final interpretation belongs to the author after verification.

- [ ] **Step 2: Write the 3–5 minute demo and exactly 15 viva questions**

The demo must cover page replacement, three classical policies, learned features and labels, midpoint shift, important files, `python main.py`, generated outputs, and a final instruction to report only the measured observation. Answers must be short, undergraduate-level, and must not claim that Learned wins.

- [ ] **Step 3: Run the complete test suite from a clean process**

Run: `python -m unittest discover -s tests -v`

Expected: every listed test reports `ok`, with `Ran N tests` and final `OK`.

- [ ] **Step 4: Generate default artifacts from the real program**

Run: `python main.py --pages 10 --frames 4 --seed 42 --references-per-phase 1000 --output-dir results`

Expected: exit code 0; CSV and all three figures exist and are non-empty. Record but do not interpret the measured values.

- [ ] **Step 5: Independently validate the generated CSV and images**

Run a read-only verification command that loads the CSV with pandas, asserts 12 rows and all accounting identities, opens each PNG with matplotlib/Pillow-compatible image reading, and prints dimensions. Expected: no assertion failure, 12 validated rows, and three positive image dimensions.

- [ ] **Step 6: Verify repository cleanliness and requirements coverage**

Run: `git diff --check`

Expected: no output.

Run: `git status --short`

Expected before the documentation commit: only the README, demo, and generated result artifacts are untracked/modified.

Review the specification checklist line-by-line against files and test names; add no unrequested frameworks or services.

- [ ] **Step 7: Commit documentation and generated artifacts**

```text
git add README.md docs/demo_and_viva.md results
git commit -m "docs: add reproducible results and student guide"
```

- [ ] **Step 8: Request independent whole-branch code review and resolve findings**

Provide the reviewer with the approved specification, this plan, and the full diff from the root design commit through `HEAD`. Fix every Critical or Important finding with a failing regression test first, then rerun the complete suite.

- [ ] **Step 9: Perform final fresh verification**

Run the complete test suite, default experiment, CSV/image validator, `git diff --check`, and `git status --short --branch` again. Completion requires fresh exit-code-zero evidence and a clean working tree.

- [ ] **Step 10: Upload to the configured GitHub repository**

Run: `git push -u origin main`

Expected: the remote `main` branch contains source, tests, README, requirements, raw results, and all three figures. If command-line authentication remains unavailable, use the user's already signed-in GitHub interface or ask the user for the smallest necessary authentication action; do not request or expose a personal access token in chat.
