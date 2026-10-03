# Learned Page Replacement Experiment — Design Specification

**Course:** CSE-307 Operating Systems

**Track:** Track 1 — Learned Page Replacement

**Date:** 2026-10-03

## 1. Purpose

Build a small, understandable Python experiment that compares FIFO, LRU,
Optimal (Belady), and a decision-tree eviction policy on one held-out page
reference trace. The main research question is how each policy's measured
performance changes when the workload deliberately shifts halfway through the
trace from locality-heavy/sequential access to random/bursty access.

The project must run on a normal laptop from the VS Code terminal with
`python main.py`. It will not require Linux, Docker, a GPU, a database, a web
application, or deep-learning software.

## 2. Success Criteria

The completed project must:

- implement FIFO, LRU, and Optimal from scratch;
- train a lightweight `DecisionTreeClassifier` for eviction decisions;
- keep all learned-model input features limited to information available at
  decision time;
- generate a reproducible two-phase trace with one explicit midpoint shift;
- use the identical evaluation trace for all four policies;
- preserve memory state across the shift;
- measure hits, page faults, and hit ratio before the shift, after the shift,
  and overall;
- accept page count and frame count through a simple terminal interface;
- validate user input and provide documented defaults;
- generate real CSV results and three meaningful figures when executed;
- include automated tests, setup instructions, algorithm explanations, manual
  examples, limitations, a demo outline, and viva questions;
- contain no invented or hard-coded experimental results.

Writing the three-page term paper and drawing final experimental conclusions
are deliberately deferred until the author runs and verifies the generated
results.

## 3. User Interface and Configuration

Running `python main.py` with no arguments opens a terminal interaction:

```text
Learned Page Replacement Experiment
-----------------------------------
Number of unique pages [10]:
Number of frames [4]:
Random seed [42]:
```

Pressing Enter accepts a default. Invalid values produce a short explanation
and another prompt. The page count must be at least 3. The frame count must be
at least 2 and strictly smaller than the page count. Two frames are the minimum
because a one-frame system has no eviction choice and therefore cannot produce
both classes required by the candidate-wise classifier.

The same values can be supplied non-interactively for testing and exact
reproduction:

```text
python main.py --pages 10 --frames 4 --seed 42
```

The standard assignment configuration uses 10 unique pages, 4 frames, seed
42, and 1,000 references in each phase. Page and frame counts remain dynamic;
the standard values are defaults, not hard-coded algorithm assumptions.

## 4. Project Structure

```text
OS_B_TermPaper/
├── src/
│   ├── __init__.py
│   ├── page_replacement.py
│   ├── workload.py
│   ├── ml_model.py
│   └── experiment.py
├── tests/
│   ├── test_page_replacement.py
│   ├── test_workload.py
│   ├── test_ml_model.py
│   └── test_experiment.py
├── results/
│   ├── raw_results.csv
│   └── figures/
├── docs/
│   ├── demo_and_viva.md
│   └── superpowers/
│       ├── specs/
│       └── plans/
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 5. Architecture

### `src/page_replacement.py`

Contains independent simulations for FIFO, LRU, and Optimal, plus shared
result/event data structures. Every simulation accepts a reference trace and
frame count and records one hit/fault event per reference. FIFO maintains
arrival order. LRU maintains last-access times. Optimal inspects only the
remaining evaluation trace and evicts the resident page whose next use is
farthest away (or absent).

Optimal is an offline theoretical benchmark. Its access to the future is
expected and is not available to any practical policy.

### `src/workload.py`

Generates deterministic training and evaluation traces from explicit seeds.
It returns both the trace and the midpoint shift index.

### `src/ml_model.py`

Builds training samples, trains a small decision tree, and simulates the
learned eviction policy. Training-label generation is separate from runtime
prediction so the future-information boundary is visible and testable.

### `src/experiment.py`

Runs the four policies on the same evaluation trace, aggregates phase and
overall metrics, saves `results/raw_results.csv`, and creates the figures.

### `main.py`

Parses optional command-line arguments, prompts for missing configuration,
validates values, calls the experiment, and prints a concise results table and
output paths.

## 6. Workload Shift

The shift is the central experimental variable.

```text
Phase 1: locality-heavy / mostly sequential
references 0 ... 999
                         ↓ shift_index = 1000
Phase 2: random / bursty
references 1000 ... 1999
```

### Phase 1 — Locality-heavy/sequential

The active working-set size is
`min(page_count, max(2, min(frame_count, 4)))`. With 85% probability, the
generator advances to the next page in that working set's cycle. With 15%
probability, it selects uniformly from the complete page universe. This
produces recognizable repetition without making the phase perfectly
deterministic or engineering a particular winner.

### Phase 2 — Random/bursty

The generator uses the complete configured page universe. Each reference is
uniformly random with 70% probability. The other 30% is selected uniformly
from a temporary hot set of `min(3, page_count)` pages. The hot set is sampled
again without replacement every 25 references. This represents a clear
distribution change while avoiding a trace tailored to any replacement
policy.

### Fairness and reproducibility

- Page identifiers are integers from 1 through the configured page count.
- The shift occurs exactly at half the evaluation trace.
- The memory contents are not cleared at the shift.
- FIFO, LRU, Optimal, and Learned receive the same trace object/value sequence.
- NumPy's explicitly seeded random generator produces the trace.
- Training traces use deterministic seeds distinct from the evaluation seed.
- A workload figure marks the shift and makes the two access patterns visible.

## 7. Learned Eviction Policy

### Learning problem

The model is a candidate-wise binary classifier. At each training-time page
fault where memory is full, every resident page becomes one training sample.

### Input features

Each sample contains only values known at that decision point:

1. `recency`: references elapsed since the page was last accessed;
2. `total_frequency`: number of accesses to the page so far;
3. `recent_frequency`: accesses to the page within the previous 20 references;
4. `residency_time`: references elapsed since the page entered its current
   frame.

Page identity and any future-use value are not model features. A decision tree
does not require feature scaling.

### Target label

For one full-memory fault, Belady identifies one deterministic reference
victim. That candidate receives label `1`; the other resident candidates
receive label `0`. If multiple pages are never used again or have equal
farthest-next-use positions, the lowest page identifier is the deterministic
victim.

Belady's future information is used only to construct offline labels from
separate training traces. It is never passed to `fit` as a feature and is never
consulted while simulating the learned policy on the evaluation trace.

### Training

The project generates three separate training traces with seeds
`evaluation_seed + 1`, `evaluation_seed + 2`, and `evaluation_seed + 3`.
They use the same documented workload family and length but contain different
exact references from the held-out evaluation trace. A
`DecisionTreeClassifier` uses `max_depth=5`, `min_samples_leaf=5`, balanced
class weights, and `random_state=evaluation_seed`. The model remains
intentionally small enough to inspect and explain.

Training accuracy may be printed as a diagnostic and clearly identified as
training accuracy; it is not an experimental performance claim.

### Runtime eviction

At an evaluation-time fault with full memory, the learned policy constructs
the four past-only features for every resident candidate. It asks the tree for
the probability of label `1` and evicts the highest-scoring candidate.
Prediction ties are resolved deterministically by greater recency and then
page number. The model is not retrained on the evaluation trace.

This policy is adaptive in the limited, explicit sense that its decisions
respond to changing online recency and frequency state. It is not an online
learning system, and the documentation will not claim otherwise.

## 8. Metrics

Each simulation emits a hit/fault sequence aligned one-to-one with the
evaluation references. Aggregation slices that sequence at `shift_index`:

- **Before shift:** events `[0:shift_index]`;
- **After shift:** events `[shift_index:end]`;
- **Overall:** all events.

For every algorithm and period:

```text
references = hits + page_faults
hit_ratio = hits / references
```

The CSV contains one row per algorithm/period combination and configuration
columns including pages, frames, seed, trace length, and shift index. A derived
shift effect may be reported as after-shift hit ratio minus before-shift hit
ratio, but it never replaces the required raw metrics.

## 9. Output Artifacts

Execution creates directories when needed and replaces artifacts for the
current configuration:

- `results/raw_results.csv` — before, after, and overall metrics;
- `results/figures/workload_shift.png` — page references over time with a
  visible midpoint marker;
- `results/figures/hit_ratio_comparison.png` — grouped before/after/overall
  hit ratios for all policies;
- `results/figures/page_faults_by_phase.png` — before/after page-fault counts.

All values and charts are computed from the current run. No result number is
embedded in source code or documentation.

## 10. Testing Strategy

Tests use Python's built-in `unittest` so no testing framework is added to the
requirements. Hand-calculated traces verify the exact hit/fault counts and,
where relevant, victim behavior of FIFO, LRU, and Optimal. Further tests cover:

- one recorded event for every reference;
- configurable frame counts;
- invalid frame/page configurations;
- identical workloads for identical seeds and different workloads for
  different seeds;
- a midpoint shift and measurable change in workload characteristics;
- feature values derived from history only;
- one positive Belady label per full-memory training decision;
- learned prediction and deterministic tie-breaking;
- phase metrics summing to overall metrics;
- all four policies consuming the same evaluation trace;
- CSV and three non-empty figure files produced in a temporary directory.

Tests will be written and observed failing before their corresponding
production behavior is implemented.

## 11. Error Handling

- Terminal input rejects non-integers and out-of-range values with a clear
  retry message.
- Programmatic functions raise `ValueError` for empty traces, invalid page
  identifiers, or invalid frame/page relationships.
- File output creates missing directories and reports the final absolute paths.
- A model without both training classes causes an explicit error rather than a
  misleading experiment.

## 12. Documentation and Demonstration

The README will explain the problem, algorithms, learned component, leakage
boundary, workload shift, metrics, setup, execution, reproduction, structure,
manual examples, tests, limitations, and required AI-assistance disclosure:

> AI tools were used for implementation guidance, debugging assistance, and
> code/documentation review. Experimental execution, result generation,
> interpretation, and final analysis were performed and verified by the
> author.

`docs/demo_and_viva.md` will provide a 3–5 minute demonstration sequence and
15 short viva questions and answers. Its demo script tells the author where to
insert the verified observation, but does not supply an unmeasured conclusion.

## 13. Limitations

- The workload is synthetic and is not a production operating-system trace.
- The model is a small offline-trained decision tree, not continuous online
  learning.
- Belady is not implementable as an online OS policy because it requires future
  knowledge.
- Results depend on selected pages, frames, trace length, seed, and workload
  generator.
- One default run illustrates behavior but is not sufficient for broad
  statistical claims.
- Page replacement overhead, dirty pages, I/O latency, and hardware effects are
  outside scope.

## 14. Out of Scope for This Stage

- GUI, web interface, API, database, Docker, VM, or Linux-only setup;
- neural networks or large ML frameworks;
- claims that Learned must outperform a classical policy;
- fabricated observations, screenshots, citations, or final conclusions;
- the final three-page PDF before author-verified results are available.
