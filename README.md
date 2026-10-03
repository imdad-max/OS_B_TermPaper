# CSE-307 Operating Systems Term Paper

## Track 1 — Learned Page Replacement

This small Python project compares three classical page-replacement policies
with a lightweight learned eviction policy. Its central experiment measures
what happens when the page-reference workload deliberately changes halfway
through one continuous simulation.

The project is designed for a normal laptop and the VS Code terminal. It does
not require Linux, a virtual machine, Docker, a GPU, a database, a web
application, or deep learning.

## Problem Description

An operating system keeps a limited number of pages in physical-memory frames.
When a requested page is not resident, a **page fault** occurs. If every frame
is occupied, the replacement policy must choose one resident page to evict.
A useful policy tries to reduce faults without using information that a real
online system could not know.

This experiment compares:

- FIFO;
- LRU;
- Optimal (Belady's theoretical offline policy);
- a learned policy using `DecisionTreeClassifier`.

Every policy receives the same evaluation trace, frame count, and initial
empty memory. Memory is not reset at the workload shift.

## Classical Algorithms

### FIFO

First-In, First-Out evicts the page that entered memory first. A hit does not
change the arrival order. FIFO is simple but ignores how recently or often a
page has been used.

### LRU

Least Recently Used evicts the page whose last access occurred farthest in the
past. It uses access history and often works well when recent locality
continues.

### Optimal / Belady

Optimal evicts the resident page whose next use is farthest in the future. A
page that is never used again is the strongest eviction candidate. This gives
a theoretical lower bound on faults, but a real online operating system does
not know future references.

### Small hand-worked example

Use three frames and this reference string:

```text
8 1 2 3 1 4 1 5 3 4 1 4 3
```

The values below are manually derived examples, not experimental results:

| Policy | Hits | Faults | Eviction order |
|---|---:|---:|---|
| FIFO | 3 | 10 | 8, 1, 2, 3, 4, 1, 5 |
| LRU | 4 | 9 | 8, 2, 3, 4, 1, 5 |
| Optimal | 6 | 7 | 2, 8, 1, 5 |

For example, when page `3` first arrives, FIFO and LRU evict `8`. Optimal
instead evicts `2`, because among the resident candidates its next use is
farthest away. Automated tests check these counts and victim rules.

## Learned Eviction Component

The learned method is a candidate-wise binary classifier.

1. **One training sample:** one resident page being considered at one
   full-memory page fault.
2. **Input features:** recency, total frequency so far, frequency within the
   previous 20 references, and time in the current frame.
3. **Target:** `1` if the page is Belady's selected victim; otherwise `0`.
4. **Label generation:** Belady examines the future of a separate training
   trace and supplies exactly one positive candidate per eviction decision.
5. **Training:** three separately seeded training traces feed a depth-5
   `DecisionTreeClassifier` with balanced class weights.
6. **Prediction:** on an evaluation fault, the model scores every resident
   page; the page with the highest probability of label `1` is evicted.
7. **Tie-breaking:** greater recency wins, followed by the lower page number.

A decision tree is appropriate because it is small, fast on a CPU, handles
nonlinear feature rules, needs no feature scaling, and remains inspectable.

### No future-information leakage

Future references are used only for:

- the standalone Optimal benchmark; and
- offline labels on separate training traces.

They are never model features and are never consulted during learned-policy
evaluation. The evaluation trace uses the requested seed, while training uses
`seed + 1`, `seed + 2`, and `seed + 3`.

The model is trained once. It reacts to changing online recency and frequency
state, but it does not retrain itself during the evaluation trace.

## Workload Design and Shift

The default trace contains 2,000 references and shifts exactly at index 1,000:

```text
Phase 1: locality-heavy / sequential
references 0–999
                         ↓ WORKLOAD SHIFT
Phase 2: random / bursty
references 1000–1999
```

### Phase 1

- Active-set size: `min(page_count, max(2, min(frame_count, 4)))`.
- 85% probability: advance through that active set sequentially.
- 15% probability: select from the complete page universe.

### Phase 2

- 70% probability: select uniformly from the complete page universe.
- 30% probability: select from a temporary hot set of up to three pages.
- The temporary hot set changes every 25 references.

The generator uses NumPy's seeded random generator. It does not inspect any
replacement policy, so the trace is not constructed to force a winner.

## Metrics

For every policy, the program records these metrics before the shift, after
the shift, and overall:

- hits;
- page-fault count;
- hit ratio, calculated as `hits / references`.

For every row, `hits + page_faults = references`. The results are split by the
shift index without clearing resident frames.

## Project Structure

```text
OS_B_TermPaper/
├── src/
│   ├── page_replacement.py
│   ├── workload.py
│   ├── ml_model.py
│   └── experiment.py
├── tests/
│   ├── test_page_replacement.py
│   ├── test_workload.py
│   ├── test_ml_model.py
│   ├── test_experiment.py
│   └── test_main.py
├── results/
│   ├── raw_results.csv
│   └── figures/
├── docs/
│   ├── demo_and_viva.md
│   └── superpowers/
├── main.py
├── requirements.txt
└── README.md
```

## VS Code Setup

Requirements: Python 3.10 or newer and VS Code with the Microsoft Python
extension.

Open PowerShell in the project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

In VS Code, press `Ctrl+Shift+P`, select **Python: Select Interpreter**, and
choose `.venv\Scripts\python.exe`.

If PowerShell activation is restricted, activation is optional. Use the venv
interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

## How to Run

Interactive terminal interface:

```powershell
python main.py
```

The program prompts for page count, frame count, and seed. Press Enter to use
the defaults. Valid configurations require at least 3 pages and
`2 <= frames < pages`.

Example non-interactive run:

```powershell
python main.py --pages 8 --frames 3 --seed 7
```

## How to Reproduce the Default Results

Run the tests, then execute the exact default configuration:

```powershell
python -m unittest discover -s tests -v
python main.py --pages 10 --frames 4 --seed 42 --references-per-phase 1000 --output-dir results
```

Running the same command again replaces the current result artifacts with a
fresh deterministic run.

## Generated Results

The program creates:

```text
results/raw_results.csv
results/figures/workload_shift.png
results/figures/hit_ratio_comparison.png
results/figures/page_faults_by_phase.png
```

The committed files in `results/` are generated by the default command above;
they are not hard-coded. They are provided as a reproducibility run, not as a
substitute for the author's own execution and verification.

When writing the paper, separate claims carefully:

- **FACT:** a number read from the generated CSV.
- **INTERPRETATION:** the likely operating-system reason for that number.

Do not assume the learned policy must outperform a classical policy.

## Limitations

- The trace is synthetic rather than a production operating-system trace.
- The decision tree is trained offline and does not perform online retraining.
- Optimal is not practical online because it requires future knowledge.
- Results depend on the chosen pages, frames, phase length, seed, and generator.
- One seed illustrates behavior but does not support broad statistical claims.
- The simulator does not model dirty pages, I/O latency, replacement overhead,
  processes, or hardware-specific effects.

## AI-Assistance Disclosure

> AI tools were used for implementation guidance, debugging assistance, and code/documentation review. Experimental execution, result generation, interpretation, and final analysis were performed and verified by the author.
