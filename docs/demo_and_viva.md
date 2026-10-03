# 3–5 Minute Demonstration and Viva Guide

## Demonstration Sequence

### 0:00–0:30 — Define page replacement

Explain that physical memory has limited frames. A missing referenced page
causes a page fault; when frames are full, a policy selects a victim page.

### 0:30–1:10 — Introduce the policies

- FIFO removes the oldest-loaded page.
- LRU removes the least recently accessed page.
- Optimal removes the page used farthest in the future and is a theoretical
  benchmark.
- Learned scores current resident pages using a small decision tree.

Show `src/page_replacement.py` briefly.

### 1:10–1:50 — Explain the learned method

Show `src/ml_model.py`. One candidate page is one training row. Point to the
four features: recency, total frequency, recent-window frequency, and residency
time. Explain that Belady creates labels only on separate training traces;
future use is not a prediction feature.

### 1:50–2:25 — Show the workload shift

Show `src/workload.py` and the generated `workload_shift.png` figure. State the
exact midpoint and explain Phase 1's 85/15 sequential/noise mixture and Phase
2's 70/30 random/burst mixture. Emphasize that memory is not reset.

### 2:25–3:10 — Run the project

In the VS Code terminal, run:

```powershell
python main.py
```

Accept the defaults or enter a page and frame count. Show the printed
before-shift, after-shift, and overall table.

### 3:10–3:45 — Show artifacts

Open:

```text
results/raw_results.csv
results/figures/hit_ratio_comparison.png
results/figures/page_faults_by_phase.png
```

Explain that every file is generated from the run and no metric is hard-coded.

### 3:45–4:30 — State the measured observation

Read the verified CSV before presenting. Identify which policy has the lowest
faults and highest hit ratio in each phase, then state which policy's hit ratio
changes most. Separate the measured fact from the likely locality-based reason.
Do not claim that Learned is always superior.

## Fifteen Likely Viva Questions

1. **What is a page fault?**

   It occurs when a referenced page is not currently in a physical-memory frame.

2. **Why is page replacement needed?**

   When all frames are occupied, one resident page must leave before a missing page can enter.

3. **How does FIFO choose a victim?**

   It removes the page that entered memory earliest.

4. **How does LRU choose a victim?**

   It removes the page whose last access is oldest.

5. **Why is Optimal useful if it cannot be implemented online?**

   It provides a theoretical benchmark for the smallest achievable fault count on a known trace.

6. **What does one ML training sample represent?**

   It represents one resident page considered during one full-memory fault.

7. **What are the model's input features?**

   Recency, total frequency, frequency in the previous 20 references, and residency time.

8. **What is the target label?**

   Label `1` means Belady selected that resident page for eviction; otherwise it is `0`.

9. **Does the learned model receive future information?**

   No. Future information creates offline training labels only and is not a prediction feature.

10. **Why use a decision tree?**

    It is lightweight, interpretable, CPU-friendly, and can learn nonlinear feature rules.

11. **What is the workload shift?**

    The trace changes halfway from mostly sequential locality to random and temporary burst access.

12. **Is memory cleared at the shift?**

    No. The same simulation continues so only the access distribution changes.

13. **How is the comparison kept fair?**

    Every policy receives the identical seeded evaluation trace and frame count.

14. **Is the learned policy guaranteed to win?**

    No. Its performance depends on its features, training distribution, and the evaluated workload.

15. **What is the main limitation?**

    The experiment uses a synthetic trace and a small offline-trained model rather than a complete operating system.
