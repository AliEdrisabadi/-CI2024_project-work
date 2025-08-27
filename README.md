# CI2024 – Symbolic Regression (s316628)

This repository contains a compact pipeline to **evaluate** and **train/export** a submission for the CI2024 Symbolic Regression project.

---

## Contents

This repo contains:
- `s316628.py` — submission file with eight functions `f1..f8`
- `train_and_export.py` —  trainer + exporter 
- `eval_and_plot.py` — evaluator + plotting
- `plot_utils.py` — plotting helpers
- `data/` — datasets (`problem_1.npz` … `problem_8.npz`)
- `plots/` — output folder for generated figures 
- `Report.pdf` — project report 

---

1) Python
   • Use Python 3.12+.

2) Install dependencies
   pip install numpy matplotlib


-------------------------------------

### Protected math (stability)

- `EPS = 1e-9`
- `pdiv(a, b)` → safe division (denominator clipped away from zero)
- `plog(a)` → `log(|a| + EPS)`
- `psqrt(a)` → `sqrt(|a|)`
- `pexp(a)` → `exp(clip(a, -20, 20))`
- `ppow(a, b)` → power with base/exponent clipping
- `_sanitize(y)` → replace NaN/Inf with finite values (used on outputs)

These guards keep evaluation and plotting stable even for extreme inputs.

### Results 
Results
In the following, the outcomes found for each problem can be observed:

| Problem | X shape | y shape | Variables | MSE (eval) |
|---|---|---|---|---|
| P1 | (1, 500) | (500,) | ['x0'] | 7.12594e-34 |
| P2 | (3, 5000) | (5000,) | ['x0', 'x1', 'x2'] | 9.56306e+12 |
| P3 | (3, 5000) | (5000,) | ['x0', 'x1', 'x2'] | 117.037 |
| P4 | (2, 5000) | (5000,) | ['x0', 'x1'] | 3.71471 |
| P5 | (2, 5000) | (5000,) | ['x0', 'x1'] | 2.23596e-18 |
| P6 | (2, 5000) | (5000,) | ['x0', 'x1'] | 2.53209e-06 |
| P7 | (2, 5000) | (5000,) | ['x0', 'x1'] | 54.3914 |
| P8 | (6, 50000) | (50000,) | ['x0', 'x1', 'x2', 'x3', 'x4', 'x5'] | 700139 |

