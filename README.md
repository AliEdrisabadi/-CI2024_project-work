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

2) Install minimal dependencies
   pip install numpy matplotlib

---

-------------------------------------
- Uses protected math to avoid NaN/Inf and numeric blow-ups:
    EPS = 1e-9
    pdiv(a,b)  -> safe division (denominator clipped away from zero)
    plog(a)    -> log(|a| + EPS)
    psqrt(a)   -> sqrt(|a|)
    pexp(a)    -> exp(clip(a, -20, 20))
    ppow(a,b)  -> power with base/exponent clipping
    _sanitize  -> replaces NaN/Inf with finite values (used on outputs)
- This keeps evaluation and plotting stable even for extreme inputs.

