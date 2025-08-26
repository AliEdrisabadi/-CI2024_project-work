# plot_utils.py
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe backend for headless use
import matplotlib.pyplot as plt

def _ensure_outdir(outdir: str) -> str:
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    return outdir

def _finite_clip(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.array([0.0])
    return np.clip(a, -1e6, 1e6)

def plot_residual_hist(y_true: np.ndarray, y_pred: np.ndarray, outdir: str, name: str, bins: int = 50) -> str:
    outdir = _ensure_outdir(outdir)
    resid = _finite_clip(y_true - y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(resid, bins=bins)
    ax.set_title(f"Residuals histogram: {name}")
    ax.set_xlabel("residual")
    ax.set_ylabel("count")
    fig.tight_layout()

    path = os.path.join(outdir, f"{name}_residual_hist.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    if not os.path.exists(path):
        raise RuntimeError(f"Could not save plot: {path}")
    return path

def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, outdir: str, name: str) -> str:
    outdir = _ensure_outdir(outdir)
    yt = _finite_clip(y_true)
    yp = _finite_clip(y_pred)
    n = min(len(yt), len(yp))
    yt, yp = yt[:n], yp[:n]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(yt, yp, s=6, alpha=0.6)

    lo = float(min(np.min(yt), np.min(yp)))
    hi = float(max(np.max(yt), np.max(yp)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = -1.0, 1.0

    ax.plot([lo, hi], [lo, hi], lw=1)  # y=x
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(f"Predicted vs True: {name}")
    ax.set_xlabel("true")
    ax.set_ylabel("pred")
    fig.tight_layout()

    path = os.path.join(outdir, f"{name}_pred_vs_true.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    if not os.path.exists(path):
        raise RuntimeError(f"Could not save plot: {path}")
    return path
