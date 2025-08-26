# eval_and_plot.py
import os
import argparse
import importlib.util
import numpy as np
from plot_utils import plot_residual_hist, plot_pred_vs_true

def load_submission(module_path: str):
    module_path = os.path.abspath(module_path)
    spec = importlib.util.spec_from_file_location("submission_mod", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    funcs = []
    for i in range(1, 8+1):
        fn = getattr(mod, f"f{i}", None)
        if fn is None:
            raise AttributeError(f"Function f{i} not found in {module_path}")
        funcs.append(fn)
    return funcs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--problems", required=True, help="comma-separated list, e.g. 1,2,3")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--submission", default="s316628.py", help="path to submission .py")
    args = ap.parse_args()

    cwd = os.path.abspath(os.getcwd())
    data_dir = os.path.abspath(args.data_dir)
    outdir = os.path.abspath(args.outdir)
    submission = os.path.abspath(args.submission)

    os.makedirs(outdir, exist_ok=True)

    print(f"[info] cwd         = {cwd}")
    print(f"[info] data_dir    = {data_dir}")
    print(f"[info] outdir      = {outdir}")
    print(f"[info] submission  = {submission}")

    funcs = load_submission(submission)
    problems = [int(p.strip()) for p in args.problems.split(",") if p.strip()]

    for p in problems:
        name = f"problem_{p}"
        npz_path = os.path.join(data_dir, f"{name}.npz")
        if not os.path.exists(npz_path):
            print(f"[warn] not found: {npz_path}")
            continue

        data = np.load(npz_path)
        x, y = data["x"], data["y"]
        print(f"[{name}] x.shape={x.shape} y.shape={y.shape}")

        # run model
        y_pred = funcs[p-1](x.copy())

        # shape sanity
        y_pred = np.asarray(y_pred).reshape(-1)
        y      = np.asarray(y).reshape(-1)
        if y_pred.shape != y.shape:
            n = min(len(y_pred), len(y))
            print(f"[warn] shape mismatch: y_pred={y_pred.shape} y={y.shape} -> trunc to {n}")
            y_pred = y_pred[:n]
            y = y[:n]

        
        y_pred_safe = np.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=-1e6)
        mse = float(np.mean((y - y_pred_safe)**2))
        print(f"[{name}] MSE={mse:g}")

        # plots
        p1 = plot_residual_hist(y, y_pred_safe, outdir, name)
        p2 = plot_pred_vs_true(y, y_pred_safe, outdir, name)
        print(f"[{name}] plots saved:\n  - {p1}\n  - {p2}")

    print("[done]")

if __name__ == "__main__":
    main()
