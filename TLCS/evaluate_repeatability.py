# evaluate_repeatability.py

import numpy as np

from testing_main import run_training_once

def summarize(runs):
    """Given a list of dicts, compute mean±std per key."""
    summary = {}
    for key in runs[0]:
        arr = np.array([r[key] for r in runs], dtype=float)
        summary[key] = (arr.mean(), arr.std())
    return summary

if __name__ == "__main__":
    N = 5
    base_seed = 42
    runs = []

    for i in range(N):
        seed = base_seed + i * 100
        print(f"\n=== Run {i+1}/{N} (seed={seed}) ===")
        metrics = run_training_once(seed)
        for k,v in metrics.items():
            print(f"{k:30s}: {v:.4f}")
        runs.append(metrics)

    print("\n=== Summary (mean ± std) ===")
    summary = summarize(runs)
    for k,(μ,σ) in summary.items():
        print(f"{k:30s}: {μ:.4f} ± {σ:.4f}")
