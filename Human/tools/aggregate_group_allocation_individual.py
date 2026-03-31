# file: scripts/analyze_entropy.py
"""Individual-level entropy analysis for group allocation results (science-style figures).

- Input: root folder(s). Recursively find all `group_allocation.csv` files created earlier.
- Read `id` and `entropy` per participant (skip the `average` row).
- Report extreme-zero entropy cases (all slots to one group): count and percentage.
- Produce publication-friendly histogram PDF (+ optional ECDF panel).

Usage:
    python scripts/analyze_entropy.py /path/to/folder1 [/path/to/folder2 ...] \
        --outdir reports --bins 30 --zero-thresh 1e-14 --save-csv --ecdf --density
"""
from __future__ import annotations

import argparse
import math
import os
from typing import List, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# ---- Plot style (science-like, clean, no background grid) ----
mpl.rcParams.update({
    "font.family": "Arial",
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.grid": False,
})

# Use a blue gradient colormap for science-style look
COLORMAP = plt.cm.Blues


def _find_allocation_csvs(root_dirs: List[str]) -> List[str]:
    out: List[str] = []
    for root in root_dirs:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn == "group_allocation.csv":
                    out.append(os.path.join(dirpath, fn))
    return out


def _load_entropies(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"id": str})
    df = df[df["id"].astype(str).str.lower() == "average"].copy()
    df["entropy"] = pd.to_numeric(df["entropy"], errors="coerce")
    df = df.dropna(subset=["entropy"]).reset_index(drop=True)
    df["source"] = os.path.dirname(path)
    return df[["id", "entropy", "source"]]


def analyze_entropies(dfs: List[pd.DataFrame], zero_thresh: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["id", "entropy", "source"])
    all_df = all_df.dropna(subset=["entropy"]).reset_index(drop=True)
    all_df["is_zero"] = all_df["entropy"].le(zero_thresh)

    total = len(all_df)
    zero_n = int(all_df["is_zero"].sum())
    zero_pct = (zero_n / total * 100.0) if total else 0.0
    overall = pd.DataFrame({
        "metric": ["participants", "zero_entropy_count", "zero_entropy_percent", "mean_entropy", "median_entropy"],
        "value": [total, zero_n, zero_pct, float(all_df["entropy"].mean()) if total else 0.0, float(all_df["entropy"].median()) if total else 0.0],
    })
    return all_df, overall


def _ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    n = x.size
    y = np.arange(1, n + 1) / n
    return x, y


def plot_entropy_fig(entropies: pd.Series, outpath_base: str, bins: int, density: bool, show_ecdf: bool, zero_pct: float) -> None:
    if entropies.empty:
        print("No entropies to plot.")
        return

    max_entropy = math.log(10, 2)
    ncols = 2 if show_ecdf else 1
    fig, axes = plt.subplots(1, ncols, figsize=(12, 5), squeeze=False)
    ax = axes[0, 0]

    # Histogram and manual per-bar coloring
    counts, bin_edges, patches = ax.hist(entropies, bins=bins, density=density)

    # Identify the bin containing 0 (treat as the "zero" bar)
    zero_bin_idx = 0
    for i in range(len(bin_edges) - 1):
        if bin_edges[i] <= 0 < bin_edges[i + 1]:
            zero_bin_idx = i
            break

    # Color rules: zero-bin = yellow; others = blue gradient based on their heights
    nonzero_vals = [counts[i] for i in range(len(counts)) if i != zero_bin_idx]
    vmax = max(nonzero_vals) if nonzero_vals else (max(counts) if len(counts) else 1)
    norm = plt.Normalize(vmin=0, vmax=vmax if vmax > 0 else 1)

    for i, (c, p) in enumerate(zip(counts, patches)):
        if i == zero_bin_idx:
            plt.setp(p, "facecolor", "#FFD700")  # yellow for exact-zero bin
        else:
            plt.setp(p, "facecolor", COLORMAP(norm(c)))

    ax.set_xlabel("Entropy")
    ax.set_ylabel("Density" if density else "Count")
    # ax.set_title(f"Distribution of Individual Entropy  •  Zero-entropy: {zero_pct:.1f}%")
    ax.set_xlim(0, max_entropy * 1.02)
    ax.set_ylim(0, 140)
    ax.axvline(0, linestyle=":", linewidth=0.9, color="black")
    ax.axvline(max_entropy, linestyle=":", linewidth=0.9, color="black")

    m = float(entropies.mean())
    med = float(entropies.median())
    ax.axvline(m, linestyle="-.", linewidth=0.9, color="black")
    ax.axvline(med, linestyle="-.", linewidth=0.9, color="black")
    ax.text(m, 135, "mean", rotation=90, va="top", ha="left", fontsize=12)
    ax.text(med, 135, "median", rotation=90, va="top", ha="left", fontsize=12)

    if show_ecdf:
        ax2 = axes[0, 1]
        x, y = _ecdf(entropies.to_numpy(dtype=float))
        ax2.plot(x, y, color="darkblue")
        ax2.set_xlabel("Entropy (bits)")
        ax2.set_ylabel("ECDF")
        ax2.set_title("Empirical CDF of Entropy")
        ax2.set_xlim(0, max_entropy * 1.02)
        ax2.set_ylim(0, 1)

    fig.tight_layout()
    pdf_path = outpath_base + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {pdf_path}")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze individual entropies from group_allocation.csv files.")
    parser.add_argument("folders", nargs="+", help="Root folder(s) to scan recursively.")
    parser.add_argument("--outdir", default="reports", help="Output directory for figures and summaries.")
    parser.add_argument("--bins", type=int, default=30, help="Number of bins for histogram.")
    parser.add_argument("--zero-thresh", type=float, default=1e-14, help="Threshold to treat entropy as zero.")
    parser.add_argument("--save-csv", action="store_true", help="Save concatenated entropies and summary as CSV.")
    parser.add_argument("--ecdf", action="store_true", help="Add an ECDF panel alongside the histogram.")
    parser.add_argument("--density", action="store_true", help="Plot normalized histogram (density) instead of counts.")
    args = parser.parse_args(argv)

    files = _find_allocation_csvs(args.folders)
    if not files:
        print("No group_allocation.csv found.")
        return

    dfs: List[pd.DataFrame] = []
    skipped = 0
    for f in files:
        try:
            dfs.append(_load_entropies(f))
        except Exception as e:
            skipped += 1
            print(f"Skip {f}: {e}")

    all_df, overall = analyze_entropies(dfs, zero_thresh=args.zero_thresh)

    os.makedirs(args.outdir, exist_ok=True)
    hist_out_base = os.path.join(args.outdir, "entropy_distribution")
    zero_pct = float(overall.loc[overall["metric"] == "zero_entropy_percent", "value"].iloc[0]) if not overall.empty else 0.0
    plot_entropy_fig(all_df["entropy"], hist_out_base, bins=args.bins, density=args.density, show_ecdf=args.ecdf, zero_pct=zero_pct)

    if args.save_csv:
        ent_csv = os.path.join(args.outdir, "individual_entropies.csv")
        sum_csv = os.path.join(args.outdir, "summary_overall.csv")
        all_df.to_csv(ent_csv, index=False)
        overall.to_csv(sum_csv, index=False)
        print(f"Saved CSVs: {ent_csv}, {sum_csv}")

    total = int(overall.loc[overall["metric"] == "participants", "value"].iloc[0]) if not overall.empty else 0
    zero_n = int(overall.loc[overall["metric"] == "zero_entropy_count", "value"].iloc[0]) if not overall.empty else 0
    zero_pct = float(overall.loc[overall["metric"] == "zero_entropy_percent", "value"].iloc[0]) if not overall.empty else 0.0
    print(f"Participants: {total}\nZero-entropy: {zero_n} ({zero_pct:.2f}%)\nSkipped files: {skipped}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])