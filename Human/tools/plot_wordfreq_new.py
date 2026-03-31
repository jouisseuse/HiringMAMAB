"""Plot science-style word-frequency bar charts

Stacked mode:
1) Sort words by total across conditions from big to small
2) Keep top 20
3) No x-axis tick labels (below the axis) for a cleaner look
4) Use a Nature-like scientific color palette
5) Show white numbers on each bar segment indicating quantity, placed at the bottom-right corner of each segment
"""

from __future__ import annotations
import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
})

def _read_and_validate(csv_file: str) -> pd.DataFrame:
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File not found: {csv_file}")
    df = pd.read_csv(csv_file)
    if df.empty:
        return df
    need = {"word", "frequency"}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {need}: {csv_file}")
    df["word"] = df["word"].astype(str)
    df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce").fillna(0.0)
    return df

# Nature-style color palette (inspired by Nature publishing group figures)
def _nature_palette(n: int) -> List[str]:
    base = [
        "#4DBBD5",  # blue
        "#E64B35",  # red
        "#00A087",  # green
        "#3C5488",  # dark blue
        "#F39B7F",  # orange
        "#8491B4",  # gray blue
        "#91D1C2",  # teal
        "#DC0000",  # bright red
    ]
    return [base[i % len(base)] for i in range(n)]

def _pick_top_words_by_total(dfs: List[pd.DataFrame], topk: int) -> List[str]:
    totals = {}
    for df in dfs:
        if df.empty:
            continue
        s = df.groupby("word")["frequency"].sum()
        for w, v in s.items():
            totals[w] = totals.get(w, 0.0) + float(v)
    ordered = [w for w, _ in sorted(totals.items(), key=lambda x: x[1], reverse=True)]
    return ordered[:topk]

def plot_stacked(
    csv_files: List[str],
    names: List[str] | None,
    topk: int,
    output_dir: str,
    save_format: str = "pdf",
    width: float = 7.8,
    height_per_row: float = 0.44,
) -> str:
    dfs = [_read_and_validate(p) for p in csv_files]
    if names is None or len(names) != len(csv_files):
        names = [os.path.splitext(os.path.basename(p))[0] for p in csv_files]

    words = _pick_top_words_by_total(dfs, topk)
    n_rows = len(words)
    mat = np.zeros((n_rows, len(dfs)))
    for j, df in enumerate(dfs):
        s = df.groupby("word")["frequency"].sum()
        mat[:, j] = [float(s.get(w, 0.0)) for w in words]

    height = max(3.0, n_rows * height_per_row)
    fig, ax = plt.subplots(figsize=(width, height))

    colors = _nature_palette(len(dfs))
    left = np.zeros(n_rows)
    containers = []
    for j in range(len(dfs)):
        h = ax.barh(words, mat[:, j], left=left, height=0.58, color=colors[j], label=names[j])
        containers.append(h)
        left += mat[:, j]

    # Show axis line but remove x-axis tick labels
    ax.tick_params(axis='x', which='both', labelbottom=False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_visible(False)

    ax.invert_yaxis()
    ax.xaxis.grid(True, linestyle=(0, (4, 4)), linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)

    # Add white numbers at bottom-right corner of each bar segment
    for j, bars in enumerate(containers):
        for i, rect in enumerate(bars):
            seg = mat[i, j]
            if seg <= 0:
                continue
            x = rect.get_x() + rect.get_width() / 2.0
            y = rect.get_y() + rect.get_height() / 2.0
            ax.text(x, y, f"{int(round(seg))}", ha="center", va="center", fontsize=8, color="white")
            
    ax.legend(loc="lower center", bbox_to_anchor=(0.68, 0.02), ncol=min(len(dfs), 4), frameon=False)

    fig.subplots_adjust(left=0.33, right=0.98, top=0.90, bottom=0.10)

    os.makedirs(output_dir, exist_ok=True)
    joined = "_vs_".join([os.path.splitext(os.path.basename(p))[0] for p in csv_files])
    base = f"stacked_strategy"[:200]
    out_path = os.path.join(output_dir, f"{base}.{save_format.lower()}")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved stacked plot: {out_path}")
    return out_path

def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot science-style bar charts from word frequency CSVs.")
    parser.add_argument("csv_files", nargs="+", help="Word frequency CSV files to plot.")
    parser.add_argument("--topk", type=int, default=20, help="Top-K words sorted by total.")
    parser.add_argument("--outdir", default="plots", help="Directory to save plots.")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"], help="Output format.")
    parser.add_argument("--names", nargs="+", default=None, help="Legend labels for conditions.")
    parser.add_argument("--width", type=float, default=12, help="Figure width in inches.")
    parser.add_argument("--height-per-row", type=float, default=0.33, help="Row height per word.")
    args = parser.parse_args(argv)

    plot_stacked(
        csv_files=args.csv_files,
        names=args.names,
        topk=args.topk,
        output_dir=args.outdir,
        save_format=args.format,
        width=args.width,
        height_per_row=args.height_per_row,
    )

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
