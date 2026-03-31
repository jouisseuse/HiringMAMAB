# filepath: visualize_strategy_distribution.py

"""
Publication-style plotting of participant strategy proportions.

Key features
- PDF output (vector), Arial font (or system sans-serif fallback), no title.
- Larger, consistent typography; minimal ink; tight layout.
- Jittered scatter per strategy (one dot = one participant).
- Optional vertical reference line (default 0.6).
- Strategies ordered by median proportion (descending) for interpretability.

CLI
    python visualize_strategy_distribution.py <base_folder> <strategy_type>

Expected directory layout
    base_folder/
      ├─ <participant_id>/
      │   └─ analysis/
      │       └─ *_labeled.csv

CSV expectations
    Must contain column named exactly as <strategy_type> (e.g., "Pattern" or "Judgment").

"""

from __future__ import annotations

import os
import sys
import math
import random
from typing import Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Configuration helpers
# -----------------------------

def configure_matplotlib_publication(base_fontsize: int = 14) -> None:
    """Set Matplotlib rcParams for a clean, publication-style look.

    Why: enforce consistent typography and embed TrueType fonts in PDF.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Nimbus Sans"],
        "pdf.fonttype": 42,            # Embed TrueType fonts
        "ps.fonttype": 42,
        "axes.labelsize": base_fontsize + 2,
        "xtick.labelsize": base_fontsize,
        "ytick.labelsize": base_fontsize,
        "legend.fontsize": base_fontsize,
        "axes.titlesize": base_fontsize + 4,  # kept for completeness; we won't use a title
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "figure.dpi": 200,
    })


# -----------------------------
# Data loading
# -----------------------------

def load_strategy_data(folder: str, strategy_type: str) -> pd.DataFrame:
    """Load per-participant normalized strategy proportions.

    Returns
        DataFrame with columns: [playerID, <strategy1>, <strategy2>, ...]
    """
    strategy_counts: Dict[str, Dict[str, float]] = {}

    for subdir in os.listdir(folder):
        analysis_dir = os.path.join(folder, subdir, "analysis")
        if not os.path.isdir(analysis_dir):
            continue

        for file in os.listdir(analysis_dir):
            if not file.endswith("_labeled.csv"):
                continue
            csv_path = os.path.join(analysis_dir, file)
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                print(f"[WARN] Skipping unreadable CSV: {csv_path} ({exc})")
                continue

            if strategy_type not in df.columns:
                print(f"[WARN] '{strategy_type}' column not found in {csv_path}; skipping.")
                continue

            pid = f"{subdir}_{file.replace('_labeled.csv', '')}"
            counts = (
                df[strategy_type]
                .dropna()
                .astype(str)
                .value_counts(normalize=False)
                .to_dict()
            )
            total = sum(counts.values())
            if total == 0:
                continue

            # Normalize, exclude explicit N/A-like labels (case-insensitive exact)
            norm_counts = {k: v / total for k, v in counts.items() if k.strip().upper() != "N/A"}
            if not norm_counts:
                continue

            strategy_counts[pid] = norm_counts

    if not strategy_counts:
        # Return empty structure with expected shape
        return pd.DataFrame({"playerID": []})

    df_prop = pd.DataFrame(strategy_counts).T.fillna(0.0)
    df_prop.index.name = "playerID"
    df_prop.reset_index(inplace=True)
    return df_prop


# -----------------------------
# Plotting
# -----------------------------

def _compute_long_and_order(df_prop: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """Melt to long format and order strategies by median proportion (desc)."""
    if df_prop.empty or df_prop.shape[1] <= 1:
        return pd.DataFrame(columns=["playerID", "Strategy", "Proportion"]), []

    long_df = df_prop.melt(id_vars="playerID", var_name="Strategy", value_name="Proportion")
    long_df = long_df[long_df["Strategy"] != "N/A"]

    # Strategy order by median proportion (desc), then alphabetical
    medians = long_df.groupby("Strategy")["Proportion"].median().sort_values(ascending=False)
    order = medians.index.tolist()
    return long_df, order


def plot_strategy_distribution(
    df_prop: pd.DataFrame,
    strategy_type_label: str,
    save_path: str | None = None,
    vline_at: float | None = 0.6,
    jitter: float = 0.12,
    seed: int = 42,
) -> None:
    """Render a publication-style jittered scatter and save/show.

    Why: stripplot-like visual without seaborn for full control and portability.
    """
    configure_matplotlib_publication(base_fontsize=16)  # Bigger base font, no title.

    long_df, strategy_order = _compute_long_and_order(df_prop)
    if long_df.empty:
        print("[INFO] No data to plot.")
        return

    y_positions = {s: i for i, s in enumerate(strategy_order)}

    if strategy_type_label == "Judgment":
        v = 5
    elif strategy_type_label == "Pattern":
        v = 6


    fig = plt.figure(figsize=(10, v))
    ax = plt.gca()

    rng = random.Random(seed)

    # Draw points with vertical jitter around the strategy index
    xs = []
    ys = []
    for _, row in long_df.iterrows():
        s = row["Strategy"]
        p = float(row["Proportion"]) if pd.notna(row["Proportion"]) else 0.0
        base_y = y_positions[s]
        # Why: symmetric jitter to reduce overplotting while preserving order
        y = base_y + rng.uniform(-jitter, jitter)
        xs.append(p)
        ys.append(y)

    ax.scatter(xs, ys, s=16, alpha=0.45, linewidths=0)

    # if vline_at is not None:
    #     ax.axvline(vline_at, linestyle="--", linewidth=1.0, alpha=0.5)

    # Y ticks at strategy centers
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(strategy_order)

    ax.set_xlabel("Proportion of Use")
    # ax.set_ylabel(strategy_type_label)
    ax.set_xlim(0, 1)

    # Clean margins and layout
    plt.tight_layout()

    if save_path:
        # Enforce PDF by extension if user asked for PDF-like output
        root, ext = os.path.splitext(save_path)
        if ext.lower() not in {".pdf", ".png", ".svg"}:
            save_path = root + ".pdf"
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


# -----------------------------
# Runner
# -----------------------------

def run_analysis(base_folder: str, strategy_type: str) -> None:
    df_prop = load_strategy_data(base_folder, strategy_type)
    # Save as PDF with Arial, no title, larger fonts
    out_path = os.path.join(base_folder, f"{strategy_type}_strategy_distribution.pdf")
    plot_strategy_distribution(df_prop, strategy_type_label=strategy_type, save_path=out_path)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize_strategy_distribution.py <base_folder> <strategy_type>")
        sys.exit(1)
    base_input_dir = sys.argv[1]
    strategy_type_arg = sys.argv[2]  # e.g., 'Pattern' or 'Judgment'
    run_analysis(base_input_dir, strategy_type_arg)
