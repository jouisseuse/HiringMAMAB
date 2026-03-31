# filepath: visualize_share_distribution.py

"""
Publication-style distribution plot for share columns in a CSV.

Input CSV columns (required):
    participant_id, own_share, group_share, prev_share, others_share, n, experiment

Output:
    - Jittered scatter distribution of the four shares on a common x-axis [0,1].
    - Arial (sans-serif fallback), no title, large fonts.
    - Saves vector PDF by default.

CLI usage:
    python visualize_share_distribution.py <input_csv> [--experiment EXP] [--out OUTPATH]

Examples:
    python visualize_share_distribution.py data.csv
    python visualize_share_distribution.py data.csv --experiment A --out dist_A.pdf
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Iterable, Tuple

import pandas as pd
import matplotlib.pyplot as plt


SHARE_COLUMNS = ["own_share", "group_share", "prev_share", "others_share"]

def configure_matplotlib_publication(base_fontsize: int = 16) -> None:
    """Set Matplotlib rcParams for a clean, publication-style look.

    Why: enforce consistent typography and embed TrueType fonts in PDF.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Nimbus Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelsize": base_fontsize + 2,
        "xtick.labelsize": base_fontsize,
        "ytick.labelsize": base_fontsize,
        "legend.fontsize": base_fontsize,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "figure.dpi": 200,
    })


def load_share_csv(path: str, experiment: str | None = None) -> pd.DataFrame:
    """Load CSV and return a long-format DataFrame ready for plotting.

    Returns columns: [participant_id, experiment, Metric, Share]
    """
    df = pd.read_csv(path)

    missing = [c for c in ["participant_id", "experiment", *SHARE_COLUMNS] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if experiment is not None:
        df = df[df["experiment"].astype(str) == str(experiment)].copy()
        if df.empty:
            raise ValueError(f"No rows found for experiment = {experiment}")

    # Clip to [0,1] for safety; users may comment out if not desired.
    df[SHARE_COLUMNS] = df[SHARE_COLUMNS].clip(lower=0.0, upper=1.0)

    long_df = df.melt(
        id_vars=["participant_id", "experiment"],
        value_vars=SHARE_COLUMNS,
        var_name="Metric",
        value_name="Share",
    )

    # Human-friendly order and labels
    cat_type = pd.CategoricalDtype(categories=SHARE_COLUMNS, ordered=True)
    long_df["Metric"] = long_df["Metric"].astype(cat_type)

    return long_df


def _compute_order(long_df: pd.DataFrame) -> Tuple[list[str], dict[str, int]]:
    """Use fixed order; return positions for y-axis placement."""
    order = SHARE_COLUMNS.copy()
    y_positions = {s: i for i, s in enumerate(order)}
    return order, y_positions


def plot_share_distribution(
    long_df: pd.DataFrame,
    save_path: str | None = None,
    jitter: float = 0.12,
    seed: int = 42,
    figsize: Tuple[float, float] = (10.0, 5.5),
) -> None:
    """Render a publication-style jittered scatter and save/show.

    One dot = one (participant, metric) observation.
    """
    configure_matplotlib_publication(base_fontsize=16)

    if long_df.empty:
        print("[INFO] No data to plot.")
        return

    order, y_positions = _compute_order(long_df)

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    rng = random.Random(seed)

    xs: list[float] = []
    ys: list[float] = []

    # Iterate rows once to build jittered coordinates
    for _, r in long_df.iterrows():
        metric = r["Metric"]
        share = float(r["Share"]) if pd.notna(r["Share"]) else float("nan")
        if not (0.0 <= share <= 1.0):
            continue
        base_y = y_positions[str(metric)]
        y = base_y + rng.uniform(-jitter, jitter)  # reduces overplotting
        xs.append(share)
        ys.append(y)

    ax.scatter(xs, ys, s=16, alpha=0.45, linewidths=0)

    # Y ticks at metric centers
    ax.set_yticks(list(y_positions.values()))
    # Pretty names
    pretty = {
        "own_share": "Own",
        "group_share": "Group",
        "prev_share": "Previous",
        "others_share": "Others",
    }
    ax.set_yticklabels([pretty.get(s, s) for s in order])

    # ax.set_xlabel("Share")
    ax.set_xlim(0, 1)

    # Optional: thin grid for reading values (kept subtle)
    ax.xaxis.grid(True, linestyle=":", linewidth=0.8, alpha=0.5)

    plt.tight_layout()

    if save_path:
        root, ext = os.path.splitext(save_path)
        if ext.lower() not in {".pdf", ".png", ".svg"}:
            save_path = root + ".pdf"
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot distribution for share columns.")
    parser.add_argument("input_csv", type=str, help="Path to CSV with share columns")
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="If provided, filter to a single experiment",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path (pdf/png/svg). Defaults to <input_basename>_share_distribution.pdf",
    )
    args = parser.parse_args()

    long_df = load_share_csv(args.input_csv, experiment=args.experiment)

    out_path = args.out
    if out_path is None:
        base, _ = os.path.splitext(os.path.basename(args.input_csv))
        suffix = f"_{args.experiment}" if args.experiment is not None else ""
        out_path = f"{base}{suffix}_share_distribution.pdf"

    plot_share_distribution(long_df, save_path=out_path)


if __name__ == "__main__":
    main()
