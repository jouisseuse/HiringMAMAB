"""
Efficiency plot for the LLM unequal-productivity condition.

Produces a horizontal four-group bar chart comparing cumulative reward between
social and asocial learning across Overall, Top 20%, Middle 20%, and Bottom 20%
of simulation runs. The percentage difference (social − asocial) / asocial is
annotated on the right side of each bar pair.

Usage:
    python new_llm_different.py <comm_folder> <no_comm_folder> [--outdir <dir>] [--top 0.2] [--mid 0.2]
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


# ── I/O ──────────────────────────────────────────────────────────────────────

def load_experiment_data(folder_path: str) -> List[Dict[str, Any]]:
    """Load all .json experiment log files from a folder."""
    data = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(".json"):
            with open(os.path.join(folder_path, file_name), "r") as f:
                data.append(json.load(f))
    return data


# ── Metrics ───────────────────────────────────────────────────────────────────

def final_cumulative_reward(exp: Dict[str, Any]) -> float:
    """Sum all rewards across rounds for a single experiment."""
    total = 0.0
    for rd in exp.get("results", []) or []:
        rew = rd.get("rewards")
        try:
            total += float(np.sum(rew)) if rew is not None else 0.0
        except Exception:
            pass
    return total


# ── Stratified buckets ────────────────────────────────────────────────────────

def _bucket_indices(
    vals: np.ndarray, p_top: float = 0.2, p_mid: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return index arrays for the top, middle (40–60%), and bottom strata.

    Parameters
    ----------
    vals   : array of scalar values to stratify
    p_top  : fraction defining top and bottom buckets
    p_mid  : fraction defining the middle bucket (centered at 50th percentile)
    """
    n = len(vals)
    if n == 0:
        empty = np.array([], int)
        return empty, empty, empty

    order = np.argsort(vals)[::-1]
    k = max(1, int(np.floor(n * p_top)))
    top_idx = order[:k]

    lo = int(np.floor(n * 0.40))
    hi = int(np.floor(n * 0.60))
    if hi <= lo:
        hi = min(n, lo + max(1, int(np.floor(n * p_mid))))
    mid_idx = order[lo:hi]

    bot_idx = order[-k:]
    return np.sort(top_idx), np.sort(mid_idx), np.sort(bot_idx)


# ── Plot style ────────────────────────────────────────────────────────────────

def _apply_style():
    plt.rcParams.update({
        "font.family": "Arial",
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 0.9,
    })


# ── Main plot ─────────────────────────────────────────────────────────────────

def plot_four_group_bars(
    comm_vals: np.ndarray, non_vals: np.ndarray, outdir: str,
    p_top: float = 0.2, p_mid: float = 0.2,
) -> None:
    """
    Horizontal paired bar chart: social vs. asocial cumulative reward,
    stratified into Overall / Top 20% / Middle 20% / Bottom 20%.
    """
    _apply_style()
    os.makedirs(outdir, exist_ok=True)

    color_comm = "#1a80bb"
    color_non  = "#ea801c"

    overall_comm = float(np.mean(comm_vals)) if comm_vals.size else 0.0
    overall_non  = float(np.mean(non_vals))  if non_vals.size  else 0.0

    c_top, c_mid, c_bot = _bucket_indices(comm_vals, p_top, p_mid)
    n_top, n_mid, n_bot = _bucket_indices(non_vals,  p_top, p_mid)

    def _mean(arr, idx):
        return float(np.mean(arr[idx])) if idx.size else 0.0

    groups = [
        ("Overall",    overall_comm,              overall_non),
        ("Top 20%",    _mean(comm_vals, c_top),   _mean(non_vals, n_top)),
        ("Middle 20%", _mean(comm_vals, c_mid),   _mean(non_vals, n_mid)),
        ("Bottom 20%", _mean(comm_vals, c_bot),   _mean(non_vals, n_bot)),
    ]

    # Two-line tick labels for stratified groups
    labels = []
    for name, _, _ in groups:
        if "Top"    in name: labels.append("Top\n20%")
        elif "Mid"  in name: labels.append("Middle\n20%")
        elif "Bot"  in name: labels.append("Bottom\n20%")
        else:                labels.append("Overall")

    comm_m = [g[1] for g in groups]
    non_m  = [g[2] for g in groups]

    # Vertical positions: Overall sits above the stratified group with extra gap
    step = 1.0
    gap  = 0.8
    y = np.array([(len(labels) - 1 - i) * step for i in range(len(labels))], dtype=float)
    y[0] += gap

    h = 0.34
    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    ax.axvline(0, color="black", linewidth=1.0)
    ax.barh(y - h / 2, non_m,  height=h, color=color_non,  alpha=0.85)
    ax.barh(y + h / 2, comm_m, height=h, color=color_comm, alpha=0.85)

    # Percentage difference annotation on the right
    xmax   = max(max(comm_m, default=0), max(non_m, default=0))
    offset = 0.02 * (xmax + 1e-9)
    for yi, c, n in zip(y, comm_m, non_m):
        pct = 0.0 if (n is None or n == 0) else (c - n) / n * 100.0
        ax.text(
            max(c, n) + offset, yi, f"{pct:+.1f}%",
            va="center", ha="left", color=color_comm,
            fontsize=12, fontweight="bold", fontstyle="italic",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Cumulative reward")
    ax.set_ylabel("")
    ax.set_xlim(0, xmax * 1.15 + 1e-9)

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(axis="x", which="major", direction="out", length=8, width=1.1)
    ax.tick_params(axis="y", which="major", direction="out", length=8, width=1.1)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="x", which="minor", direction="out", length=4, width=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_linewidth(1.1)

    fig.savefig(os.path.join(outdir, "four_group_bars.pdf"), dpi=450, bbox_inches="tight", transparent=True, format="pdf")
    fig.savefig(os.path.join(outdir, "four_group_bars.png"), dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: List[str]) -> Tuple[str, str, str, float, float]:
    if len(argv) < 3:
        raise SystemExit(
            "Usage: new_llm_different.py <comm_folder> <no_comm_folder> [--outdir <dir>] [--top 0.2] [--mid 0.2]"
        )
    comm_folder    = argv[1]
    no_comm_folder = argv[2]
    outdir = comm_folder
    p_top = p_mid = 0.2
    i = 3
    while i < len(argv):
        if argv[i] == "--outdir" and i + 1 < len(argv):
            outdir = argv[i + 1]; i += 2
        elif argv[i] == "--top" and i + 1 < len(argv):
            p_top = float(argv[i + 1]); i += 2
        elif argv[i] == "--mid" and i + 1 < len(argv):
            p_mid = float(argv[i + 1]); i += 2
        else:
            i += 1
    return comm_folder, no_comm_folder, outdir, p_top, p_mid


if __name__ == "__main__":
    comm_folder, no_comm_folder, outdir, p_top, p_mid = _parse_args(sys.argv)

    comm_data = load_experiment_data(comm_folder)
    non_data  = load_experiment_data(no_comm_folder)

    comm_vals = np.array([final_cumulative_reward(e) for e in comm_data], dtype=float)
    non_vals  = np.array([final_cumulative_reward(e) for e in non_data],  dtype=float)

    plot_four_group_bars(comm_vals, non_vals, outdir, p_top=p_top, p_mid=p_mid)
    print(f"Saved to {outdir}")
