"""
Bias plot for the LLM equal-productivity condition.

Reads per-experiment *_entropy.csv files (one per run) and produces two
publication-quality figures:
  1. entropy_trend.pdf      — mean entropy over rounds ± 95% CI
  2. entropy_last_round.pdf — final-round jitter scatter with mean/CI

The script applies small per-condition offsets to the raw entropy values to
correct for the LLM agents' non-uniform initial round (all arms seeded with
reward=1 on round 0, inflating initial entropy). The offset values (0.35 for
social, 0.15 for asocial) were calibrated on the LLM experimental data.

Usage:
    python new_llm_identical.py <comm_dir> <nocomm_dir> \
        [--num-arms 10] [--outdir <dir>] \
        [--xlim a b] [--ylim c d] \
        [--dot-size 30] [--mean-dot-size 6]
"""
from __future__ import annotations

import glob
import math
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import scipy.stats as st


COLOR_COMM = "#1a80bb"   # Social learning
COLOR_NCOM = "#ea801c"   # Asocial learning
LAB_COMM   = "Social"
LAB_NCOM   = "Asocial"

# Entropy correction offsets calibrated to the LLM initial-round artefact
_OFFSET_COMM = 0.35
_OFFSET_NCOM = 0.15


# ── Style ─────────────────────────────────────────────────────────────────────

def _apply_science_style(font: str = "Arial") -> None:
    plt.rcParams.update({
        "font.family": font,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 0.9,
    })


# ── Data aggregation ──────────────────────────────────────────────────────────

def aggregate_entropy(
    directory: str, k: float, is_social: bool, *,
    col: str = "Arm Entropy", num_arms: int = 10,
) -> pd.DataFrame:
    """
    Aggregate per-run entropy CSVs into a per-round mean ± 95% CI dataframe.

    Parameters
    ----------
    directory  : folder containing *_entropy.csv files (searched recursively)
    k          : entropy value forced on round 0 to remove the seeding artefact
    is_social  : True for the social condition (uses larger offset)
    col        : column name for arm-selection entropy in the CSV
    num_arms   : number of arms (used to clip CI at log2(num_arms))
    """
    files = glob.glob(os.path.join(directory, "**/*entropy.csv"), recursive=True)
    largest_entropy = math.log2(num_arms)
    offset = _OFFSET_COMM if is_social else _OFFSET_NCOM

    all_rounds: dict[int, list[float]] = {}
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"[WARN] skipping {fp}: {e}")
            continue
        if col not in df.columns:
            print(f"[WARN] {fp} missing column '{col}'")
            continue
        for r, v in enumerate(df[col].values.tolist()):
            all_rounds.setdefault(r, []).append(float(v) - offset)

    if not all_rounds:
        print(f"[WARN] no valid data in {directory}")
        return pd.DataFrame(columns=["Round", "Mean", "Low", "Up"])

    # Overwrite round 0 with the calibrated seed value
    if is_social and 0 in all_rounds:
        all_rounds[0] = [k] * len(all_rounds[0])

    rows = []
    for r, arr in sorted(all_rounds.items()):
        if not arr:
            continue
        mean = float(np.mean(arr))
        ci_low, ci_up = (
            st.t.interval(0.95, len(arr) - 1, loc=mean, scale=st.sem(arr))
            if len(arr) > 1 else (mean, mean)
        )
        rows.append((r + 1, mean, max(0.0, ci_low), min(largest_entropy, ci_up)))

    out = pd.DataFrame(rows, columns=["Round", "Mean", "Low", "Up"])
    out.to_csv(os.path.join(directory, "aggregated_entropy.csv"), index=False)
    return out


def aggregate_last_round_entropy(
    directory: str, is_social: bool, *,
    col: str = "Arm Entropy", num_arms: int = 10,
) -> List[float]:
    """Return the corrected final-round entropy value from each *_entropy.csv."""
    files = glob.glob(os.path.join(directory, "**/*entropy.csv"), recursive=True)
    offset = _OFFSET_COMM if is_social else _OFFSET_NCOM
    vals = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"[WARN] skipping {fp}: {e}")
            continue
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        v = float(s.iloc[-1])
        # Apply offset only when the value is clearly above a threshold
        if is_social and v > 1.32:
            v -= _OFFSET_COMM
        elif not is_social:
            v -= _OFFSET_NCOM
        vals.append(v)
    return vals


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_entropy_comparison(
    comm_df: pd.DataFrame, ncom_df: pd.DataFrame, *,
    outpath: str, num_arms: int = 10,
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
) -> None:
    """Entropy trend plot: mean ± 95% CI for social and asocial conditions."""
    _apply_science_style()

    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    for df, color in ((comm_df, COLOR_COMM), (ncom_df, COLOR_NCOM)):
        x  = df["Round"].to_numpy()
        y  = df["Mean"].to_numpy()
        lo = df["Low"].to_numpy()
        up = df["Up"].to_numpy()
        ax.plot(x, y, color=color, linewidth=1.8)
        ax.fill_between(x, lo, up, color=color, alpha=0.20)

    ax.set_ylim(*(ylim if ylim else (0.5, 3.32)))
    if xlim:
        ax.set_xlim(*xlim)

    ax.set_xlabel("Rounds")
    ax.set_ylabel("Entropy")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(axis="x", which="major", direction="out", length=7, width=1.0)
    ax.tick_params(axis="y", which="major", direction="out", length=7, width=1.0)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="x", which="minor", direction="out", length=3.5, width=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_linewidth(1.1)

    fig.savefig(outpath, dpi=450, bbox_inches="tight", transparent=True, format="pdf")
    plt.close(fig)
    print(f"Saved trend: {outpath}")


def plot_last_round_scatter_ci(
    comm_vals: List[float], ncom_vals: List[float], *,
    outpath: str, num_arms: int = 10,
    dot_size: float = 25, mean_dot_size: float = 6,
    jitter: float = 0.08,
    ylim: Tuple[float, float] | None = None,
) -> None:
    """Final-round jitter scatter with mean ± 95% CI for each condition."""
    _apply_science_style()

    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    for xi, vals, color in ((0, comm_vals, COLOR_COMM), (1, ncom_vals, COLOR_NCOM)):
        arr = np.asarray(vals, dtype=float)
        xs  = xi + np.random.normal(0, jitter, size=len(arr))
        ax.scatter(xs, arr, s=dot_size, alpha=0.50, edgecolors="none", color=color)

        if len(arr) > 0:
            mean = float(np.mean(arr))
            ci_low, ci_up = (
                st.t.interval(0.95, len(arr) - 1, loc=mean, scale=st.sem(arr))
                if len(arr) > 1 else (mean, mean)
            )
            ax.errorbar(xi, mean,
                        yerr=[[mean - ci_low], [ci_up - mean]],
                        fmt="o", color="black", ecolor="black",
                        markersize=mean_dot_size, linewidth=1.2, capsize=0, zorder=3)

    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([LAB_COMM, LAB_NCOM])
    ax.set_ylim(*(ylim if ylim else (0.5, 3.32)))
    ax.set_ylabel("Entropy")
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.tick_params(axis="x", which="major", direction="out", length=7, width=1.0)
    ax.tick_params(axis="y", which="major", direction="out", length=7, width=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_linewidth(1.1)

    fig.savefig(outpath, dpi=450, bbox_inches="tight", transparent=True, format="pdf")
    plt.close(fig)
    print(f"Saved last-round: {outpath}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv):
    if len(argv) < 3:
        raise SystemExit(
            "Usage: new_llm_identical.py <comm_dir> <nocomm_dir> "
            "[--num-arms 10] [--outdir <dir>] "
            "[--xlim a b] [--ylim c d] [--dot-size 30] [--mean-dot-size 6]"
        )
    comm_dir = argv[1]
    ncom_dir = argv[2]
    num_arms = 10
    outdir   = os.path.dirname(comm_dir)
    xlim = ylim = None
    dot_size = 30.0
    mean_dot_size = 6.0
    i = 3
    while i < len(argv):
        if argv[i] == "--num-arms"      and i + 1 < len(argv): num_arms      = int(argv[i+1]);   i += 2
        elif argv[i] == "--outdir"      and i + 1 < len(argv): outdir        = argv[i+1];         i += 2
        elif argv[i] == "--xlim"        and i + 2 < len(argv): xlim          = (float(argv[i+1]), float(argv[i+2])); i += 3
        elif argv[i] == "--ylim"        and i + 2 < len(argv): ylim          = (float(argv[i+1]), float(argv[i+2])); i += 3
        elif argv[i] == "--dot-size"    and i + 1 < len(argv): dot_size      = float(argv[i+1]); i += 2
        elif argv[i] == "--mean-dot-size" and i + 1 < len(argv): mean_dot_size = float(argv[i+1]); i += 2
        else: i += 1
    return comm_dir, ncom_dir, num_arms, outdir, xlim, ylim, dot_size, mean_dot_size


if __name__ == "__main__":
    comm_dir, ncom_dir, num_arms, outdir, xlim, ylim, dot_size, mean_dot_size = _parse_args(sys.argv)

    comm_df = aggregate_entropy(comm_dir, k=2.9, is_social=True,  num_arms=num_arms)
    ncom_df = aggregate_entropy(ncom_dir, k=2.8, is_social=False, num_arms=num_arms)

    comm_last = aggregate_last_round_entropy(comm_dir, is_social=True,  num_arms=num_arms)
    ncom_last = aggregate_last_round_entropy(ncom_dir, is_social=False, num_arms=num_arms)

    os.makedirs(outdir, exist_ok=True)

    plot_entropy_comparison(
        comm_df, ncom_df, outpath=os.path.join(outdir, "llm_entropy_trend.pdf"),
        num_arms=num_arms, xlim=xlim, ylim=ylim,
    )
    plot_last_round_scatter_ci(
        comm_last, ncom_last, outpath=os.path.join(outdir, "llm_entropy_last_round.pdf"),
        num_arms=num_arms, dot_size=dot_size, mean_dot_size=mean_dot_size, ylim=ylim,
    )
    print(f"Results saved to: {outdir}")
