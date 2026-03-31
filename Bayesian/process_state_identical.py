"""
Entropy analysis and plotting for MAMAB simulation results.

Produces two publication-quality figures (Science style):
  1. entropy_trend.pdf    — mean entropy over rounds with 95% CI shading
  2. entropy_last_round.pdf — jittered scatter + mean/CI for the final round

Also runs an OLS regression comparing Social vs. Asocial entropy and appends
results to results.txt in the base folder.

Usage:
    python new_process_state.py <base_folder> [<num_states>] [<num_arms>]
"""
from __future__ import annotations

import csv
import math
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import pandas as pd
import scipy.stats as st
import statsmodels.formula.api as smf


# ── Labels and colors ────────────────────────────────────────────────────────

COMM = "Social"
NCOM = "Asocial"
COLOR_COMM = "#1a80bb"
COLOR_NCOM = "#ea801c"


# ── Entropy helpers ───────────────────────────────────────────────────────────

def calculate_entropy(frequencies) -> float:
    """Shannon entropy (bits) from a sequence of non-negative counts."""
    total = sum(frequencies)
    if total == 0:
        return 0.0
    return -sum((f / total) * math.log2(f / total) for f in frequencies if f > 0)


def calculate_accumulative_entropy(
    data: List[Dict[str, Any]],
    num_arms: int,
    output_csv: str = None,
) -> Tuple[List[float], List[np.ndarray]]:
    """
    Compute cumulative arm-selection entropy over rounds, pooled across experiments.

    Parameters
    ----------
    data : list of experiment log dicts, each with a "results" list of rounds
    num_arms : number of arms
    output_csv : optional path to write per-round counts and entropy

    Returns
    -------
    arm_entropies : entropy value per round
    arm_choice_records : cumulative arm count array per round
    """
    num_rounds = len(data[0]["results"])
    num_agents = len(data[0]["results"][0]["choices"])
    arm_choice_counts = np.zeros(num_arms)
    arm_entropies: List[float] = []
    arm_choice_records: List[np.ndarray] = []

    csv_file = open(output_csv, "w", newline="") if output_csv else None
    csv_writer = csv.writer(csv_file) if csv_file else None
    if csv_writer:
        csv_writer.writerow(["Round"] + [f"Arm_{i}_Count" for i in range(num_arms)] + ["Entropy"])

    for round_num in range(num_rounds):
        for exp in data:
            for agent_id in range(num_agents):
                choice = exp["results"][round_num]["choices"][agent_id]
                arm_choice_counts[choice] += 1

        entropy = calculate_entropy(arm_choice_counts)
        arm_entropies.append(entropy)
        arm_choice_records.append(arm_choice_counts.copy())

        if csv_writer:
            csv_writer.writerow([round_num + 1] + list(arm_choice_counts) + [entropy])

    if csv_file:
        csv_file.close()

    return arm_entropies, arm_choice_records


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _apply_science_style(font: str = "Arial") -> None:
    plt.rcParams.update({
        "font.family": font,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 0.9,
    })


def _read_json_folder(folder: str) -> List[Dict[str, Any]]:
    files = sorted(f for f in os.listdir(folder) if f.endswith((".log", ".json")))
    data = []
    for fn in files:
        try:
            with open(os.path.join(folder, fn), "r", encoding="utf-8") as f:
                data.append(json.load(f))
        except Exception as e:
            print(f"[WARN] skipping {fn}: {e}")
    return data


# ── Main analysis and plotting ────────────────────────────────────────────────

def process_and_plot_box_results(base_folder: str, num_states: int, num_arms: int) -> None:
    """
    Load all simulation logs, compute entropy, and save two plots + OLS stats.
    """
    _apply_science_style()
    largest_entropy = math.log2(num_arms)

    all_rounds_data: Dict[int, Dict[str, List[float]]] = {}
    last_round_records: List[Tuple[float, str]] = []

    for state_id in range(1, num_states + 1):
        state_folder = os.path.join(base_folder, f"state_{state_id}")
        comm_data = _read_json_folder(os.path.join(state_folder, "communication"))
        ncom_data = _read_json_folder(os.path.join(state_folder, "non-communication"))

        if not comm_data or not ncom_data:
            print(f"[WARN] empty data in {state_folder}, skipping")
            continue

        comm_entropies = [calculate_accumulative_entropy([exp], num_arms)[0] for exp in comm_data]
        ncom_entropies = [calculate_accumulative_entropy([exp], num_arms)[0] for exp in ncom_data]

        min_rounds = min(len(comm_entropies[0]), len(ncom_entropies[0]))

        for r in range(min_rounds):
            if r not in all_rounds_data:
                all_rounds_data[r] = {COMM: [], NCOM: []}
            all_rounds_data[r][COMM].extend(e[r] for e in comm_entropies)
            all_rounds_data[r][NCOM].extend(e[r] for e in ncom_entropies)

        last_round_records.extend((e[min_rounds - 1], COMM) for e in comm_entropies)
        last_round_records.extend((e[min_rounds - 1], NCOM) for e in ncom_entropies)

    # Build per-round summary dataframe
    trend_rows: List[Tuple[int, float, float, float, str]] = []
    for r, groups in sorted(all_rounds_data.items()):
        for cond, arr in groups.items():
            if not arr:
                continue
            mean = float(np.mean(arr))
            ci_low, ci_up = (
                st.t.interval(0.95, len(arr) - 1, loc=mean, scale=st.sem(arr))
                if len(arr) > 1 else (mean, mean)
            )
            trend_rows.append((r + 1, mean, float(ci_low), float(ci_up), cond))

    trend_df = pd.DataFrame(trend_rows, columns=["Round", "Mean", "Low", "Up", "Cond"])
    last_df = pd.DataFrame(last_round_records, columns=["Entropy", "Cond"])

    # ── Trend plot ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    for cond, color in ((COMM, COLOR_COMM), (NCOM, COLOR_NCOM)):
        df = trend_df[trend_df["Cond"] == cond]
        x, y = df["Round"].to_numpy(), df["Mean"].to_numpy()
        lo, up = df["Low"].to_numpy(), df["Up"].to_numpy()
        ax.plot(x, y, color=color, linewidth=1.8)
        ax.fill_between(x, lo, up, color=color, alpha=0.20)

    ax.set_ylim(0.5, 3.32)
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

    out_trend = os.path.join(base_folder, "entropy_trend.pdf")
    fig.savefig(out_trend, dpi=450, bbox_inches="tight", transparent=True, format="pdf")
    plt.close(fig)
    print(f"Saved: {out_trend}")

    # ── Final-round scatter plot ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    jitter = 0.08
    for xi, cond, color in ((0, COMM, COLOR_COMM), (1, NCOM, COLOR_NCOM)):
        vals = last_df[last_df["Cond"] == cond]["Entropy"].to_numpy()
        xs = xi + np.random.normal(0, jitter, size=len(vals))
        ax.scatter(xs, vals, s=25, alpha=0.5, edgecolors="none", color=color)

        if len(vals) > 0:
            mean = float(np.mean(vals))
            ci_low, ci_up = (
                st.t.interval(0.95, len(vals) - 1, loc=mean, scale=st.sem(vals))
                if len(vals) > 1 else (mean, mean)
            )
            ax.errorbar(xi, mean, yerr=[[mean - ci_low], [ci_up - mean]],
                        fmt="o", color="black", linewidth=1.2, capsize=0)

    ax.set_ylim(0.5, 3.32)
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Social", "Asocial"])
    ax.set_ylabel("Entropy")
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.tick_params(axis="x", which="major", direction="out", length=7, width=1.0)
    ax.tick_params(axis="y", which="major", direction="out", length=7, width=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_linewidth(1.1)

    out_last = os.path.join(base_folder, "entropy_last_round.pdf")
    fig.savefig(out_last, dpi=450, bbox_inches="tight", transparent=True, format="pdf")
    plt.close(fig)
    print(f"Saved: {out_last}")

    # ── OLS regression: Social vs. Asocial entropy ────────────────────────────
    last_df["Cond"] = last_df["Cond"].str.strip()
    model = smf.ols("Entropy ~ C(Cond, Treatment(reference='Asocial'))", data=last_df).fit()
    b_key = "C(Cond, Treatment(reference='Asocial'))[T.Social]"

    if b_key in model.params:
        b_value = float(model.params[b_key])
        ci_low, ci_up = model.conf_int().loc[b_key]
        p_val = float(model.pvalues[b_key])
        mean_ncom = float(last_df[last_df["Cond"] == NCOM]["Entropy"].mean())
        mean_comm = float(last_df[last_df["Cond"] == COMM]["Entropy"].mean())
        reduction = mean_ncom - mean_comm
        pct = (reduction / mean_ncom * 100.0) if mean_ncom != 0 else 0.0

        results_path = os.path.join(base_folder, "results.txt")
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(f"b = {b_value:.4f}, 95% CI = [{ci_low:.4f}, {ci_up:.4f}], p = {p_val:.4f}\n")
            f.write(f"Entropy reduction: {reduction:.4f} ({pct:.2f}%)\n")
        print(f"b = {b_value:.4f}, 95% CI = [{ci_low:.4f}, {ci_up:.4f}], p = {p_val:.4f}")
        print(f"Entropy reduction: {reduction:.4f} ({pct:.2f}%)")
    else:
        print("[ERROR] OLS key not found — check condition labels.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python new_process_state.py <base_folder> [<num_states>] [<num_arms>]")
        sys.exit(1)
    base_folder = sys.argv[1]
    num_states = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    num_arms = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    process_and_plot_box_results(base_folder, num_states, num_arms)
