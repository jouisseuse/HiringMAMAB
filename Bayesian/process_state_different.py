"""
Efficiency analysis and plotting for the unequal-productivity MAMAB condition.

Two analyses are run on a folder of simulation logs:
  1. Optimal arm rate over rounds — fraction of agents selecting the best arm
     per round, pooled across all initial states. Outputs a trend plot and CSVs.
  2. Cumulative reward improvement — bootstrapped effect size (% gain) and
     Welch's t-test comparing social vs. asocial learning.

Usage:
    python process_state_different.py <exp_dir>

    exp_dir must contain subdirectories named state_1, state_2, …, each with
    communication/ and non-communication/ subfolders holding .log files.
"""

import os
import sys
import json

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
import seaborn as sns


# Index of the arm with the highest success probability (0-based).
# For the default 10-arm "different" setup the best arm is index 1 (p=0.95).
OPTIMAL_ARM = 1


# ── I/O ──────────────────────────────────────────────────────────────────────

def load_experiment_data(folder_path: str) -> list:
    """Load all .log JSON files in a folder and return a list of experiment dicts."""
    experiment_data = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(".log"):
            with open(os.path.join(folder_path, file_name), "r") as f:
                experiment_data.append(json.load(f))
    return experiment_data


# ── Optimal arm rate ──────────────────────────────────────────────────────────

def analyze_all_states(exp_dir: str, save_path: str, optimal_arm: int):
    """
    Pool data across all initial states and compute per-round optimal arm rate.

    Returns two lists of (round, mean, ci_lower, ci_upper) tuples for the
    social and asocial conditions respectively, and saves a trend plot.
    """
    all_rounds_comm: dict[int, list[float]] = {}
    all_rounds_ncom: dict[int, list[float]] = {}

    states = sorted(
        s for s in os.listdir(exp_dir)
        if os.path.isdir(os.path.join(exp_dir, s)) and s != "results"
    )

    for state_name in states:
        state_path = os.path.join(exp_dir, state_name)
        comm_data = load_experiment_data(os.path.join(state_path, "communication"))
        ncom_data = load_experiment_data(os.path.join(state_path, "non-communication"))

        for bucket, data in ((all_rounds_comm, comm_data), (all_rounds_ncom, ncom_data)):
            for exp in data:
                for round_num, rd in enumerate(exp["results"]):
                    choices = rd.get("choices", [])
                    if choices:
                        rate = sum(1 for a in choices if a == optimal_arm) / len(choices)
                        bucket.setdefault(round_num, []).append(rate)

    def _summarize(rounds_dict):
        rows = []
        for round_num, rates in sorted(rounds_dict.items()):
            mean = np.mean(rates)
            ci_low, ci_up = (
                st.t.interval(0.95, len(rates) - 1, loc=mean, scale=st.sem(rates))
                if len(rates) > 1 else (mean, mean)
            )
            rows.append([round_num + 1, mean, ci_low, ci_up])
        return rows

    final_comm = _summarize(all_rounds_comm)
    final_ncom = _summarize(all_rounds_ncom)

    pd.DataFrame(final_comm, columns=["Round", "Mean Rate", "Lower Bound", "Upper Bound"]).to_csv(
        os.path.join(save_path, "optimal_arm_rate_comm.csv"), index=False
    )
    pd.DataFrame(final_ncom, columns=["Round", "Mean Rate", "Lower Bound", "Upper Bound"]).to_csv(
        os.path.join(save_path, "optimal_arm_rate_no_comm.csv"), index=False
    )

    plot_optimal_arm_rate(final_comm, final_ncom, save_path)
    return final_comm, final_ncom


def plot_optimal_arm_rate(optimal_rates_comm: list, optimal_rates_ncom: list, save_path: str):
    """Plot mean optimal arm rate over rounds with 95% CI shading."""
    sns.set_theme(style="white")
    color_comm = "#1a80bb"
    color_ncom = "#ea801c"

    rounds      = [x[0] for x in optimal_rates_comm]
    mean_comm   = [x[1] for x in optimal_rates_comm]
    lower_comm  = [x[2] for x in optimal_rates_comm]
    upper_comm  = [x[3] for x in optimal_rates_comm]
    mean_ncom   = [x[1] for x in optimal_rates_ncom]
    lower_ncom  = [x[2] for x in optimal_rates_ncom]
    upper_ncom  = [x[3] for x in optimal_rates_ncom]

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.plot(rounds, mean_comm, label="Social learning",  linestyle="-", linewidth=1.8, color=color_comm)
    ax.fill_between(rounds, lower_comm, upper_comm, color=color_comm, alpha=0.2)

    ax.plot(rounds, mean_ncom, label="Asocial learning", linestyle="-", linewidth=1.8, color=color_ncom)
    ax.fill_between(rounds, lower_ncom, upper_ncom, color=color_ncom, alpha=0.2)

    ax.set_ylim(0, 1)
    ax.set_xlabel("Rounds",          fontsize=12, fontname="Arial")
    ax.set_ylabel("Optimal Arm Rate", fontsize=12, fontname="Arial")
    ax.tick_params(axis="both", direction="out", length=3, width=0.8, labelsize=10)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    legend = ax.legend(frameon=False, fontsize=10, loc="lower center", bbox_to_anchor=(0.55, 0), ncol=2)
    for text in legend.get_texts():
        text.set_color("black")

    fig.savefig(
        os.path.join(save_path, "optimal_arm_rate.pdf"),
        dpi=300, bbox_inches="tight", transparent=True, format="pdf",
    )
    plt.close(fig)


# ── Reward improvement ────────────────────────────────────────────────────────

def calculate_reward_improvement(exp_dir: str) -> dict:
    """
    Compute the relative cumulative reward gain of social over asocial learning.

    Effect size b = (mean_social - mean_asocial) / mean_asocial × 100%.
    95% CI via bootstrap (10 000 resamples); p-value via Welch's t-test.
    """
    rewards_comm = []
    rewards_ncom = []

    states = sorted(
        s for s in os.listdir(exp_dir)
        if os.path.isdir(os.path.join(exp_dir, s)) and s != "results"
    )

    for state_name in states:
        state_path = os.path.join(exp_dir, state_name)
        for bucket, folder in (
            (rewards_comm, "communication"),
            (rewards_ncom, "non-communication"),
        ):
            for exp in load_experiment_data(os.path.join(state_path, folder)):
                bucket.append(np.sum([rd.get("rewards", []) for rd in exp["results"]]))

    mean_comm = np.mean(rewards_comm)
    mean_ncom = np.mean(rewards_ncom)
    b = (mean_comm - mean_ncom) / mean_ncom * 100.0

    def _bootstrap_b(data_comm, data_ncom):
        s_c = np.random.choice(data_comm, size=len(data_comm), replace=True)
        s_n = np.random.choice(data_ncom, size=len(data_ncom), replace=True)
        return (np.mean(s_c) - np.mean(s_n)) / np.mean(s_n) * 100.0

    bootstrap = [_bootstrap_b(rewards_comm, rewards_ncom) for _ in range(10_000)]
    ci_low, ci_up = np.percentile(bootstrap, [2.5, 97.5])

    _, p_value = st.ttest_ind(rewards_comm, rewards_ncom, equal_var=False)

    print(f"b = {b:.2f}%, 95% CI [{ci_low:.2f}%, {ci_up:.2f}%], p = {p_value:.4f}")
    return {"b": b, "ci_lower": ci_low, "ci_upper": ci_up, "p_value": p_value}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_state_different.py <exp_dir>")
        sys.exit(1)

    exp_dir = sys.argv[1]
    save_path = os.path.join(exp_dir, "results")
    os.makedirs(save_path, exist_ok=True)

    analyze_all_states(exp_dir, save_path, OPTIMAL_ARM)
    calculate_reward_improvement(exp_dir)
    print(f"Results saved to {save_path}")
