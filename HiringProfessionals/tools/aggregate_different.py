from __future__ import annotations
import glob
import scipy.stats as st
import pandas as pd
import seaborn as sns

import os
import sys
import json
from typing import Sequence, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


marker_size = 2

def load_experiment_data(folder_path):
    """Load all playerRound.log files from folder and return as a list of parsed dicts."""
    experiment_data = []
    log_files = glob.glob(os.path.join(folder_path, "**/playerRound.log"), recursive=True)

    for file_path in log_files:
        with open(file_path, "r") as f:
            data = json.load(f)
            experiment_data.append(data)

    return experiment_data


def calculate_cumulative_reward_over_rounds(data: Sequence[Dict[str, Any]]) -> list[float]:
    """
    Compute per-experiment cumulative reward, then average across experiments per round.
    Steps: build per-experiment cumulative sum, then take mean at each round index.
    """
    if not data:
        return []

    # Use minimum round count across experiments for comparable indexing
    round_counts = [len(exp.get("results", [])) for exp in data if isinstance(exp.get("results"), list)]
    if not round_counts:
        raise ValueError("Data structure error: each experiment must contain a 'results' list.")
    num_rounds = int(min(round_counts))

    per_exp_cum: list[list[float]] = []
    for exp in data:
        results = exp.get("results", [])
        running = 0.0
        cum: list[float] = []
        for r in range(1, num_rounds):
            rd = results[r] if r < len(results) else {}
            rewards = rd.get("rewards")
            try:
                inc = float(np.sum(rewards)) if rewards is not None else 0.0
            except Exception:
                inc = 0.0
            running += inc
            cum.append(running)
        per_exp_cum.append(cum)

    arr = np.asarray(per_exp_cum, dtype=float)  # shape: (n_experiments, num_rounds)
    mean_cum = arr.mean(axis=0)
    return mean_cum.tolist()


def plot_and_save_cumulative_reward(
    cum_rewards_comm: Sequence[float],
    cum_rewards_no_comm: Sequence[float],
    save_path: str,
    *,
    max_per_round: float = 10.0,
    figsize: tuple[float, float] = (3, 4),
    line_width: float = 1.8,
    font_family: str = "Arial",
) -> None:
    """Plot cumulative reward curves (publication style) and save to PDF."""
    plt.rcParams.update({
        "font.family": font_family,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 0.8,
    })

    # Color scheme
    color_comm = "#1a80bb"
    color_nocomm = "#ea801c"

    os.makedirs(save_path, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    x_comm = np.arange(1, len(cum_rewards_comm) + 1)
    x_nocomm = np.arange(1, len(cum_rewards_no_comm) + 1)

    # Main curves (no markers)
    ax.plot(x_comm, cum_rewards_comm, color=color_comm, linewidth=line_width)
    ax.plot(x_nocomm, cum_rewards_no_comm, color=color_nocomm, linewidth=line_width)

    # Shaded area fill
    ax.fill_between(x_comm, cum_rewards_comm, 0, alpha=0.25, color=color_comm)
    ax.fill_between(x_nocomm, cum_rewards_no_comm, 0, alpha=0.25, color=color_nocomm)

    # Theoretical max reference line
    max_rounds = max(len(x_comm), len(x_nocomm))
    x_ref = np.arange(0, max_rounds + 1)
    y_ref = max_per_round * x_ref
    ax.plot(x_ref, y_ref, linestyle="--", linewidth=1.2, color="gray", alpha=0.7)

    # Axes and spines
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Reward")

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='x', which='major', direction='out', length=8, width=1.1)
    ax.tick_params(axis='y', which='major', direction='out', length=8, width=1.1)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='x', which='minor', direction='out', length=4, width=1.0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ('bottom', 'left'):
        ax.spines[spine].set_linewidth(1.1)

    # Axis limits
    xmax = max(x_comm[-1] if len(x_comm) else 0, x_nocomm[-1] if len(x_nocomm) else 0)
    ymax = max(max(cum_rewards_comm) if len(cum_rewards_comm) else 0,
               max(cum_rewards_no_comm) if len(cum_rewards_no_comm) else 0,
               y_ref[-1])
    ax.set_xlim(0, xmax * 1.03 + 1)
    ax.set_ylim(0, ymax * 1.05)

    fig.savefig(os.path.join(save_path, "cumulative_reward.pdf"), dpi=450, bbox_inches="tight", transparent=True, format="pdf")
    plt.close(fig)

def calculate_optimalarm_rate(data, optimal_arm):
    """Compute per-round rate of choosing the optimal arm across experiments."""
    num_rounds = len(data[0]["results"])
    exploration_rates = []

    for round_num in range(num_rounds):
        round_choices = [data_exp["results"][int(round_num)]["choices"] for data_exp in data]
        exploration_rate = np.mean([np.mean([1 if choice == optimal_arm else 0 for choice in choices]) for choices in round_choices])
        exploration_rates.append(exploration_rate)

    return exploration_rates

def calculate_exploration_rate(data):
    """Compute per-round exploration rate (fraction choosing a non-locally-optimal arm)."""
    num_rounds = len(data[0]["results"])
    num_agents = len(data[0]["results"][0]["choices"])
    exploration_rates = []

    for round_num in range(num_rounds):
        round_explorations = 0
        total_agents = 0

        for data_exp in data:
            for agent_index in range(num_agents):
                choices = [round_data["choices"][agent_index] for round_data in data_exp["results"][:round_num + 1]]
                rewards = [round_data['rewards'][agent_index] for round_data in data_exp["results"][:round_num + 1]]

                # Find the arm this agent currently considers best
                choice_counts = np.bincount(choices, weights=rewards, minlength=len(choices))
                current_best_arm = np.argmax(choice_counts)

                if data_exp["results"][int(round_num)]["choices"][agent_index] != current_best_arm:
                    round_explorations += 1

                total_agents += 1

        exploration_rate = round_explorations / total_agents
        exploration_rates.append(exploration_rate)

    return exploration_rates

def plot_and_save_exploration_rate(exploration_rate_comm, exploration_rate_no_comm, save_path):
    """Plot exploration rate curves and save to PNG."""
    plt.figure(figsize=(10, 6))
    plt.plot(exploration_rate_comm, label="With Communication", linestyle='-', marker='o', markersize=marker_size)
    plt.plot(exploration_rate_no_comm, label="No Communication", linestyle='-', marker='x', markersize=marker_size)
    plt.xlabel("Round")
    plt.ylabel("Exploration Rate")
    plt.title("Exploration Rate Over Rounds")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "exploration_rate.png"), dpi=600)
    plt.close()

def plot_and_save_optimalarm_rate(exploration_rate_comm, exploration_rate_no_comm, save_path):
    """Plot optimal arm choice rate curves and save to PNG."""
    plt.figure(figsize=(10, 6))
    plt.plot(exploration_rate_comm, label="With Communication", linestyle='-', marker='o', markersize=marker_size)
    plt.plot(exploration_rate_no_comm, label="No Communication", linestyle='-', marker='x', markersize=marker_size)
    plt.xlabel("Round")
    plt.ylabel("Optimal Arm Choice Rate")
    plt.title("Optimal Arm Choice Over Rounds")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "optimalarm_rate.png"), dpi=600)
    plt.close()

def find_optimal_arm(arms_probabilities):
    """Return the index of the arm with the highest success probability."""
    return np.argmax(arms_probabilities)

def calculate_avg_round_to_find_optimal(data, optimal_arm):
    """Compute the mean and SEM round at which each agent first selects the optimal arm."""
    rounds_to_find_optimal = []

    for experiment in data:
        for agent_choices in zip(*[round_data["choices"] for round_data in experiment["results"]]):
            found_rounds = [round_num for round_num, choice in enumerate(agent_choices) if choice == optimal_arm]
            if found_rounds:
                rounds_to_find_optimal.append(found_rounds[0])

    return np.mean(rounds_to_find_optimal), np.std(rounds_to_find_optimal) / np.sqrt(len(rounds_to_find_optimal))

def plot_avg_round_to_find_optimal(avg_round_comm, std_err_comm, avg_round_no_comm, std_err_no_comm, save_path):
    """Bar chart comparing average rounds to first find the optimal arm."""
    labels = ['With Communication', 'No Communication']
    avg_rounds = [avg_round_comm, avg_round_no_comm]
    errors = [std_err_comm, std_err_no_comm]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, avg_rounds, yerr=errors, capsize=5, color=['skyblue', 'salmon'])
    plt.xlabel("Condition")
    plt.ylabel("Average Round to Find Optimal Arm")
    plt.title("Comparison of Average Rounds to Find Optimal Arm")
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(save_path, "avg_round_to_find_optimal.png"), dpi=600)
    plt.close()

def calculate_reward_improvement(comm_data, no_comm_data):
    """
    Compute the social-learning reward improvement over asocial learning.
    Returns b (% improvement), 95% bootstrap CI, and Welch's t-test p-value.
    """
    final_rewards_comm = []
    final_rewards_no_comm = []

    # Compute total cumulative reward per experiment
    for data_exp in comm_data:
        final_rewards_comm.append(np.sum([round_data.get("rewards", []) for round_data in data_exp["results"]]))

    for data_exp in no_comm_data:
        final_rewards_no_comm.append(np.sum([round_data.get("rewards", []) for round_data in data_exp["results"]]))

    mean_comm = np.mean(final_rewards_comm)
    mean_no_comm = np.mean(final_rewards_no_comm)
    print(mean_comm, mean_no_comm)
    print(mean_comm - mean_no_comm)

    # Percentage improvement
    b = (mean_comm - mean_no_comm) / mean_no_comm * 100

    # Bootstrap 95% CI (10,000 resamples)
    def bootstrap_statistic(data_comm, data_no_comm):
        sample_comm = np.random.choice(data_comm, size=len(data_comm), replace=True)
        sample_no_comm = np.random.choice(data_no_comm, size=len(data_no_comm), replace=True)
        return (np.mean(sample_comm) - np.mean(sample_no_comm)) / np.mean(sample_no_comm) * 100

    bootstrap_results = [bootstrap_statistic(final_rewards_comm, final_rewards_no_comm) for _ in range(10000)]
    ci_lower, ci_upper = np.percentile(bootstrap_results, [2.5, 97.5])

    # Welch's t-test
    t_stat, p_value = st.ttest_ind(final_rewards_comm, final_rewards_no_comm, equal_var=False)

    results = {
        "b_increase_percentage": b,
        "b_ci_lower": ci_lower,
        "b_ci_upper": ci_upper,
        "p_value": p_value
    }
    print(f"b = {b:.2f}, 95% CI [{ci_lower:.2f}, {ci_upper:.2f}], p = {p_value:.4f}")

    return results

if __name__ == "__main__":
    comm_folder = sys.argv[1]
    no_comm_folder = sys.argv[2]

    comm_data = load_experiment_data(comm_folder)
    no_comm_data = load_experiment_data(no_comm_folder)

    save_path = comm_folder
    os.makedirs(save_path, exist_ok=True)

    # Arm index with highest success probability
    optimal_arm = 1

    # Cumulative reward
    cum_rewards_comm = calculate_cumulative_reward_over_rounds(comm_data)
    cum_rewards_no_comm = calculate_cumulative_reward_over_rounds(no_comm_data)

    plot_and_save_cumulative_reward(cum_rewards_comm, cum_rewards_no_comm, save_path)

    # # Optimal arm rate
    # optimalarm_rate_comm = calculate_optimalarm_rate(comm_data, optimal_arm)
    # optimalarm_rate_no_comm = calculate_optimalarm_rate(no_comm_data, optimal_arm)
    # plot_and_save_optimalarm_rate(optimalarm_rate_comm, optimalarm_rate_no_comm, save_path)

    # # Average rounds to first find optimal arm
    # avg_round_comm, std_err_comm = calculate_avg_round_to_find_optimal(comm_data, optimal_arm)
    # avg_round_no_comm, std_err_no_comm = calculate_avg_round_to_find_optimal(no_comm_data, optimal_arm)
    # plot_avg_round_to_find_optimal(avg_round_comm, std_err_comm, avg_round_no_comm, std_err_no_comm, save_path)

    # # Exploration rate
    # exploration_rate_comm = calculate_exploration_rate(comm_data)
    # exploration_rate_no_comm = calculate_exploration_rate(no_comm_data)
    # plot_and_save_exploration_rate(exploration_rate_comm, exploration_rate_no_comm, save_path)

    calculate_reward_improvement(comm_data, no_comm_data)

    print(f"Plots saved in {save_path}")
