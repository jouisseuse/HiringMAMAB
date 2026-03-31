import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.stats as st
import pandas as pd
import seaborn as sns

marker_size = 2

def load_experiment_data(folder_path):
    """Load all playerRound.log files from a folder and return the parsed experiment data."""
    experiment_data = []
    log_files = glob.glob(os.path.join(folder_path, "**/playerRound.log"), recursive=True)

    for file_path in log_files:
        with open(file_path, "r") as f:
            data = json.load(f)
            experiment_data.append(data)

    return experiment_data

def calculate_average_reward_over_rounds(data):
    """Compute the average reward per round across all experiments."""
    num_rounds = len(data[0]["results"])
    avg_rewards = []

    for round_num in range(num_rounds):
        print(data[1]["results"][round_num])
        round_rewards = []
        for data_exp in data:
                # Check if data_exp["results"] is a list and has the expected structure
                if isinstance(data_exp["results"], list) and round_num < len(data_exp["results"]):
                    # Further check if the expected keys exist
                    if "rewards" in data_exp["results"][round_num]:
                        round_rewards.append(data_exp["results"][round_num]["rewards"])
                    else:
                        print(f"Warning: 'rewards' not found in results at round {round_num}")
                else:
                    print(f"Warning: Unexpected structure in results for {data_exp}")
        avg_reward = np.mean([np.mean(rewards) for rewards in round_rewards])
        avg_rewards.append(avg_reward)

    return avg_rewards

def plot_and_save_cumulative_reward(avg_rewards_comm, avg_rewards_no_comm, save_path):
    """Plot and save cumulative reward curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(avg_rewards_comm), label="With Communication", linestyle='-', marker='o', markersize=marker_size)
    plt.plot(np.cumsum(avg_rewards_no_comm), label="No Communication", linestyle='-', marker='x', markersize=marker_size)
    plt.xlabel("Round")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward Over Rounds")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "cumulative_reward.png"), dpi=600)
    plt.close()

    # === Compute percentage gain ===
    cumulative_comm = np.cumsum(avg_rewards_comm)
    cumulative_no_comm = np.cumsum(avg_rewards_no_comm)
    reward_gain = cumulative_comm[-1] - cumulative_no_comm[-1]
    percent_gain = (reward_gain / cumulative_no_comm[-1]) * 100

    print(f"Cumulative Reward Gain with Communication: {percent_gain:.2f}%")

def calculate_optimalarm_rate(data, optimal_arm):
    """Compute the optimal arm selection rate per round."""
    num_rounds = len(data[0]["results"])
    exploration_rates = []

    for round_num in range(num_rounds):
        round_choices = [data_exp["results"][int(round_num)]["choices"] for data_exp in data]
        exploration_rate = np.mean([np.mean([1 if choice == optimal_arm else 0 for choice in choices]) for choices in round_choices])
        exploration_rates.append(exploration_rate)

    return exploration_rates

def calculate_exploration_rate(data):
    """Compute exploration rate per round (fraction choosing non-current-best arm)."""
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

                # Identify current best arm for this agent
                choice_counts = np.bincount(choices, weights=rewards, minlength=len(choices))
                current_best_arm = np.argmax(choice_counts)

                # Check if agent chose a non-optimal arm
                if data_exp["results"][int(round_num)]["choices"][agent_index] != current_best_arm:
                    round_explorations += 1

                total_agents += 1

        # Compute round exploration rate
        exploration_rate = round_explorations / total_agents
        exploration_rates.append(exploration_rate)

    return exploration_rates

def plot_and_save_exploration_rate(exploration_rate_comm, exploration_rate_no_comm, save_path):
    """Plot and save exploration rate curves."""
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
    """Plot and save optimal arm choice rate curves."""
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
    """Compute the average round at which each agent first selects the optimal arm."""
    rounds_to_find_optimal = []

    for experiment in data:
        for agent_choices in zip(*[round_data["choices"] for round_data in experiment["results"]]):
            found_rounds = [round_num for round_num, choice in enumerate(agent_choices) if choice == optimal_arm]
            if found_rounds:
                rounds_to_find_optimal.append(found_rounds[0])

    return np.mean(rounds_to_find_optimal), np.std(rounds_to_find_optimal) / np.sqrt(len(rounds_to_find_optimal))

def plot_avg_round_to_find_optimal(avg_round_comm, std_err_comm, avg_round_no_comm, std_err_no_comm, save_path):
    """Plot and save the average rounds-to-optimal comparison bar chart."""
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
    """Compute the cumulative reward improvement of communication over no-communication, with 95% CI and p-value."""
    final_rewards_comm = []
    final_rewards_no_comm = []

    # Accumulate final total reward per experiment
    for data_exp in comm_data:
        final_rewards_comm.append(np.sum([round_data.get("rewards", []) for round_data in data_exp["results"]]))

    for data_exp in no_comm_data:
        final_rewards_no_comm.append(np.sum([round_data.get("rewards", []) for round_data in data_exp["results"]]))

    # Compute group means
    mean_comm = np.mean(final_rewards_comm)
    mean_no_comm = np.mean(final_rewards_no_comm)
    print(mean_comm, mean_no_comm)

    # Compute percentage gain b
    b = (mean_comm - mean_no_comm) / mean_no_comm * 100

    # Bootstrap 95% CI (10,000 resamples)
    def bootstrap_statistic(data_comm, data_no_comm):
        sample_comm = np.random.choice(data_comm, size=len(data_comm), replace=True)
        sample_no_comm = np.random.choice(data_no_comm, size=len(data_no_comm), replace=True)
        return (np.mean(sample_comm) - np.mean(sample_no_comm)) / np.mean(sample_no_comm) * 100

    bootstrap_results = [bootstrap_statistic(final_rewards_comm, final_rewards_no_comm) for _ in range(10000)]
    ci_lower, ci_upper = np.percentile(bootstrap_results, [2.5, 97.5])

    # Welch's t-test for p-value
    t_stat, p_value = st.ttest_ind(final_rewards_comm, final_rewards_no_comm, equal_var=False)

    results = {
        "b_increase_percentage": b,
        "b_ci_lower": ci_lower,
        "b_ci_upper": ci_upper,
        "p_value": p_value
    }
    print(results)

    return results

if __name__ == "__main__":
    # Read folder arguments
    comm_folder = sys.argv[1]
    no_comm_folder = sys.argv[2]

    # Load experiment logs
    comm_data = load_experiment_data(comm_folder)
    no_comm_data = load_experiment_data(no_comm_folder)

    # Save outputs to the communication folder
    save_path = comm_folder
    os.makedirs(save_path, exist_ok=True)

    # Optimal arm index (Bright Green = 1)
    optimal_arm = 1

    # Cumulative reward
    avg_rewards_comm = calculate_average_reward_over_rounds(comm_data)
    avg_rewards_no_comm = calculate_average_reward_over_rounds(no_comm_data)
    plot_and_save_cumulative_reward(avg_rewards_comm, avg_rewards_no_comm, save_path)

    # Optimal arm rate
    optimalarm_rate_comm = calculate_optimalarm_rate(comm_data, optimal_arm)
    optimalarm_rate_no_comm = calculate_optimalarm_rate(no_comm_data, optimal_arm)
    plot_and_save_optimalarm_rate(optimalarm_rate_comm, optimalarm_rate_no_comm, save_path)

    # Average rounds to find optimal arm
    avg_round_comm, std_err_comm = calculate_avg_round_to_find_optimal(comm_data, optimal_arm)
    avg_round_no_comm, std_err_no_comm = calculate_avg_round_to_find_optimal(no_comm_data, optimal_arm)
    plot_avg_round_to_find_optimal(avg_round_comm, std_err_comm, avg_round_no_comm, std_err_no_comm, save_path)

    # Exploration rate (commented out)
    # exploration_rate_comm = calculate_exploration_rate(comm_data)
    # exploration_rate_no_comm = calculate_exploration_rate(no_comm_data)
    # plot_and_save_exploration_rate(exploration_rate_comm, exploration_rate_no_comm, save_path)

    calculate_reward_improvement(comm_data, no_comm_data)

    print(f"Plots saved in {save_path}")
