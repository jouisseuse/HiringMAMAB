import numpy as np
import random
import json
import os
from datetime import datetime
from typing import List, Dict
from scipy.stats import beta as Beta

INITIAL_STATES_FILE = "initial_states.json"


class Arm:
    def __init__(self, success_prob: float):
        self.success_prob = success_prob

    def sample_reward(self) -> int:
        return np.random.binomial(1, self.success_prob)


class Agent:
    def __init__(self, num_arms: int, strategy: str, epsilon: float = 0.1, communication: bool = False):
        self.num_arms = num_arms
        self.strategy = strategy
        self.epsilon = epsilon
        self.rewards = np.zeros(num_arms)
        self.counts = np.zeros(num_arms)
        self.alpha = np.ones(num_arms)
        self.beta_param = np.ones(num_arms)
        self.shared_alpha = np.ones(num_arms)
        self.shared_beta = np.ones(num_arms)
        self.communication = communication

    def select_arm(self) -> int:
        if self.strategy == "greedy":
            if np.random.rand() < self.epsilon:
                return np.random.randint(0, self.num_arms)
            avg_rewards = self.rewards / (self.counts + 1e-5)
            max_value = np.max(avg_rewards)
            best_arms = np.flatnonzero(avg_rewards == max_value)
            return np.random.choice(best_arms)

        elif self.strategy == "UCB":
            alpha = self.shared_alpha if self.communication else self.alpha
            beta = self.shared_beta if self.communication else self.beta_param
            total_counts = np.sum(alpha + beta)
            if total_counts == 0:
                return random.randint(0, self.num_arms - 1)
            empirical_means = alpha / (alpha + beta + 1e-5)
            ucb_values = empirical_means + np.sqrt(2 * np.log(total_counts + 1) / (alpha + beta + 1e-5))
            max_ucb = np.max(ucb_values)
            best_arms = np.flatnonzero(ucb_values == max_ucb)
            return np.random.choice(best_arms)

        elif self.strategy == "Thompson":
            alpha = self.shared_alpha if self.communication else self.alpha
            beta = self.shared_beta if self.communication else self.beta_param
            alpha = np.maximum(alpha, 1e-9)
            beta = np.maximum(beta, 1e-9)
            samples = Beta.rvs(alpha, beta)
            max_sample = np.max(samples)
            best_arms = np.flatnonzero(np.abs(samples - max_sample) <= 1e-15)
            return int(np.random.choice(best_arms))

    def update(self, arm_index: int, reward: int):
        self.counts[arm_index] += 1
        self.rewards[arm_index] += reward
        self.alpha[arm_index] += reward
        self.beta_param[arm_index] += 1 - reward

    def share_information(self, shared_alpha, shared_beta):
        self.shared_alpha = shared_alpha
        self.shared_beta = shared_beta


class BanditEnvironment:
    def __init__(self, num_arms: int, ground_truth: str, strategy: str, epsilon: float, rounds: int, communication: bool):
        if ground_truth == "identical":
            self.arms = [Arm(0.9) for _ in range(num_arms)]
        elif ground_truth == "different":
            probs = [0.53, 0.95, 0.30, 0.10, 0.68, 0.40, 0.16, 0.85, 0.22, 0.12]
            self.arms = [Arm(p) for p in probs]

        self.agents = [
            Agent(num_arms=num_arms, strategy=strategy, epsilon=epsilon, communication=communication)
            for _ in range(len(self.arms))
        ]
        self.rounds = rounds
        self.communication = communication

    def _aggregate_beliefs(self):
        """Pool alpha/beta across all agents (shared Bayesian update)."""
        shared_alpha = np.sum([a.alpha for a in self.agents], axis=0) - len(self.agents) + 1
        shared_beta = np.sum([a.beta_param for a in self.agents], axis=0) - len(self.agents) + 1
        for agent in self.agents:
            agent.share_information(shared_alpha, shared_beta)

    def run_simulation(self, initial_state: List[int]) -> Dict:
        logs = {"results": []}

        # Round 0: forced initial arm assignments
        first_round_log = {"choices": [], "rewards": []}
        for agent, arm_choice in zip(self.agents, initial_state):
            reward = self.arms[arm_choice].sample_reward()
            agent.update(arm_choice, reward)
            first_round_log["choices"].append(arm_choice)
            first_round_log["rewards"].append(reward)

        if self.communication:
            self._aggregate_beliefs()

        logs["results"].append(first_round_log)

        # Rounds 1+: agents choose arms based on their strategy
        for _ in range(1, self.rounds):
            round_log = {"choices": [], "rewards": []}
            for agent in self.agents:
                arm_index = agent.select_arm()
                reward = self.arms[arm_index].sample_reward()
                agent.update(arm_index, reward)
                round_log["choices"].append(arm_index)
                round_log["rewards"].append(reward)

            if self.communication:
                self._aggregate_beliefs()

            logs["results"].append(round_log)

        return logs


def load_initial_states(filename: str) -> List[List[int]]:
    with open(filename, "r") as f:
        return json.load(f)


def save_logs(base_folder: str, state_id: int, setting: str, repeat_id: int, logs: Dict):
    folder_path = os.path.join(base_folder, f"state_{state_id}", setting)
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{repeat_id}.log")

    def _to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_to_serializable(v) for v in obj]
        return obj

    with open(file_path, "w") as f:
        json.dump(_to_serializable(logs), f, indent=4)


def main():
    num_arms = 10
    ground_truth = "different"  # "identical" | "different"
    strategy = "Thompson"       # "Thompson" | "UCB" | "greedy"
    epsilon = 0.1
    rounds = 1000
    experiment_repeats = 100
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_folder = f"experiment_logs_{timestamp}"

    initial_states = load_initial_states(INITIAL_STATES_FILE)
    for state_id, initial_state in enumerate(initial_states, start=1):
        for communication in [True, False]:
            setting = "communication" if communication else "non-communication"
            for repeat_id in range(1, experiment_repeats + 1):
                env = BanditEnvironment(
                    num_arms=num_arms,
                    ground_truth=ground_truth,
                    strategy=strategy,
                    epsilon=epsilon,
                    rounds=rounds,
                    communication=communication,
                )
                logs = env.run_simulation(initial_state)
                save_logs(base_folder, state_id, setting, repeat_id, logs)


if __name__ == "__main__":
    main()
