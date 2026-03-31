import numpy as np
import random
import json
import os
import time
from datetime import datetime
from typing import List, Dict
from scipy.stats import beta as Beta
from joblib import Parallel, delayed


def generate_initial_states(num_agents: int, num_arms: int, max_states: int = 100, timeout_sec: float = 10.0) -> List[List[int]]:
    """
    Generate up to max_states unique initial arm assignments for num_agents agents.
    Uniqueness is defined by the multiset of arm counts, not the exact assignment.
    A timeout prevents infinite loops in sparse configurations.
    """
    start_time = time.time()
    seen_keys = set()

    while len(seen_keys) < max_states:
        if time.time() - start_time > timeout_sec:
            print(f"[Timeout] Generated {len(seen_keys)} unique states in {timeout_sec}s.")
            break
        assignment = [random.randint(0, num_arms - 1) for _ in range(num_agents)]
        key = tuple(sorted(np.bincount(assignment, minlength=num_arms)))
        seen_keys.add(key)

    # Reconstruct one concrete assignment per unique count key
    results = []
    for counts in seen_keys:
        state = []
        for arm_id, c in enumerate(counts):
            state.extend([arm_id] * c)
        random.shuffle(state)
        results.append(state)

    return results


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
    def __init__(self, num_arms: int, num_agents: int, ground_truth: str, strategy: str, epsilon: float, rounds: int, communication: bool):
        if ground_truth == "identical":
            self.arms = [Arm(0.9) for _ in range(num_arms)]
        elif ground_truth == "different":
            probs = [0.53, 0.95, 0.30, 0.10, 0.68, 0.40, 0.16, 0.85, 0.22, 0.12]
            self.arms = [Arm(p) for p in probs]

        self.agents = [
            Agent(num_arms=num_arms, strategy=strategy, epsilon=epsilon, communication=communication)
            for _ in range(num_agents)
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


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    return obj


def save_logs(base_folder: str, setting_folder: str, state_id: int, setting: str, repeat_id: int, logs: Dict):
    path = os.path.join(base_folder, setting_folder, f"state_{state_id}", setting)
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{repeat_id}.log")
    with open(file_path, "w") as f:
        json.dump(convert_to_serializable(logs), f, indent=4)


def run_single_sim(num_agents, num_arms, ground_truth, strategy, epsilon, rounds,
                   base_folder, setting_folder, state_id, state, comm, repeat_id):
    label = "communication" if comm else "non-communication"
    env = BanditEnvironment(
        num_arms=num_arms,
        num_agents=num_agents,
        ground_truth=ground_truth,
        strategy=strategy,
        epsilon=epsilon,
        rounds=rounds,
        communication=comm,
    )
    logs = env.run_simulation(list(state))
    save_logs(base_folder, setting_folder, state_id, label, repeat_id, logs)
    print(f"Done: agents={num_agents}, arms={num_arms}, comm={comm}, state={state_id}, repeat={repeat_id}")


def main():
    ground_truth = "identical"  # "identical" | "different"
    strategy = "Thompson"       # "Thompson" | "UCB" | "greedy"
    epsilon = 0.1
    rounds = 1000
    experiment_repeats = 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_folder = f"preExp_logs_{timestamp}"

    # Agent-arm scaling configurations: (num_agents, num_arms)
    settings = [
        (2,2),
        (2,5),
        (5,5),
        (5,2),
        (20,2),
        (20,5),
        (2,20),
        (5,20),
        (20,20),
        (2,100),
        (5,100),
        (20,100),
        (2,1000),
        (5,1000),
        (20,1000),
        (100,100),
        (100,2),
        (100,5),
        (100,20),
        (1000,2),
        (1000,5),
        (1000,10),
        (1000,20),
        (1000,100),
        (100,1000),
        (1000,1000),
    ]

    jobs = []
    for num_agents, num_arms in settings:
        setting_folder = f"agents{num_agents}_arms{num_arms}"
        states = generate_initial_states(num_agents, num_arms, timeout_sec=60)
        print(f"agents={num_agents}, arms={num_arms}: {len(states)} unique initial states")
        for state_id, state in enumerate(states, 1):
            for comm in [True, False]:
                for repeat_id in range(1, experiment_repeats + 1):
                    jobs.append(delayed(run_single_sim)(
                        num_agents, num_arms, ground_truth, strategy, epsilon,
                        rounds, base_folder, setting_folder,
                        state_id, state, comm, repeat_id,
                    ))

    Parallel(n_jobs=-1)(jobs)


if __name__ == "__main__":
    main()
