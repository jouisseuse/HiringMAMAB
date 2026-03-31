"""
LLM-based MAMAB simulation — unequal-productivity condition.

Each agent is a GPT-4o instance that reads a running summary of arm outcomes
and responds with its next arm choice. In the social learning condition, the
summary reflects pooled outcomes from all agents; in the asocial condition,
each agent sees only its own history.

State is checkpointed after every round so interrupted runs can resume.

Usage:
    Set API_KEY and OPENAI_BASE_URL environment variables, then run:
        python mamab_llm.py
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, List

import httpx
import numpy as np
from openai import OpenAI

MODEL_NAME  = "gpt-4o"
TEMPERATURE = 0.0


# ── Client initialization ─────────────────────────────────────────────────────

def initialize_clients_sync(num_agents: int) -> List[OpenAI]:
    """Create one OpenAI client per agent."""
    proxy_url = os.environ.get("HTTPS_PROXY", "http://127.0.0.1:7890")
    clients = []
    for _ in range(num_agents):
        clients.append(OpenAI(
            api_key=os.environ.get("API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
            http_client=httpx.Client(proxies={"http://": proxy_url, "https://": proxy_url}),
        ))
    return clients


# ── Prompt helpers ────────────────────────────────────────────────────────────

def generate_summary(reward_history: Dict) -> str:
    """Format arm history as a human-readable summary for the LLM prompt."""
    lines = []
    for arm, counts in reward_history.items():
        total = counts["1"] + counts["0"]
        if total > 0:
            avg = counts["1"] / total
            lines.append(f"A{arm + 1}: {total} times, average reward: {avg:.2f}")
        else:
            lines.append(f"A{arm + 1}: 0 times")
    return "\n".join(lines)


# ── Core classes ──────────────────────────────────────────────────────────────

class Arm:
    def __init__(self, success_prob: float):
        self.success_prob = success_prob

    def sample_reward(self) -> int:
        return np.random.binomial(1, self.success_prob)


class Agent:
    def __init__(self, name: str, llm_client: OpenAI, num_arms: int, communication: bool):
        self.name = name
        self.num_arms = num_arms
        self.llm_client = llm_client
        self.communication = communication
        self.rewards = np.zeros(num_arms)
        self.counts = np.zeros(num_arms)
        self.alpha = np.ones(num_arms)
        self.beta = np.ones(num_arms)
        self.llm_prompt: List[dict] = []
        self.llm_choices: List[tuple] = []
        self.reward_history: Dict = {arm: {"1": 0, "0": 0} for arm in range(num_arms)}

    def select_arm(self, round_idx: int, communication: bool, global_reward_history: Dict):
        # Build system prompt on the first call
        if round_idx == 0:
            if communication:
                content = (
                    "You are one of several bandit algorithms tasked with maximizing your rewards "
                    "by playing one of 10 arms labeled A1-A10. Each arm is associated with a "
                    "Bernoulli distribution that gives a reward of either 0 or 1, with unknown "
                    "probabilities. Your goal is to maximize your total reward over a series of "
                    "rounds by deciding which arm to pull next, based on the feedback provided "
                    "after each round. At each time step, you will receive a summary of the results "
                    "not only about your own choices and rewards but also about the choices and "
                    "rewards of other agents. You must decide which arm (A1-A10) to pull, based on "
                    "your understanding of the rewards so far. Let's think step by step to make sure "
                    "we make the best decision. After deciding, provide your final answer within the "
                    "tags <Answer>ARM</Answer> where ARM is one of A1-A10."
                )
            else:
                content = (
                    "You are a bandit algorithm tasked with maximizing your rewards by playing one "
                    "of 10 arms labeled A1-A10. Each arm is associated with a Bernoulli distribution "
                    "that gives a reward of either 0 or 1, with unknown probabilities. Your goal is "
                    "to maximize your total reward over a series of rounds by deciding which arm to "
                    "pull next, based on the feedback provided after each round. At each time step, "
                    "you will receive a summary of the results from all previous rounds. You must "
                    "decide which arm (A1-A10) to pull, based on your understanding of the rewards "
                    "so far. Let's think step by step to make sure we make the best decision. After "
                    "deciding, provide your final answer within the tags <Answer>ARM</Answer> where "
                    "ARM is one of A1-A10."
                )
            self.llm_prompt.append({"role": "system", "content": content})

        last_choice, last_reward = self.llm_choices[-1] if self.llm_choices else (0, 1)
        self.llm_prompt.append({
            "role": "assistant",
            "content": f"A{last_choice + 1}.",
        })

        summary = generate_summary(global_reward_history if communication else self.reward_history)
        if communication:
            user_content = (
                f"In the last round, you chose A{last_choice + 1} and the reward is {last_reward}. "
                f"So far you have played {round_idx + 1} times. Your previous choices and rewards, "
                f"along with shared information from other agents, are summarized as follows:\n"
                f"{summary}\n"
                "Which arm will you choose next? Remember, YOU MUST provide your final answer within "
                "the tags <Answer>ARM</Answer> where ARM is one of A1-A10. Let's think step by step "
                "to make the best choice."
            )
        else:
            user_content = (
                f"In the last round, you chose A{last_choice + 1} and the reward is {last_reward}. "
                f"So far you have played {round_idx + 1} times with your past choices and rewards "
                f"summarized as follows:\n{summary}\n"
                "Which arm will you choose next? Remember, YOU MUST provide your final answer within "
                "the tags <Answer>ARM</Answer> where ARM is one of A1-A10. Let's think step by step "
                "to make the best choice."
            )
        self.llm_prompt.append({"role": "user", "content": user_content})

        response = self._query_llm()
        chosen_arm = self._parse_response(response)
        return chosen_arm, response

    def _query_llm(self) -> str:
        resp = self.llm_client.chat.completions.create(
            messages=self.llm_prompt,
            model=MODEL_NAME,
            temperature=TEMPERATURE,
        )
        return resp.choices[0].message.content.strip()

    @staticmethod
    def _parse_response(response: str) -> int:
        """Extract arm index (0-based) from LLM response."""
        match = re.search(r"<Answer>A(10|[1-9])</Answer>", response)
        if not match:
            match = re.search(r"A(10|[1-9])", response)
        if match:
            return int(match.group(1)) - 1
        raise ValueError(f"No valid arm choice found in response: {response!r}")

    def update(self, arm_index: int, reward: int):
        self.counts[arm_index] += 1
        self.rewards[arm_index] += reward
        self.alpha[arm_index] += reward
        self.beta[arm_index] += 1 - reward
        self.llm_choices.append((arm_index, reward))
        self.reward_history[arm_index][str(reward)] += 1

    def share_information(self, shared_alpha, shared_beta):
        self.alpha = shared_alpha
        self.beta = shared_beta

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "rewards": self.rewards.tolist(),
            "counts": self.counts.tolist(),
            "alpha": self.alpha.tolist(),
            "beta": self.beta.tolist(),
            "llm_prompt": self.llm_prompt,
            "llm_choices": self.llm_choices,
            "reward_history": self.reward_history,
            "communication": self.communication,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Agent":
        client = initialize_clients_sync(1)[0]
        agent = cls(data["name"], client, len(data["rewards"]), data["communication"])
        agent.rewards = np.array(data["rewards"])
        agent.counts = np.array(data["counts"])
        agent.alpha = np.array(data["alpha"])
        agent.beta = np.array(data["beta"])
        agent.llm_prompt = data["llm_prompt"]
        agent.llm_choices = data["llm_choices"]
        agent.reward_history = {int(k): v for k, v in data["reward_history"].items()}
        return agent


class BanditEnvironment:
    # Success probabilities for the 10 candidate arms (unequal condition)
    ARM_PROBS = [0.53, 0.95, 0.30, 0.10, 0.68, 0.40, 0.16, 0.85, 0.22, 0.12]

    def __init__(self, num_arms: int, num_agents: int, ground_truth: str,
                 rounds: int, communication: bool, model: str,
                 state_filename: str, log_file: str):
        self.num_arms = num_arms
        self.rounds = rounds
        self.communication = communication
        self.model = model

        probs = sorted(self.ARM_PROBS) if ground_truth == "different" else [0.9] * num_arms
        self.arms = [Arm(p) for p in probs]

        if os.path.exists(state_filename):
            self.agents, self.global_reward_history, self.current_round = _load_state(state_filename)
        else:
            self.global_reward_history = {i: {"1": 0, "0": 0} for i in range(num_arms)}
            self.current_round = 0
            clients = initialize_clients_sync(num_agents)
            self.agents = [Agent(f"Agent{i+1}", clients[i], num_arms, communication) for i in range(num_agents)]
            self._initialize_first_round(log_file)

    def _initialize_first_round(self, log_file: str):
        """Assign each agent to a unique arm on round 0 with reward=1."""
        round_log = {"round": 0, "choices": [], "rewards": [], "responses": []}
        for i, agent in enumerate(self.agents):
            agent.update(i, reward=1)
            self.global_reward_history[i]["1"] += 1
            round_log["choices"].append(i)
            round_log["rewards"].append(1)
        _append_log(log_file, round_log)

    def run_simulation(self, log_file: str, state_filename: str):
        with open(log_file, "a") as f:
            f.write('{"results": [')

        for round_num in range(self.current_round, self.rounds):
            round_log = {"round": round_num + 1, "choices": [], "rewards": [], "responses": []}

            for i, agent in enumerate(self.agents):
                arm_index, llm_response = agent.select_arm(round_num, self.communication, self.global_reward_history)
                reward = self.arms[arm_index].sample_reward()
                agent.update(arm_index, reward)
                round_log["choices"].append(arm_index)
                round_log["rewards"].append(reward)
                round_log["responses"].append({
                    "agent": agent.name,
                    "choice": arm_index + 1,
                    "reward": reward,
                    "response": llm_response,
                })

            for arm, reward in zip(round_log["choices"], round_log["rewards"]):
                self._update_global_history(arm, reward)

            if self.communication:
                shared_alpha = np.mean([a.alpha for a in self.agents], axis=0)
                shared_beta  = np.mean([a.beta  for a in self.agents], axis=0)
                for agent in self.agents:
                    agent.share_information(shared_alpha, shared_beta)

            _append_log(log_file, round_log)
            _save_state(state_filename, self.agents, self.global_reward_history, round_num + 1)

        with open(log_file, "a") as f:
            f.write("]}")

    def _update_global_history(self, arm: int, reward: int):
        self.global_reward_history[arm][str(reward)] = (
            self.global_reward_history[arm].get(str(reward), 0) + 1
        )


# ── State persistence ─────────────────────────────────────────────────────────

def _save_state(filename: str, agents: List[Agent], global_reward_history: Dict, current_round: int):
    """Checkpoint agent state, reward history, and current round to disk."""
    state = {
        "agents": [a.to_dict() for a in agents],
        "global_reward_history": global_reward_history,
        "current_round": current_round,
    }
    with open(filename, "w") as f:
        json.dump(state, f, indent=4)


def _load_state(filename: str):
    """Restore agents, reward history, and current round from a checkpoint."""
    with open(filename, "r") as f:
        state = json.load(f)
    global_reward_history = {int(k): v for k, v in state["global_reward_history"].items()}
    agents = [Agent.from_dict(d) for d in state["agents"]]
    return agents, global_reward_history, state["current_round"]


def _append_log(log_file: str, round_log: dict):
    with open(log_file, "a") as f:
        json.dump(round_log, f)
        f.write(",\n")


# ── Experiment runner ─────────────────────────────────────────────────────────

def setup_experiment(num_agents: int, num_arms: int, ground_truth: str,
                     rounds: int, communication: bool,
                     experiment_repeats: int = 10, model: str = MODEL_NAME):
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_folder  = f"experiment_logs_{timestamp}"
    state_folder = "state"
    os.makedirs(log_folder, exist_ok=True)

    for exp_num in range(1, experiment_repeats + 1):
        log_file   = os.path.join(log_folder,  f"experiment_{exp_num}_log.json")
        state_file = os.path.join(state_folder, f"experiment_{exp_num}_state.json")
        env = BanditEnvironment(
            num_arms=num_arms, num_agents=num_agents, ground_truth=ground_truth,
            rounds=rounds, communication=communication, model=model,
            state_filename=state_file, log_file=log_file,
        )
        env.run_simulation(log_file, state_file)
        print(f"Experiment {exp_num} completed → {log_file}")


if __name__ == "__main__":
    setup_experiment(
        num_agents=10,
        num_arms=10,
        ground_truth="different",
        rounds=200,
        communication=False,
        experiment_repeats=3,
        model=MODEL_NAME,
    )
