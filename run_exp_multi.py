import os
import sys
import csv
import numpy as np
import logging
import httpx
import re
import pickle
from openai import OpenAI, AsyncOpenAI, DefaultHttpxClient, DefaultAsyncHttpxClient
from datetime import datetime
from pprint import pprint

# Function to initialize multiple OpenAI clients
def initialize_clients_sync():
    """Initialize a list of AsyncOpenAI clients."""
    return OpenAI(
        api_key=os.environ.get("API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        http_client=DefaultHttpxClient(
            proxies="http://127.0.0.1:7890",
            transport=httpx.HTTPTransport(local_address="0.0.0.0"),
        )
    )

# UCB strategy class
class UCB:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.counts = np.full(num_arms,0)  # Number of times each arm has been selected
        self.values = np.full(num_arms,0.5)  # Average reward for each arm

    def select_arm(self):
        ucb_values = np.zeros(self.num_arms)
        total_counts = np.sum(self.counts)  # Total number of selections so far
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                return arm  # Select arm if it has not been explored yet
            bonus = np.sqrt(2 * np.log(total_counts) / self.counts[arm])
            ucb_values[arm] = self.values[arm] + bonus  # UCB formula

        max_value = np.max(ucb_values)
        max_indices = np.where(ucb_values == max_value)[0]
        chosen_arm = np.random.choice(max_indices)
        return chosen_arm  # Return the arm with the highest UCB value

    def update(self, chosen_arm, reward):
        # Update the selected arm's count and reward value
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        self.values[chosen_arm] += (reward - self.values[chosen_arm]) / n

# Thompson Sampling strategy class
class ThompsonSampling:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.successes = np.ones(num_arms)  # Success counts (reward = 1)
        self.failures = np.ones(num_arms)  # Failure counts (reward = 0)

    def select_arm(self):
        # Draw random samples from Beta distribution for each arm
        samples = np.random.beta(self.successes, self.failures)
        return np.argmax(samples)  # Select the arm with the highest sample

    def update(self, chosen_arm, reward):
        # Update the success or failure count based on the reward
        if reward == 1:
            self.successes[chosen_arm] += 1
        else:
            self.failures[chosen_arm] += 1

# e-greedy strategy class
class EGreedy:
    def __init__(self, num_arms, epsilon):
        self.num_arms = num_arms
        self.epsilon = epsilon  # Probability of exploring
        self.counts = np.full(num_arms,0)  # Number of times each arm has been selected
        self.values = np.full(num_arms,0.5)  # Average reward for each arm

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_arms)  # Explore randomly
        return np.argmax(self.values)  # Exploit the best arm

    def update(self, chosen_arm, reward):
        # Update the selected arm's count and reward value
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        self.values[chosen_arm] += (reward - self.values[chosen_arm]) / n


# Save the state including agent_choices
def save_state(filename, agent_states, global_reward_history, current_round, agent_choices, arm_estimates):
    state = {
        'agent_states': agent_states,
        'global_reward_history': global_reward_history,
        'current_round': current_round,
        'agent_choices': agent_choices,
        'arm_estimates': arm_estimates
    }
    with open(filename, 'wb') as f:
        pickle.dump(state, f)
    # logging.info(f"State saved to {filename}")


# Load the state including agent_choices
def load_state(filename):
    with open(filename, 'rb') as f:
        state = pickle.load(f)
    logging.info(f"State loaded from {filename}")
    return (state['agent_states'], state['global_reward_history'], state['current_round'], state['agent_choices'], state['arm_estimates'])  # Return agent_choices


def get_choice_prompt(llm_response):
    match = re.search(r'<Answer>A([1-4])</Answer>', llm_response)
    if match:
        return int(match.group(1))
    else:
        match = re.search(r'A([1-4])', llm_response)
        if match:
            return int(match.group(1))
        else:
            raise ValueError("LLM's response does not contain a valid arm choice.")

# Function to update reward history for each arm

# Function to generate a summary of rewards for each arm
def generate_summary(reward_history):
    summary = ""
    for arm in reward_history:
        summary += f"A{arm + 1}: 1: {reward_history[arm]['1']} times, 0: {reward_history[arm]['0']} times\n"
    return summary

def generate_summary_ref(reward_history):
    summary = ""
    for arm in reward_history:
        total_plays = reward_history[arm]['1'] + reward_history[arm]['0']
        if total_plays > 0:
            avg_reward = reward_history[arm]['1'] / total_plays
            summary += f"A{arm + 1}: {total_plays} times, average reward: {avg_reward:.2f}\n"
        else:
            summary += f"A{arm + 1}: {total_plays} times\n"
    return summary

def save_results_to_csv(agent_choices, arm_estimates, filename):

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Dynamically generate header based on the number of agents
        agent_names = list(agent_choices.keys())
        header = ["Round"]

        # For each agent, add two columns for choice and reward
        for agent in agent_names:
            header.append(f"{agent}_choice")
            header.append(f"{agent}_reward")

        # Add columns for shared arm estimates (A1-A4)
        header.append("A1_estimate")
        header.append("A2_estimate")
        header.append("A3_estimate")
        header.append("A4_estimate")

        # Write the header to the CSV
        writer.writerow(header)

        # Determine the number of rounds (all agents should have the same number of rounds)
        num_rounds = len(agent_choices[agent_names[0]])

        # Write each round's data
        for round_idx in range(num_rounds):
            row = [round_idx]  # Start with the round number
            for agent in agent_names:
                agent_choice, agent_reward = agent_choices[agent][round_idx]
                row.extend([agent_choice, agent_reward])  # Add choice and reward for each agent

            # Add shared arm estimates for this round
            row.extend(arm_estimates[round_idx])  # Add estimates for arms A1-A4

            # Write the row to the CSV
            writer.writerow(row)

# Save results to CSV with agent choices and rewards
def save_results_to_csv_multi(agent_choices, filename):
    # Assume `agent_choices` is a dictionary where keys are agent names (e.g., "Agent1", "Agent2", etc.)
    # and values are lists of tuples (choice, reward) for each round.
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Dynamically generate header based on the number of agents
        agent_names = list(agent_choices.keys())
        header = ["Round"]

        # For each agent, add two columns: one for choice and one for reward
        for agent in agent_names:
            header.append(f"{agent}_choice")
            header.append(f"{agent}_reward")

        # Write the header to the CSV
        writer.writerow(header)

        # Determine the number of rounds (all agents should have the same number of rounds)
        num_rounds = len(agent_choices[agent_names[0]])

        # Write each round's data
        for round_idx in range(num_rounds):
            row = [round_idx]  # Start with the round number
            for agent in agent_names:
                agent_choice, agent_reward = agent_choices[agent][round_idx]
                row.extend([agent_choice, agent_reward])  # Add choice and reward for each agent

            # Write the row to the CSV
            writer.writerow(row)

# Calculate arm estimates based on global_reward_history
def calculate_arm_estimates(global_reward_history):
    estimates = []
    for arm in range(4):  # Assume there are 4 arms (A1 to A4)
        total_rewards = global_reward_history[arm]['1']
        total_plays = global_reward_history[arm]['1'] + global_reward_history[arm]['0']
        if total_plays == 0:
            estimates.append(0)  # If the arm has never been played, estimate is 0
        else:
            estimates.append(total_rewards / total_plays)  # Average reward
    return estimates

# Main function to run the multi-armed bandit experiment
def run_multi_llm_bandit_experiment(turn_num, num_rounds, llm_agents, theta_params, state_filename, logger, csv_filename):
    def update_reward_history(reward_history, arm, reward):
        if reward == 1:
            reward_history[arm]['1'] += 1
        else:
            reward_history[arm]['0'] += 1
    # Try to load the state if it exists, otherwise start from scratch
    try:
        agent_states, global_reward_history, current_round, agent_choices, arm_estimates = load_state(state_filename)
        logger.info("Resuming from saved state")
    except FileNotFoundError:
        # Initialize each LLM agent's state based on the number of agents (llm_agents)
        agent_states = []
        global_reward_history = {
            0: {'1': 0, '0': 0},  # A1
            1: {'1': 0, '0': 0},  # A2
            2: {'1': 0, '0': 0},  # A3
            3: {'1': 0, '0': 0}   # A4
        }
        agent_choices = {}  # Initialize agent choices
        arm_estimates = []  # list of lists

        arm_estimates.append([0, 0, 0, 0])

        for i in range(llm_agents):
            agent_name = f"Agent{i+1}"  # Assign each agent a name like Agent1, Agent2, etc.
            initial_arm = i % 4 + 1
            chosen_arms = {
                agent_name: [initial_arm]
            }
            llm_choices = [(initial_arm, 1)]  # Initial reward for this LLM agent

            
            # Set the initial system-level prompt to include multi-agent information sharing
            llm_prompt = [
                {"role": "system", "content": (
                    "You are one of several bandit algorithms playing a multi-armed bandit game. "
                    "Your goal is to maximize your rewards by playing one of 4 arms labeled A1, A2, A3, and A4. "
                    "Each arm is associated with a Bernoulli distribution that gives a reward of either 0 or 1, with unknown probabilities. "
                    "At each time step, you will receive a summary of the results not only about your own choices and rewards but also about the choices and rewards of other agents. "
                    "Use this shared information to make better decisions over time. "
                    "Let’s think step by step to make sure we make the best decision."
                    "After deciding, provide your final answer within the tags <Answer>ARM</Answer> where ARM is one of A1, A2, A3, A4."
                )}
            ]

            # Initialize LLM agent's reward history
            reward_history = {
                0: {'1': 0, '0': 0},  # A1
                1: {'1': 0, '0': 0},  # A2
                2: {'1': 0, '0': 0},  # A3
                3: {'1': 0, '0': 0}   # A4
            }
            # Simulate agent's initial reward
            initial_reward = 1
            update_reward_history(reward_history, initial_arm - 1, initial_reward)
            update_reward_history(global_reward_history, initial_arm - 1, initial_reward)

            # Store initialization details including llm_prompt
            agent_states.append({
                "name": agent_name,
                "chosen_arms": chosen_arms,
                "llm_choices": llm_choices,
                "reward_history": reward_history,
                "llm_prompt": llm_prompt # Store llm_prompt for the agent
            })

            # Initialize agent_choices for this agent
            agent_choices[agent_name] = []
            agent_choices[agent_name].append((initial_arm, 1))
            
        
        current_round = 0

    llm_agents = [ initialize_clients_sync() for _ in range(llm_agents)]

    logger.info(f"==========Experiment Start========")
    print(f"==========Experiment Start========")
    
    for round_idx in range(current_round, num_rounds):
        logger.info(f"========Round {round_idx+1}======")
        print(f"========Round {round_idx+1}======")

        current_estimates = calculate_arm_estimates(global_reward_history)
        arm_estimates.append(current_estimates)

        # Each LLM agent takes turns to make a decision
        for i, agent_state in enumerate(agent_states):
            agent_name = agent_state["name"]
            chosen_arms = agent_state["chosen_arms"]
            llm_choices = agent_state["llm_choices"]
            reward_history = agent_state["reward_history"]
            llm_prompt = agent_state["llm_prompt"]  # Retrieve llm_prompt from state

            # Construct the prompt for this agent, summarizing its choices and shared information from other agents
            prompt1 = {
                "role": "assistant",
                "content": (f"A{llm_choices[-1][0]}.")
            }
            summary_results = generate_summary_ref(global_reward_history)  # Summarizing global (shared) reward history
            prompt2 = {
                "role": "user",
                "content": (
                    f"So far, you have played {round_idx+1} times. Your previous choices and rewards, along with shared information from other agents, are summarized as follows:\n"
                    f"{summary_results}"
                    "Which arm will you choose next? Provide your final answer in the format <Answer>ARM</Answer>, where ARM is A1, A2, A3, or A4. Let’s think step by step to make the best choice."
                )
            }
            # Update llm_prompt and store it
            llm_prompt.append(prompt1)
            llm_prompt.append(prompt2)
            agent_state["llm_prompt"] = llm_prompt  # Update llm_prompt in state
            
            # Call LLM agent to make a decision
            response_agent = llm_agents[i].chat.completions.create(
                messages=llm_prompt,
                model=MODEL_NAME,
                temperature=TEMPERATURE
            )
            llm_response = response_agent.choices[0].message.content.strip()
            logger.info(f"{agent_name} Response: {[llm_response]}")
            llm_arm = get_choice_prompt(llm_response)
            chosen_arms[agent_name].append(llm_arm)

            # Reward calculation
            theta = theta_params[llm_arm - 1]
            llm_reward = np.random.binomial(1, theta)

            # Store LLM agent's choice and reward
            llm_choices.append((llm_arm, llm_reward))

            agent_choices[agent_name].append((llm_arm, llm_reward))

            # Update this agent's reward history and global reward history
            update_reward_history(reward_history, llm_arm - 1, llm_reward)

            logger.info(f"{agent_name} chose: {llm_arm}, reward: {llm_reward}")
            print(f"{agent_name} chose: {llm_arm}, reward: {llm_reward}")
        
        for i, agent_state in enumerate(agent_states):
            agent_name = agent_state["name"]
            choice = agent_choices[agent_name][-1]
            update_reward_history(global_reward_history, choice[0] - 1, choice[1])

        # Save the state after each round
        save_state(state_filename, agent_states, global_reward_history, round_idx + 1, agent_choices, arm_estimates)

    logger.info(f"==========Experiment End========")
    print(f"==========Experiment End========")

    # Save results of each agent to CSV file
    save_results_to_csv(agent_choices, arm_estimates, csv_filename)
    print("CSV file saved")

# Configure logging to store LLM responses in a log file
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# log_dir = f'result_dir/log_{timestamp}_single'
# os.makedirs(log_dir, exist_ok=True)
# logging.basicConfig(filename=f"{log_dir}/llm_responses.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.getLogger("http").setLevel(logging.WARNING)
# logging.getLogger("urllib3").setLevel(logging.WARNING)
# logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("httpcore").setLevel(logging.WARNING)
# logging.getLogger("openai").setLevel(logging.WARNING)


# Experiment setup
NUM_TURN = 10
num_rounds = 40
agent_num = 4

# scenario: bandit v.s. hiring
SCENARIO = "multibandit"
# model_name: gpt-4o-mini, gpt-4o, o1-preview
MODEL_NAME = "gpt-4o"
# Structure: Memory Chain(MC) v.s. Summary Chain(SC) v.s. Memory Chain+ Summary Chain(MCSC)
STRUCTURE = "MCSC"
# In-context example: Chain of thought(COT) v.s. hint(exploration&exploitation)(HINT) v.s. none(NONE)
IN_CONTEXT_EXAMPLE = "COT"
# Features types: Symbolized() v.s. Insensitive Semantic v.s. Sensitive semantic
FEATURES_TYPES = "Symbolized"
TEMPERATURE = 0
# TEST_TYPE = f"{SCENARIO}_{MODEL_NAME}_{STRUCTURE}_{IN_CONTEXT_EXAMPLE}_{FEATURES_TYPES}"
TEST_TYPE = f"{SCENARIO}_{MODEL_NAME}_{TEMPERATURE}_REF"

# Define Beta distribution parameters for each arm (alpha, beta)
theta_params = [0.1,0.3,0.5,0.9]

# Configure logging to store LLM responses in a log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f'result_dir/log_{timestamp}_{TEST_TYPE}'
state_dir = f'result_dir/state_multi'
os.makedirs(log_dir, exist_ok=True)

# Run the experiment and save results
for turn in range(NUM_TURN):
    log_filename = os.path.join(log_dir, f'{turn + 1}.log')
    state_filename = os.path.join(state_dir, f'{TEST_TYPE}_{turn + 1}_state.pkl')
    csv_filename = os.path.join(log_dir, f'{turn + 1}_results.csv')

    # create a logger with a file handler
    logger = logging.getLogger(f'Run_{turn + 1}')
    logger.setLevel(logging.DEBUG)
    
    # create a file handler and set its level to debug
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    
    # create a formatter and set it to the file handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)

    logger.info(f'Starting turn {turn + 1}')

    run_multi_llm_bandit_experiment(turn+1, num_rounds, agent_num, theta_params, state_filename, logger, csv_filename)

    logger.info(f'Finished turn {turn + 1}')
    
    logger.removeHandler(file_handler)
    

# def inspect_pickle_file(filename):
#     with open(filename, 'rb') as f:
#         state = pickle.load(f)  # Load the pickled data
#     return state  # Return the loaded state to inspect
# state = inspect_pickle_file('/gbf2/bai-lab/llm-exp/result_dir/log_20240915_205447_single/bandit_state.pkl')
# pprint(state)