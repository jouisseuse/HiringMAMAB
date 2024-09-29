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

# Function to save the state of the game to a file
def save_state(filename, llm_prompt, chosen_arms, llm_choices, strategies, current_round, reward_history):
    state = {
        'llm_prompt': llm_prompt,
        'chosen_arms': chosen_arms,
        'llm_choices': llm_choices,
        'strategies': strategies,
        'current_round': current_round,
        'reward_history':reward_history
    }
    with open(filename, 'wb') as f:
        pickle.dump(state, f)
    # logging.info(f"State saved to {filename}")

# Function to load the state of the game from a file
def load_state(filename):
    with open(filename, 'rb') as f:
        state = pickle.load(f)
    logging.info(f"State loaded from {filename}")
    return state['llm_prompt'], state['chosen_arms'], state['llm_choices'], state['strategies'], state['current_round'], state['reward_history']

# Function to extract the LLM's chosen arm from the response text
def get_choice_prompt(llm_response):
    match = re.search(r'A([1-4])', llm_response)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("LLM's response does not contain a valid arm choice.")

# Function to update reward history for each arm
def update_reward_history(reward_history, arm, reward):
    if reward == 1:
        reward_history[arm]['1'] += 1
    else:
        reward_history[arm]['0'] += 1

# Function to generate a summary of rewards for each arm
def generate_summary(reward_history):
    summary = ""
    for arm in reward_history:
        summary += f"A{arm + 1}: 1: {reward_history[arm]['1']} times, 0: {reward_history[arm]['0']} times\n"
    return summary

# Modified save_results_to_csv function
def save_results_to_csv(chosen_arms, llm_choices, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["Round", "LLM", "UCB", "ThompsonSampling", "EGreedy", "LLM Reward"])

        for round_idx in range(len(llm_choices)):
            llm_arm, llm_reward = llm_choices[round_idx]
            ucb_arm = chosen_arms['UCB'][round_idx]
            ts_arm = chosen_arms['ThompsonSampling'][round_idx]
            eg_arm = chosen_arms['EGreedy'][round_idx]

            writer.writerow([round_idx, llm_arm, ucb_arm, ts_arm, eg_arm, llm_reward])

# Main function to run the multi-armed bandit experiment
def run_bandit_experiment(turn_num, num_rounds, strategies,theta_params, state_filename, logger, csv_filename):
    # Try to load the state if it exists, otherwise start from scratch
    try:
        llm_prompt, chosen_arms, llm_choices, strategies, current_round, reward_history = load_state(state_filename)
        logger.info("Resuming from saved state")
    except FileNotFoundError:
        llm_prompt = [
            {"role": "system", "content": "You are a bandit algorithm tasked with maximizing your rewards by playing one of 4 arms labeled A1, A2, A3, and A4. Each arm is associated with a Bernoulli distribution that gives a reward of either 0 or 1, with unknown probabilities. Your goal is to maximize your total reward over a series of rounds by deciding which arm to pull next, based on the feedback provided after each round. At each time step, you will receive a summary of the results from all previous rounds. You must decide which arm (A1, A2, A3, A4) to pull, based on your understanding of the rewards so far. Let’s think step by step to make sure we make the best decision. After deciding, provide your final answer within the tags <Answer>ARM</Answer> where ARM is one of A1, A2, A3, A4."}
        ]
        initial_arm = turn_num
        chosen_arms = {
            "llm": [initial_arm],
            "UCB": [initial_arm],
            "ThompsonSampling": [initial_arm],
            "EGreedy": [initial_arm]
        }
        llm_choices = [(initial_arm, 1)]  # Initial reward for LLM
        for strategy in strategies:
            strategy.update(initial_arm - 1, 1)
        current_round = 0
        reward_history = {
            0: {'1': 0, '0': 0},  # A1
            1: {'1': 0, '0': 0},  # A2
            2: {'1': 0, '0': 0},  # A3
            3: {'1': 0, '0': 0}   # A4
        }
        update_reward_history(reward_history, initial_arm - 1, 1)

        # logger.info("Starting a new experiment")

    llm_agent = initialize_clients_sync()

    logger.info(f"==========Game Start========")
    print(f"==========Game Start========")
    for round_idx in range(current_round,num_rounds):
        logger.info(f"========Round {round_idx+1}======")
        print(f"========Round {round_idx+1}======")
        prompt1 = {
                "role": "assistant",
                "content": (f"A{llm_choices[-1][0]}.")
            }
        summary_results = generate_summary(reward_history)
        prompt2 ={
            "role": "user", 
            "content": (
                f"In the last round, you chose A{llm_choices[-1][0]} and the reward is {llm_choices[-1][1]}."
                f"Summary of results across all rounds:\n{summary_results}"
                # "You should explore for first several rounds and exploit for later rounds. "
                # "Choose a arm and briefly explain why." 
                "Let’s think step by step to decide which arm to choose next. Give the reason behind your decision.\n"
            )
        }
        llm_prompt.append(prompt1)
        llm_prompt.append(prompt2)
        response_agent = llm_agent.chat.completions.create(
                messages=llm_prompt,
                model=MODEL_NAME
                # model="o1-preview-2024-09-12"
            )
        llm_response = response_agent.choices[0].message.content.strip()
        logger.info(f"LLM Response: {[llm_response]}")
        llm_arm = get_choice_prompt(llm_response)
        chosen_arms['llm'].append(llm_arm)

        # reward
        theta = theta_params[llm_arm - 1]
        llm_reward = np.random.binomial(1, theta)

        # Store LLM's choice and reward
        llm_choices.append((llm_arm, llm_reward))

        for strategy in strategies:
            arm = strategy.select_arm()
            chosen_arms[strategy.__class__.__name__].append(arm+1)
            strategy.update(llm_arm - 1, llm_reward)
        
        update_reward_history(reward_history, llm_arm - 1, llm_reward)

        save_state(state_filename, llm_prompt, chosen_arms, llm_choices, strategies, round_idx+1, reward_history)

        logger.info(f"llm choose: {llm_arm}, reward: {llm_reward}")
        logger.info(f"UCB: {chosen_arms['UCB'][-1]}")
        logger.info(f"TS: {chosen_arms['ThompsonSampling'][-1]}")
        logger.info(f"EGreedy: {chosen_arms['EGreedy'][-1]}")

        print(f"llm choose: {llm_arm}, reward: {llm_reward}")
        print(f"UCB: {chosen_arms['UCB'][-1]}")
        print(f"TS: {chosen_arms['ThompsonSampling'][-1]}")
        print(f"EGreedy: {chosen_arms['EGreedy'][-1]}")

    logger.info(f"==========Game End========")
    print(f"==========Game End========")

    # Save the results of each round to a CSV file
    save_results_to_csv(chosen_arms, llm_choices, csv_filename)
    print("csv file stored")

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
NUM_TURN = 4
num_rounds = 40

# scenario: bandit v.s. hiring
SCENARIO = "bandit"
# model_name: gpt-4o-mini, gpt-4o, o1-preview
MODEL_NAME = "gpt-4o"
# Structure: Memory Chain(MC) v.s. Summary Chain(SC) v.s. Memory Chain+ Summary Chain(MCSC)
STRUCTURE = "MCSC"
# In-context example: Chain of thought(COT) v.s. hint(exploration&exploitation)(HINT) v.s. none(NONE)
IN_CONTEXT_EXAMPLE = "COT"
# Features types: Symbolized() v.s. Insensitive Semantic v.s. Sensitive semantic
FEATURES_TYPES = "Symbolized"
TEST_TYPE = f"{SCENARIO}_{MODEL_NAME}_{STRUCTURE}_{IN_CONTEXT_EXAMPLE}_{FEATURES_TYPES}"


# Define Beta distribution parameters for each arm (alpha, beta)
theta_params = [0.1,0.3,0.5,0.9]

# Configure logging to store LLM responses in a log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f'result_dir/log_{timestamp}_{TEST_TYPE}'
state_dir = f'result_dir/state'
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

    ucb_strategy = UCB(num_arms=4)
    thompson_strategy = ThompsonSampling(num_arms=4)
    egreedy_strategy = EGreedy(num_arms=4, epsilon=0.5)
    strategies = [ucb_strategy, thompson_strategy, egreedy_strategy]

    run_bandit_experiment(turn+1, num_rounds, strategies, theta_params, state_filename, logger, csv_filename)

    logger.info(f'Finished turn {turn + 1}')
    
    logger.removeHandler(file_handler)
    

# def inspect_pickle_file(filename):
#     with open(filename, 'rb') as f:
#         state = pickle.load(f)  # Load the pickled data
#     return state  # Return the loaded state to inspect
# state = inspect_pickle_file('/gbf2/bai-lab/llm-exp/result_dir/log_20240915_205447_single/bandit_state.pkl')
# pprint(state)