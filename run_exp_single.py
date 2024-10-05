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
from sklearn.linear_model import LinearRegression
from scipy.stats import truncnorm
import random

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

def save_state_contextual(filename, llm_prompt, chosen_arms, llm_choices, current_round, reward_history, X_data, y_data, estimate_highest):
    state = {
        'llm_prompt': llm_prompt,
        'chosen_arms': chosen_arms,
        'llm_choices': llm_choices,
        'current_round': current_round,
        'reward_history':reward_history,
        'X_data': X_data,
        'y_data': y_data,
        'estimate_highest': estimate_highest
    }
    with open(filename, 'wb') as f:
        pickle.dump(state, f)
    # logging.info(f"State saved to {filename}")

# Function to load the state of the game from a file
def load_state_contextual(filename):
    with open(filename, 'rb') as f:
        state = pickle.load(f)
    logging.info(f"State loaded from {filename}")
    return state['llm_prompt'], state['chosen_arms'], state['llm_choices'], state['current_round'], state['reward_history'], state['X_data'], state['y_data'], state['estimate_highest']

# Function to extract the LLM's chosen arm from the response text
# def get_choice_prompt(llm_response):
#     match = re.search(r'A([1-4])', llm_response)
#     if match:
#         return int(match.group(1))
#     else:
#         raise ValueError("LLM's response does not contain a valid arm choice.")

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

def get_choice_prompt_contextual(llm_response):
    match = re.search(r'<Answer>ARM([1-4])</Answer>', llm_response)
    if match:
        return int(match.group(1))
    else:
        match = re.search(r'ARM([1-4])', llm_response)
        if match:
            return int(match.group(1))
        else:
            raise ValueError("LLM's response does not contain a valid arm choice.")

def get_choice_prompt_contextual_hiring(llm_response):
    match = re.search(r'<Answer>Candidate([1-4])</Answer>', llm_response)
    if match:
        return int(match.group(1))
    else:
        match = re.search(r'Candidate([1-4])', llm_response)
        if match:
            return int(match.group(1))
        else:
            raise ValueError("LLM's response does not contain a valid arm choice.")

def get_choice_prompt_hiring(llm_response):
    match = re.search(r'<Answer>F([1-4])</Answer>', llm_response)
    if match:
        return int(match.group(1))
    else:
        match = re.search(r'F([1-4])', llm_response)
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

def calculate_reward_probability_vector(arm_feature, agent_type_mix, theta):
    theta_type1 = np.array(theta[:4])  # First 4 elements of theta for Type 1
    theta_type2 = np.array(theta[4:])  # Last 4 elements of theta for Type 2
    reward = (
        agent_type_mix[0] * np.dot(arm_feature, theta_type1) +  # Contribution from Type 1
        agent_type_mix[1] * np.dot(arm_feature, theta_type2)    # Contribution from Type 2
    ) + np.random.uniform(-0.1,0.1)

    return reward

def calculate_estimated_reward(arm_feature, agent_type_mix, theta):
    # Split theta into Type 1 and Type 2 components
    theta_type1 = np.array(theta[:4])  # First 4 elements for Type 1
    theta_type2 = np.array(theta[4:])  # Last 4 elements for Type 2

    # Calculate the reward based on the dot product of the arm features and theta for each type
    estimated_reward = (
        agent_type_mix[0] * np.dot(arm_feature, theta_type1) +  # Contribution from Type 1
        agent_type_mix[1] * np.dot(arm_feature, theta_type2)    # Contribution from Type 2
    )
    return estimated_reward

def initialize_ridge_regression(feature_dim, lambda_reg=1.0):
    # Initialize A as λ * I (regularization)
    A = np.identity(feature_dim)  # A = λ * I
    b = np.zeros(feature_dim)  # Initialize b as a zero vector
    return A, b

# Update A and b using arm features and reward
def update_ridge_regression(A, b, arm_features, reward):
    # Update A: A = A + arm_features * arm_features.T
    A += np.outer(arm_features, arm_features)
    
    # Update b: b = b + reward * arm_features
    b += reward * arm_features
    return A, b

# Estimate theta using the formula: (A + λI)^(-1) * b
def estimate_theta(A, b):
    # Estimate theta: A^(-1) * b
    theta = np.linalg.inv(A).dot(b)
    return theta

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

def generate_summary_hiring(hiring_history):
    summary = ""
    for firm in hiring_history:
        total_applications = hiring_history[firm]['1'] + hiring_history[firm]['0']
        if total_applications > 0:
            success_rate = hiring_history[firm]['1'] / total_applications
            summary += f"F{firm + 1}: {total_applications} applications, success rate: {success_rate:.2f}\n"
        else:
            summary += f"F{firm + 1}: {total_applications} applications\n"
    return summary

def generate_summary_contextual(reward_history, theta, agent_type_mix, all_estimated_rewards):
    def format_theta_value(value):
        if value == 0:
            return "unknown"
        return f"{value:.2f}"
    
    arm_features_labels = {
        'ARM1': ['Attribute1', 'Attribute3'],  # ARM1: X1, Y1
        'ARM2': ['Attribute2', 'Attribute3'],  # ARM2: X2, Y1
        'ARM3': ['Attribute1', 'Attribute4'],  # ARM3: X1, Y2
        'ARM4': ['Attribute2', 'Attribute4']   # ARM4: X2, Y2
    }
    summary = ""
    i = 0
    for arm, stats in reward_history.items():
        # Include the arm's features in the summary
        feature_str = ", ".join(arm_features_labels[arm])
        summary += f"{arm}[{feature_str}]: {stats['trials']} times, average reward: {stats['mean_reward']}, estimate reward: {all_estimated_rewards[i]}\n"

        i+=1

    # Global theta summary
    theta_summary = (
        f"Estimate Global theta (shared across all arms, 'unknown' means never tried): "
        f"[Attribute1-CategoryA: {format_theta_value(theta[0])}, Attribute2-CategoryA: {format_theta_value(theta[1])}, "
        f"Attribute3-CategoryA: {format_theta_value(theta[2])}, Attribute4-CategoryA: {format_theta_value(theta[3])}, "
        f"Attribute1-CategoryB : {format_theta_value(theta[4])}, Attribute2-CategoryB : {format_theta_value(theta[5])}, "
        f"Attribute3-CategoryB : {format_theta_value(theta[6])}, Attribute4-CategoryB : {format_theta_value(theta[7])}]"
    )
    
    summary += f"\n{theta_summary}\n"
    # theta_summary += f"\n{summary}\n"
    return summary

def generate_summary_contextual_hiring(reward_history, theta, agent_type_mix, all_estimated_rewards):
    def format_theta_value(value):
        if value == 0:
            return "unknown"
        return f"{value:.2f}"
    
    arm_features_labels = {
        'Candidate1': ['Attribute1', 'Attribute3'],  # Candidate1: X1, Y1
        'Candidate2': ['Attribute2', 'Attribute3'],  # Candidate2: X2, Y1
        'Candidate3': ['Attribute1', 'Attribute4'],  # Candidate3: X1, Y2
        'Candidate4': ['Attribute2', 'Attribute4']   # Candidate4: X2, Y2
    }
    summary = ""
    i = 0
    for arm, stats in reward_history.items():
        # Include the arm's features in the summary
        feature_str = ", ".join(arm_features_labels[arm])
        summary += f"{arm}[{feature_str}]: {stats['trials']} times, average reward: {stats['mean_reward']}, estimate reward: {all_estimated_rewards[i]}\n"

        i+=1

    # Global theta summary
    theta_summary = (
        f"Estimate Global theta (shared across all candidates, 'unknown' means never tried): "
        f"[Attribute1-CategoryA: {format_theta_value(theta[0])}, Attribute2-CategoryA: {format_theta_value(theta[1])}, "
        f"Attribute3-CategoryA: {format_theta_value(theta[2])}, Attribute4-CategoryA: {format_theta_value(theta[3])}, "
        f"Attribute1-CategoryB : {format_theta_value(theta[4])}, Attribute2-CategoryB : {format_theta_value(theta[5])}, "
        f"Attribute3-CategoryB : {format_theta_value(theta[6])}, Attribute4-CategoryB : {format_theta_value(theta[7])}]"
    )
    
    summary += f"\n{theta_summary}\n"
    # theta_summary += f"\n{summary}\n"
    return summary


def estimate_theta_via_regression(X_data, y_data):
    model = LinearRegression()
    model.fit(X_data, y_data)
    estimated_theta = model.coef_  # Coefficients represent the estimated theta
    # print(f"X_data:{X_data}")
    # print(f"y_data:{y_data}")
    print(f"Estimated theta: {estimated_theta}")
    return estimated_theta

def get_regression_features(arm_feature, agent_type_mix):
    # Step 1: Expand the arm features for Type 1 and Type 2
    expanded_arm_features = np.tile(arm_feature, 2)  # Duplicate arm features for both types
    
    # Step 2: Expand the agent type mix to match the dimension of the expanded arm features
    expanded_agent_mix = np.concatenate([np.full(4, agent_type_mix[0]), np.full(4, agent_type_mix[1])])
    
    # Step 3: Element-wise multiply the expanded arm features and expanded agent mix to get an 8-dimensional vector
    regression_features = expanded_arm_features * expanded_agent_mix

    return regression_features

def get_estimate_highest(estimate_theta, agent_type_mix):
    all_estimated_rewards = []
    for arm_feature in arm_features:
        predicted_reward = calculate_estimated_reward(arm_feature, agent_type_mix, estimate_theta)
        all_estimated_rewards.append(predicted_reward)
    return np.argmax(all_estimated_rewards) + 1

def save_results_to_csv_contextual(chosen_arms, llm_choices, estimated_highest, filename):
    import csv
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["Round", "LLM Choice", "LLM Reward", "Estimated highest"])

        for round_idx, (llm_arm, llm_reward) in enumerate(llm_choices):
            writer.writerow([round_idx, llm_arm, llm_reward, estimated_highest[round_idx]])

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
            {"role": "system", "content": "You are playing a multi-armed bandit game with 4 arms. Each arm gives either a 0 or 1 reward. Your goal is to get rewards as much as possible by choosing which arm (A1-A4) to play in each round. After each round, you'll get feedback on rewards, helping you make future decisions."}
        ]
        initial_arm = np.random.randint(1, 5)
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
                "Which arm will you choose next? Remember, YOU MUST provide your final answer within the tags <Answer>ARM</Answer> where ARM is one of A1, A2, A3, A4. Let’s think step by step to make the best choice."
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
def run_bandit_experiment_ref(turn_num, num_rounds, strategies,theta_params, state_filename, logger, csv_filename):
    # Try to load the state if it exists, otherwise start from scratch
    try:
        llm_prompt, chosen_arms, llm_choices, strategies, current_round, reward_history = load_state(state_filename)
        logger.info("Resuming from saved state")
    except FileNotFoundError:
        llm_prompt = [
            {"role": "system", "content": "You are a bandit algorithm tasked with maximizing your rewards by playing one of 4 arms labeled A1, A2, A3, and A4. Each arm is associated with a Bernoulli distribution that gives a reward of either 0 or 1, with unknown probabilities. Your goal is to maximize your total reward over a series of rounds by deciding which arm to pull next, based on the feedback provided after each round. At each time step, you will receive a summary of the results from all previous rounds. You must decide which arm (A1, A2, A3, A4) to pull, based on your understanding of the rewards so far. Let’s think step by step to make sure we make the best decision. After deciding, provide your final answer within the tags <Answer>ARM</Answer> where ARM is one of A1, A2, A3, A4."}
        ]
        initial_arm = np.random.randint(1, 5)
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
                f"So far you have played {round_idx+1} times with your past choices and rewards summarized as follows:"
                f"{summary_results}"
                "Which arm will you choose next? Remember, YOU MUST provide your final answer within the tags <Answer>ARM</Answer> where ARM is one of A1, A2, A3, A4. Let’s think step by step to make the best choice."
            )
        }
        llm_prompt.append(prompt1)
        llm_prompt.append(prompt2)
        response_agent = llm_agent.chat.completions.create(
                messages=llm_prompt,
                model=MODEL_NAME,
                temperature=TEMPERATURE
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

# from the side of firm
def run_contextual_experiment_ref(turn_num, num_rounds, theta, arm_features, agents, state_filename, logger, csv_filename):
    agent_type_mix = agents[1]
    # Try to load the state if it exists, otherwise start from scratch
    try:
        llm_prompt, chosen_arms, llm_choices, current_round, reward_history, A, b, estimate_highest = load_state_contextual(state_filename)
        logger.info("Resuming from saved state")
    except FileNotFoundError:
        llm_prompt = [
            {
                "role": "system", 
                "content": (
                    "You are an algorithm playing a linear bandit game. Your task is to maximize your rewards by selecting one of four arms: ARM1, ARM2, ARM3, and ARM4. "
                    "Each arm has two attributes. The attributes for each arm are as follows: " 
                    "ARM1[Attribute1, Attribute3], ARM2[Attribute2, Attribute3], ARM3[Attribute1, Attribute4], ARM4[Attribute2, Attribute4]. "
                    "Your own agent is characterized by two categories: [70% CategoryA, 30% CategoryB]. "
                    "The reward for each arm is calculated by multiplying the arm's attributes, your agent’s category features, and an unknown global parameter (theta) that is shared across all arms. "
                    "Additionally, a small random noise is added to the reward. "
                    "Your objective is to maximize your total rewards over a series of rounds by deciding which arm to select next, based on feedback from previous rounds. "
                    # "Your goal is to maximize your total reward over a series of rounds by deciding which arm to pull next, based on the feedback provided after each round. "
                    "At each time step, you will receive a summary of the results from all previous rounds. "
                    "You must decide which arm (ARM1, ARM2, ARM3, ARM4) to pull, based on your understanding of the rewards so far. "
                    "Let’s think step by step to make sure we make the best decision. "
                    "You should explore all arms multiple times in the initial rounds to minimize the effect of noise and exploit in the later rounds. "
                    "After deciding, provide your final answer within the tags <Answer>ARM</Answer> where ARM is one of ARM1, ARM2, ARM3, ARM4."
                )
            }
        ]
        estimate_highest = []
        estimate_highest.append(0)
        A, b = initialize_ridge_regression(feature_dim)
        initial_arm = np.random.randint(1, 5)
        chosen_arms = {
            "llm": [initial_arm]
        }
        arm_feature = arm_features[initial_arm - 1]
        llm_reward = calculate_estimated_reward(arm_feature, agent_type_mix, theta)
        llm_choices = [(initial_arm, llm_reward)]  # Initial reward for LLM
        current_round = 0
        reward_history = {
            'ARM1': {'trials': 0, 'mean_reward': 0},
            'ARM2': {'trials': 0, 'mean_reward': 0},
            'ARM3': {'trials': 0, 'mean_reward': 0},
            'ARM4': {'trials': 0, 'mean_reward': 0}
        }
        arm_name = f"ARM{initial_arm}"
        trials = reward_history[arm_name]['trials']
        current_mean = reward_history[arm_name]['mean_reward']
        new_mean = (current_mean * trials + llm_reward) / (trials + 1)
        
        reward_history[arm_name]['trials'] += 1
        reward_history[arm_name]['mean_reward'] = new_mean

        feature_vector = get_regression_features(arm_feature, agent_type_mix)
        A, b = update_ridge_regression(A, b, feature_vector, llm_reward)
        logger.info(f"Initial: llm choose: {initial_arm}, reward: {llm_reward}")

        # logger.info("Starting a new experiment")

    llm_agent = initialize_clients_sync()

    logger.info(f"==========Game Start========")
    print(f"==========Game Start========")
    for round_idx in range(current_round,num_rounds):
        logger.info(f"========Round {round_idx+1}======")
        print(f"========Round {round_idx+1}======")
        prompt1 = {
                "role": "assistant",
                "content": (f"ARM{llm_choices[-1][0]}.")
            }
        estimated_theta = estimate_theta(A, b)
        all_estimated_rewards = []
        for arm_feature in arm_features:
            predicted_reward = calculate_estimated_reward(arm_feature, agent_type_mix, estimated_theta)
            all_estimated_rewards.append(predicted_reward)

        estimate_highest.append(np.argmax(all_estimated_rewards)+1)
        summary_results = generate_summary_contextual(reward_history,estimated_theta, agent_type_mix, all_estimated_rewards)
        prompt2 ={
            "role": "user", 
            "content": (
                f"So far you have played {round_idx+1} times with your past choices and rewards summarized as follows: "
                f"{summary_results}"
                "Which arm will you choose next? Remember, YOU MUST provide your final answer within the tags <Answer>ARM</Answer> where ARM is one of ARM1, ARM2, ARM3, ARM4. You should explore all arms multiple times in the initial rounds to minimize the effect of noise and exploit in the later rounds. Let’s think step by step to make the best choice."
            )
        }
        llm_prompt.append(prompt1)
        llm_prompt.append(prompt2)
        # print(llm_prompt)
        response_agent = llm_agent.chat.completions.create(
                messages=llm_prompt,
                model=MODEL_NAME,
                temperature=TEMPERATURE
                # model="o1-preview-2024-09-12"
            )
        llm_response = response_agent.choices[0].message.content.strip()
        logger.info(f"LLM Response: {[llm_response]}")
        llm_arm = get_choice_prompt_contextual(llm_response)
        chosen_arms['llm'].append(llm_arm)

        # reward
        # agent_type_mix = agents[1]
        arm_feature = arm_features[llm_arm - 1]
        llm_reward = calculate_reward_probability_vector(arm_feature, agent_type_mix, theta)

        # Store LLM's choice and reward
        llm_choices.append((llm_arm, llm_reward))

        arm_name = f"ARM{llm_arm}"
        trials = reward_history[arm_name]['trials']
        current_mean = reward_history[arm_name]['mean_reward']
        new_mean = (current_mean * trials + llm_reward) / (trials + 1)
        
        reward_history[arm_name]['trials'] += 1
        reward_history[arm_name]['mean_reward'] = new_mean

        feature_vector = get_regression_features(arm_feature, agent_type_mix)
        A, b = update_ridge_regression(A, b, feature_vector, llm_reward)

        save_state_contextual(state_filename, llm_prompt, chosen_arms, llm_choices, round_idx+1, reward_history, A, b, estimate_highest)

        logger.info(f"llm choose: {llm_arm}, reward: {llm_reward}")

        print(f"llm choose: {llm_arm}, reward: {llm_reward}")

    logger.info(f"==========Game End========")
    print(f"==========Game End========")

    # Save the results of each round to a CSV file
    save_results_to_csv_contextual(chosen_arms, llm_choices, estimate_highest, csv_filename)
    print("csv file stored")

def run_hiring_experiment_ref(turn_num, num_rounds, strategies,theta_params, state_filename, logger, csv_filename):
    # Try to load the state if it exists, otherwise start from scratch
    try:
        llm_prompt, chosen_arms, llm_choices, strategies, current_round, reward_history = load_state(state_filename)
        logger.info("Resuming from saved state")
    except FileNotFoundError:
        # llm_prompt = [
        #     {"role": "system", "content": "You are a job candidate tasked with maximizing the number of times you are hired by applying to one of 4 firms labeled F1, F2, F3, and F4. Each firm is associated with an unknown hiring preference, and the outcome of your application follows a Bernoulli distribution (either hired or not hired, with unknown probabilities). Your goal is to maximize the number of times you are hired over a series of applications by deciding which firm to apply to next, based on feedback provided after each round. At each time step, you will receive a summary of the results from all previous rounds. You must decide which firm (F1, F2, F3, F4) to apply to, based on your understanding of the firms' hiring patterns so far. Let’s think step by step to make sure we make the best decision. After deciding, provide your final answer within the tags <Answer>FIRM</Answer> where FIRM is one of F1, F2, F3, F4."}
        # ]
        llm_prompt = [
            {"role": "system", "content": "You are a job candidate tasked with maximizing the number of times you are hired by applying to one of 4 firms labeled F1, F2, F3, and F4. Each firm has an unknown hiring preference, and the result of your application is binary (either hired or not hired). Your goal is to maximize the number of times you are hired over a series of applications by deciding which firm to apply to next, based on the feedback provided after each round. At each time step, you will receive a summary of the results from all previous rounds. You must decide which firm (F1, F2, F3, F4) to apply to, based on your understanding of the firms' hiring patterns so far. Let’s think step by step to make sure we make the best decision. After deciding, provide your final answer within the tags <Answer>FIRM</Answer> where FIRM is one of F1, F2, F3, F4."}
        ]
        initial_arm = np.random.randint(1, 5)
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
            0: {'1': 0, '0': 0},  # F1
            1: {'1': 0, '0': 0},  # F2
            2: {'1': 0, '0': 0},  # F3
            3: {'1': 0, '0': 0}   # F4
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
                "content": (f"F{llm_choices[-1][0]}.")
            }
        summary_results = generate_summary_hiring(reward_history)
        prompt2 = {
            "role": "user", 
            "content": (
                f"So far, you have applied {round_idx+1} times with your past applications and results summarized as follows:"
                f"{summary_results}"
                "Which firm will you apply to next? Remember, YOU MUST provide your final answer within the tags <Answer>FIRM</Answer> where FIRM is one of F1, F2, F3, F4. Let’s think step by step to make the best choice."
            )
        }
        llm_prompt.append(prompt1)
        llm_prompt.append(prompt2)
        # print(llm_prompt)
        response_agent = llm_agent.chat.completions.create(
                messages=llm_prompt,
                model=MODEL_NAME,
                temperature=TEMPERATURE
                # model="o1-preview-2024-09-12"
            )
        llm_response = response_agent.choices[0].message.content.strip()
        logger.info(f"LLM Response: {[llm_response]}")
        llm_arm = get_choice_prompt_hiring(llm_response)
        chosen_arms['llm'].append(llm_arm)
        print(llm_prompt)

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

def run_contextual_hiring_experiment_ref(turn_num, num_rounds, theta, arm_features, agents, state_filename, logger, csv_filename):
    agent_type_mix = agents[1]
    # Try to load the state if it exists, otherwise start from scratch
    try:
        llm_prompt, chosen_arms, llm_choices, current_round, reward_history, A, b, estimate_highest = load_state_contextual(state_filename)
        logger.info("Resuming from saved state")
    except FileNotFoundError:
        llm_prompt = [
            {
                "role": "system", 
                "content": (
                    "You are a firm employer tasked with maximizing your firm's rewards by hiring one of four candidates: Candidate1, Candidate2, Candidate3, and Candidate4. "
                    "Each candidate has two attributes. The attributes for each candidate are as follows: " 
                    "Candidate1[Attribute1, Attribute3], Candidate2[Attribute2, Attribute3], Candidate3[Attribute1, Attribute4], Candidate4[Attribute2, Attribute4]. "
                    "Your firm occupations are characterized by two categories: [70% CategoryA, 30% CategoryB]. "
                    "The reward for each candidate is calculated as the probability that the candidate's attributes align with your firm’s category features, multiplied by an unknown global parameter (theta), with a small random noise. "
                    "The reward ranges from 0 to 1, representing the probability that the candidate is a good fit for your firm. "
                    "Your goal is to maximize your total rewards over a series of rounds by deciding which candidate to hire next, based on feedback from previous rounds. "
                    "At each round, you will receive a summary of the results from all previous rounds. "
                    "You must decide which candidate (Candidate1, Candidate2, Candidate3, Candidate4) to hire next. "
                    "Let’s think step by step to make sure we make the best decision. "
                    "You should explore all candidates multiple times in the early rounds to minimize the effect of noise and exploit in the later rounds. "
                    "After deciding, provide your final answer within the tags <Answer>Candidate</Answer> where Candidate is one of Candidate1, Candidate2, Candidate3, Candidate4."
                )
            }
        ]
        estimate_highest = []
        estimate_highest.append(0)
        A, b = initialize_ridge_regression(feature_dim)
        initial_arm = np.random.randint(1, 5)
        chosen_arms = {
            "llm": [initial_arm]
        }
        arm_feature = arm_features[initial_arm - 1]
        llm_reward = calculate_estimated_reward(arm_feature, agent_type_mix, theta)
        llm_choices = [(initial_arm, llm_reward)]  # Initial reward for LLM
        current_round = 0
        reward_history = {
            'Candidate1': {'trials': 0, 'mean_reward': 0},
            'Candidate2': {'trials': 0, 'mean_reward': 0},
            'Candidate3': {'trials': 0, 'mean_reward': 0},
            'Candidate4': {'trials': 0, 'mean_reward': 0}
        }
        arm_name = f"Candidate{initial_arm}"
        trials = reward_history[arm_name]['trials']
        current_mean = reward_history[arm_name]['mean_reward']
        new_mean = (current_mean * trials + llm_reward) / (trials + 1)
        
        reward_history[arm_name]['trials'] += 1
        reward_history[arm_name]['mean_reward'] = new_mean

        feature_vector = get_regression_features(arm_feature, agent_type_mix)
        A, b = update_ridge_regression(A, b, feature_vector, llm_reward)
        logger.info(f"Initial: llm choose: {initial_arm}, reward: {llm_reward}")


    llm_agent = initialize_clients_sync()

    logger.info(f"==========Game Start========")
    print(f"==========Game Start========")
    for round_idx in range(current_round,num_rounds):
        logger.info(f"========Round {round_idx+1}======")
        print(f"========Round {round_idx+1}======")
        prompt1 = {
                "role": "assistant",
                "content": (f"Candidate{llm_choices[-1][0]}.")
            }
        estimated_theta = estimate_theta(A, b)
        all_estimated_rewards = []
        for arm_feature in arm_features:
            predicted_reward = calculate_estimated_reward(arm_feature, agent_type_mix, estimated_theta)
            all_estimated_rewards.append(predicted_reward)

        estimate_highest.append(np.argmax(all_estimated_rewards)+1)
        summary_results = generate_summary_contextual_hiring(reward_history,estimated_theta, agent_type_mix, all_estimated_rewards)
        prompt2 ={
            "role": "user", 
            "content": (
                f"So far you have played {round_idx+1} times with your past choices and rewards summarized as follows: "
                f"{summary_results}"
                "Which candidate will you hire next? Remember, YOU MUST provide your final answer within the tags <Answer>Candidate</Answer> where Candidate is one of Candidate1, Candidate2, Candidate3, Candidate4. Let’s think step by step to make the best choice. You should explore all candidates multiple times in the early rounds to minimize the effect of noise and exploit in the later rounds. "
            )
        }
        llm_prompt.append(prompt1)
        llm_prompt.append(prompt2)
        print(llm_prompt)
        response_agent = llm_agent.chat.completions.create(
                messages=llm_prompt,
                model=MODEL_NAME,
                temperature=TEMPERATURE
                # model="o1-preview-2024-09-12"
            )
        llm_response = response_agent.choices[0].message.content.strip()
        logger.info(f"LLM Response: {[llm_response]}")
        llm_arm = get_choice_prompt_contextual_hiring(llm_response)
        chosen_arms['llm'].append(llm_arm)

        # reward
        # agent_type_mix = agents[1]
        arm_feature = arm_features[llm_arm - 1]
        llm_reward = calculate_reward_probability_vector(arm_feature, agent_type_mix, theta)

        # Store LLM's choice and reward
        llm_choices.append((llm_arm, llm_reward))

        arm_name = f"Candidate{llm_arm}"
        trials = reward_history[arm_name]['trials']
        current_mean = reward_history[arm_name]['mean_reward']
        new_mean = (current_mean * trials + llm_reward) / (trials + 1)
        
        reward_history[arm_name]['trials'] += 1
        reward_history[arm_name]['mean_reward'] = new_mean

        feature_vector = get_regression_features(arm_feature, agent_type_mix)
        A, b = update_ridge_regression(A, b, feature_vector, llm_reward)

        save_state_contextual(state_filename, llm_prompt, chosen_arms, llm_choices, round_idx+1, reward_history, A, b, estimate_highest)

        logger.info(f"llm choose: {llm_arm}, reward: {llm_reward}")

        print(f"llm choose: {llm_arm}, reward: {llm_reward}")

    logger.info(f"==========Game End========")
    print(f"==========Game End========")

    # Save the results of each round to a CSV file
    save_results_to_csv_contextual(chosen_arms, llm_choices, estimate_highest, csv_filename)
    print("csv file stored")

# Experiment setup
NUM_TURN = 10
num_rounds = 20

# scenario: bandit v.s. hiring v.s. Linbandit v.s. Linbandit-hiring
SCENARIO = "Linbandit-hiring"
# model_name: gpt-4o-mini, gpt-4o, o1-preview
MODEL_NAME = "gpt-4o"
# Structure: Memory Chain(MC) v.s. Summary Chain(SC) v.s. Memory Chain+ Summary Chain(MCSC)
STRUCTURE = "MCSC"
# In-context example: Chain of thought(COT) v.s. hint(exploration&exploitation)(HINT) v.s. none(NONE)
IN_CONTEXT_EXAMPLE = "COT"
# Features types: Symbolized() v.s. Insensitive Semantic v.s. Sensitive semantic
FEATURES_TYPES = "Symbolized"
TEMPERATURE = 0
# reward_type: identical v.s. different
REWARD_TYPE = "different"
# TEST_TYPE = f"{SCENARIO}_{MODEL_NAME}_{STRUCTURE}_{IN_CONTEXT_EXAMPLE}_{FEATURES_TYPES}"
TEST_TYPE = f"{SCENARIO}_{FEATURES_TYPES}_{MODEL_NAME}_{TEMPERATURE}_REF"

# Define Beta distribution parameters for each arm (alpha, beta)
theta_params = [0.1,0.3,0.5,0.9]

theta = [0.5, 0.1, 0.5, 0.1, # Type 1: [theta_A1_Type1, theta_A2_Type1, theta_B1_Type1, theta_B2_Type1]
         0.1, 0.5, 0.1, 0.5]   # Type 2: [theta_A1_Type2, theta_A2_Type2, theta_B1_Type2, theta_B2_Type2]
# Agent type mixes
agents = [
    [0.90, 0.10],  # Agent 1
    [0.70, 0.30],  # Agent 2
    [0.30, 0.70],  # Agent 3
    [0.10, 0.90]   # Agent 4
]
# Arm feature vectors (4-dimensional one-hot encoded)
arm_features = [
    [1, 0, 1, 0],  # Arm 1: X1, Y1
    [0, 1, 1, 0],  # Arm 2: X2, Y1
    [1, 0, 0, 1],  # Arm 3: X1, Y2
    [0, 1, 0, 1]   # Arm 4: X2, Y2
]

feature_dim = 8

# Configure logging to store LLM responses in a log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f'result_dir/log_{timestamp}_{TEST_TYPE}'
state_dir = f'result_dir/state_linbandit'
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
    print(f'Starting turn {turn + 1}')

    # ucb_strategy = UCB(num_arms=4)
    # thompson_strategy = ThompsonSampling(num_arms=4)
    # egreedy_strategy = EGreedy(num_arms=4, epsilon=0.5)
    # strategies = [ucb_strategy, thompson_strategy, egreedy_strategy]
    # run_hiring_experiment_ref(turn+1, num_rounds, strategies, theta_params, state_filename, logger, csv_filename)

    run_contextual_hiring_experiment_ref(turn+1, num_rounds, theta, arm_features, agents, state_filename, logger, csv_filename)

    logger.info(f'Finished turn {turn + 1}')
    print(f'Finished turn {turn + 1}')
    
    logger.removeHandler(file_handler)
    

# def inspect_pickle_file(filename):
#     with open(filename, 'rb') as f:
#         state = pickle.load(f)  # Load the pickled data
#     return state  # Return the loaded state to inspect
# state = inspect_pickle_file('/gbf2/bai-lab/llm-exp/result_dir/log_20240915_205447_single/bandit_state.pkl')
# pprint(state)