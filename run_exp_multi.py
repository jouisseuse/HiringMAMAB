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
import pandas as pd

# Function to initialize multiple OpenAI clients
def initialize_clients_sync():
    """Initialize a list of AsyncOpenAI clients."""
    return OpenAI(
        # api_key=os.environ.get("API_KEY"),
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

def get_regression_features(arm_feature, agent_type_mix):
    # Step 1: Expand the arm features for Type 1 and Type 2
    expanded_arm_features = np.tile(arm_feature, 2)  # Duplicate arm features for both types
    
    # Step 2: Expand the agent type mix to match the dimension of the expanded arm features
    expanded_agent_mix = np.concatenate([np.full(4, agent_type_mix[0]), np.full(4, agent_type_mix[1])])
    
    # Step 3: Element-wise multiply the expanded arm features and expanded agent mix to get an 8-dimensional vector
    regression_features = expanded_arm_features * expanded_agent_mix

    return regression_features

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
        summary += f"{arm}[{feature_str}]: {stats['trials']} times, average reward: {stats['mean_reward']:.3f}, current estimate reward: {all_estimated_rewards[i]:.3f}\n"

        i+=1

    # Global theta summary
    theta_summary = (
        f"Current Estimate Global theta (shared across all candidates, 'unknown' means never tried): "
        f"[Attribute1-CategoryA: {format_theta_value(theta[0])}, Attribute2-CategoryA: {format_theta_value(theta[1])}, "
        f"Attribute3-CategoryA: {format_theta_value(theta[2])}, Attribute4-CategoryA: {format_theta_value(theta[3])}, "
        f"Attribute1-CategoryB : {format_theta_value(theta[4])}, Attribute2-CategoryB : {format_theta_value(theta[5])}, "
        f"Attribute3-CategoryB : {format_theta_value(theta[6])}, Attribute4-CategoryB : {format_theta_value(theta[7])}]"
    )
    
    summary += f"\n{theta_summary}\n"

    return summary

def save_results_to_csv_multi_hiring(llm_choices, estimate_highest_list, filename):
    num_agents = len(llm_choices.keys())
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Dynamically generate header based on the number of agents
        header = ["Round"]
        for i in range(1, num_agents + 1):
            header.append(f"Firm{i}_choice")
            header.append(f"Firm{i}_reward")
            header.append(f"Firm{i}_highest")

        # Write the header to the CSV
        writer.writerow(header)

        # Determine the number of rounds (assuming all agents have the same number of rounds)
        num_rounds = len(list(llm_choices.values())[0])

        # Write each round's data
        for round_idx in range(num_rounds):
            row = [round_idx]  # Start with the round number
            for i in range(1, num_agents + 1):
                agent_choice, agent_reward = llm_choices[f"Firm{i}"][round_idx]
                agent_highest = estimate_highest_list[i-1][round_idx]
                row.extend([agent_choice, agent_reward, agent_highest])  # Add choice, reward, and highest for each agent

            # Write the row to the CSV
            writer.writerow(row)

def run_multi_llm_hiring_experiment_ref(turn_num, num_rounds, theta, arm_features, agents, state_filename, logger, csv_filename):
    num_agents = len(agents)
    # Try to load the state if it exists, otherwise start from scratch
    try:
        llm_prompt, chosen_arms, llm_choices, current_round, reward_history, A_list, b_list, estimate_highest_list = load_state_multi_llm(state_filename)
        logger.info("Resuming from saved state")
    except FileNotFoundError:
        llm_prompt = {f"Firm{i+1}": [] for i in range(num_agents)}
        estimate_highest_list = [[] for _ in range(num_agents)]
        A_list, b_list = initialize_ridge_regression(feature_dim=8)
        chosen_arms = {f"Firm{i+1}": [] for i in range(num_agents)}
        llm_choices = {f"Firm{i+1}": [] for i in range(num_agents)}
        reward_history = {f"Candidate{j+1}": {'trials': 0, 'mean_reward': 0} for j in range(4)}
        current_round = 0

        # initial round
        for agent_idx in range(num_agents):
            firm_name = f"Firm{agent_idx+1}"
            agent_type_mix = agents[agent_idx]
            categoryA_percentage = int(agent_type_mix[0] * 100)
            categoryB_percentage = int(agent_type_mix[1] * 100)
            tmp_prompt = {
                "role": "system", 
                "content": (
                    f"You are one of several firms, tasked with maximizing your firm's rewards by hiring one of four candidates: Candidate1, Candidate2, Candidate3, and Candidate4. "
                    "Each candidate has two attributes. The attributes for each candidate are as follows: " 
                    "Candidate1[Attribute1, Attribute3], Candidate2[Attribute2, Attribute3], Candidate3[Attribute1, Attribute4], Candidate4[Attribute2, Attribute4]. "
                    f"Your firm's occupations are characterized by two categories: [{categoryA_percentage}% CategoryA, {categoryB_percentage}% CategoryB]. "
                    "The reward for each candidate is calculated as the probability that the candidate's attributes align with your firm’s category features, multiplied by an unknown global parameter (theta), with a small random noise. "
                    "The reward ranges from 0 to 1, representing the probability that the candidate is a good fit for your firm. "
                    "Your goal is to maximize your total rewards over a series of rounds by deciding which candidate to hire next, based on feedback from previous rounds. "
                    "At each round, you will receive a summary of the results from not only about your own choices and rewards but also about the choices and rewards of other firms. "
                    "You must decide which candidate (Candidate1, Candidate2, Candidate3, Candidate4) to hire, based on your understanding of the rewards so far. "
                    "Let’s think step by step to make sure we make the best decision. "
                    "Explore all candidates multiple times in the early rounds to minimize the effect of noise and then exploit the best candidates in later rounds. "
                    "After deciding, provide your final answer within the tags <Answer>Candidate</Answer> where Candidate is one of Candidate1, Candidate2, Candidate3, Candidate4."
                )
            }
            llm_prompt[firm_name].append(tmp_prompt)

            llm_arm = agent_idx + 1
            # llm_arm = np.random.randint(1, 5)
            agent_type_mix = agents[agent_idx]
            arm_feature = arm_features[llm_arm - 1]
            llm_reward = calculate_estimated_reward(arm_feature, agent_type_mix, theta)
            feature_vector = get_regression_features(arm_feature, agent_type_mix)
            A_list, b_list = update_ridge_regression(A_list, b_list, feature_vector, llm_reward)

            llm_choices[firm_name].append((llm_arm, llm_reward))
            chosen_arms[firm_name].append(llm_arm)
            estimate_highest_list[agent_idx].append(0)

            # Update the reward history for the candidate
            candidate_name = f"Candidate{llm_arm}"
            trials = reward_history[candidate_name]['trials']
            current_mean = reward_history[candidate_name]['mean_reward']
            new_mean = (current_mean * trials + llm_reward) / (trials + 1)

            reward_history[candidate_name]['trials'] += 1
            reward_history[candidate_name]['mean_reward'] = new_mean


    logger.info(f"==========Game Start========")
    print(f"==========Game Start========")

    # To store CSV output data
    llm_agent = initialize_clients_sync()

    # Loop through each round
    for round_idx in range(current_round, num_rounds):
        logger.info(f"========Round {round_idx+1}======")
        print(f"========Round {round_idx+1}======")

        for agent_idx in range(num_agents):
            agent_type_mix = agents[agent_idx]
            firm_name = f"Firm{agent_idx+1}"

            # Generate the LLM prompt
            prompt1 = {
                "role": "assistant",
                "content": (f"Candidate{llm_choices[firm_name][-1][0]}.") if llm_choices[firm_name] else "Choose a candidate."
            }

            # Estimate theta and rewards
            estimated_theta = estimate_theta(A_list, b_list)
            all_estimated_rewards = []
            for arm_feature in arm_features:
                predicted_reward = calculate_estimated_reward(arm_feature, agent_type_mix, estimated_theta)
                all_estimated_rewards.append(predicted_reward)
            print(f"all_estimated_rewards: {all_estimated_rewards}")

            # Find the highest estimated reward
            estimate_highest = np.argmax(all_estimated_rewards) + 1
            estimate_highest_list[agent_idx].append(estimate_highest)

            # Generate the summary for this agent
            summary_results = generate_summary_contextual_hiring(reward_history, estimated_theta, agent_type_mix, all_estimated_rewards)
            prompt2 = {
                "role": "user",
                "content": (
                    f"So far you have played {round_idx + 1} times with your past choices and rewards summarized as follows: "
                    f"{summary_results}"
                    "Which candidate will you hire next? Remember, YOU MUST provide your final answer within the tags <Answer>Candidate</Answer>."
                )
            }

            llm_prompt[firm_name].append(prompt1)
            llm_prompt[firm_name].append(prompt2)

            # print(llm_prompt[firm_name])


            # LLM selects a candidate
            response_agent = llm_agent.chat.completions.create(
                messages=llm_prompt[firm_name],
                model=MODEL_NAME,
                temperature=TEMPERATURE
            )
            llm_response = response_agent.choices[0].message.content.strip()
            llm_arm = get_choice_prompt_contextual_hiring(llm_response)
            chosen_arms[firm_name].append(llm_arm)
            logger.info(f"{firm_name} Response: {[llm_response]}")

            # Calculate the reward for the chosen candidate
            arm_feature = arm_features[llm_arm - 1]
            llm_reward = calculate_reward_probability_vector(arm_feature, agent_type_mix, theta)

            # Store LLM's choice and reward
            llm_choices[firm_name].append((llm_arm, llm_reward))

        
        for agent_idx in range(num_agents):
            agent_type_mix = agents[agent_idx]
            firm_name = f"Firm{agent_idx+1}"
            llm_arm, llm_reward = llm_choices[firm_name][-1]

            # Update the reward history for the candidate
            candidate_name = f"Candidate{llm_arm}"
            trials = reward_history[candidate_name]['trials']
            current_mean = reward_history[candidate_name]['mean_reward']
            new_mean = (current_mean * trials + llm_reward) / (trials + 1)

            reward_history[candidate_name]['trials'] += 1
            reward_history[candidate_name]['mean_reward'] = new_mean
            arm_feature = arm_features[llm_arm - 1]

            # Update the regression parameters (A, b)
            feature_vector = get_regression_features(arm_feature, agent_type_mix)
            A_list, b_list = update_ridge_regression(A_list, b_list, feature_vector, llm_reward)

            logger.info(f"{firm_name} chose: Candidate{llm_arm}, reward: {llm_reward}")
            print(f"{firm_name} chose: Candidate{llm_arm}, reward: {llm_reward}")


        # Save the state for future resumption
        save_state_multi_llm(state_filename, llm_prompt, chosen_arms, llm_choices, round_idx + 1, reward_history, A_list, b_list, estimate_highest_list)

    logger.info(f"==========Game End========")
    print(f"==========Game End========")

    # Save the results of each round to a CSV file
    save_results_to_csv_multi_hiring(llm_choices, estimate_highest_list, csv_filename)
    print("CSV file stored.")


def calculate_reward_probability_vector(arm_feature, agent_type_mix, theta):
    """
    Calculate the reward based on arm features, agent type, and theta.
    Normalized to 0-1 range, representing the probability.
    """
    theta_type1 = np.array(theta[:4])  # First 4 elements of theta for Type 1
    theta_type2 = np.array(theta[4:])  # Last 4 elements of theta for Type 2
    reward = (
        agent_type_mix[0] * np.dot(arm_feature, theta_type1) +
        agent_type_mix[1] * np.dot(arm_feature, theta_type2)
    ) + np.random.normal(0, 0.05)  # Small noise added

    return np.clip(reward, 0, 1)  # Ensure reward is between 0 and 1


def save_state_multi_llm(state_filename, llm_prompt, chosen_arms, llm_choices, current_round, reward_history, A_list, b_list, estimate_highest_list):
    """Save the current state for multi-agent LLM"""
    state = {
        'llm_prompt': llm_prompt,
        'chosen_arms': chosen_arms,
        'llm_choices': llm_choices,
        'current_round': current_round,
        'reward_history': reward_history,
        'A_list': A_list,
        'b_list': b_list,
        'estimate_highest_list': estimate_highest_list
    }
    with open(state_filename, 'wb') as f:
        pickle.dump(state, f)


def load_state_multi_llm(state_filename):
    """Load the saved state for multi-agent LLM"""
    with open(state_filename, 'rb') as f:
        state = pickle.load(f)
    return (state['llm_prompt'], state['chosen_arms'], state['llm_choices'], state['current_round'],
            state['reward_history'], state['A_list'], state['b_list'], state['estimate_highest_list'])

def initialize_multi_llm_reward_history(num_firms):
    """
    Initialize reward history for each firm, tracking trials and successes per feature for multiple candidates.
    """
    reward_history = {}
    for firm_idx in range(1, num_firms + 1):
        reward_history[f"Firm{firm_idx}"] = {
            'Attribute1': {'trials': 0, 'successes': 0},
            'Attribute2': {'trials': 0, 'successes': 0},
            'Attribute3': {'trials': 0, 'successes': 0},
            'Attribute4': {'trials': 0, 'successes': 0}
        }
    return reward_history

def get_choice_prompt_candidate_hiring(llm_response):
    match = re.search(r'<Answer>Firm([1-4])</Answer>', llm_response)
    if match:
        return int(match.group(1))
    else:
        match = re.search(r'Firm([1-4])', llm_response)
        if match:
            return int(match.group(1))
        else:
            raise ValueError("LLM's response does not contain a valid arm choice.")


def generate_candidate_summary(reward_history, candidate_name, candidate_features):
    """
    Generate a summary of trials and success rates for a specific candidate's features across all firms.
    
    Args:
    - reward_history: A dictionary containing the reward history of each firm.
    - candidate_name: The name of the candidate (e.g., 'Candidate1').
    - candidate_features: The list of attributes for the candidate.
    
    Returns:
    - summary: A formatted string summarizing the trials and success rates for each feature of the candidate across firms.
    """
    # summary = f"Summary for {candidate_name}:\n"
    summary = ""
    
    for firm, attributes in reward_history.items():
        summary += f"{firm}:\n"
        sr_list =[]
        for feature in candidate_features:
            trials = attributes[feature]['trials']
            successes = attributes[feature]['successes']
            if trials > 0:
                success_rate = successes / trials
                summary += f"  {feature}: {trials} trials, success rate: {success_rate:.2f}"
            else:
                success_rate = 0.5
                summary += f"  {feature}: {trials} trials, success rate: unknown"
            sr_list.append(success_rate)
        
        summary += f"  Average success rate: {np.mean(sr_list):.2f}\n"
    
    return summary

def update_multi_llm_reward_history(reward_history, firm_name, candidate_attributes, success_list):
    """
    Update the reward history for a firm based on the candidate's features and the result of the application.
    
    Args:
    - reward_history: The dictionary tracking trials and successes for each firm.
    - firm_name: The name of the firm (e.g., "Firm1").
    - candidate_attributes: The list of attributes (features) for the candidate (e.g., ['Attribute1', 'Attribute3']).
    - success: Whether the candidate was successful (1 for success, 0 for failure).
    """
    i = 0
    for attribute in candidate_attributes:
        reward_history[firm_name][attribute]['trials'] += 1
        reward_history[firm_name][attribute]['successes'] += success_list[i]
        i += 1

    return reward_history

def calculate_success_probability(candidate_attributes, firm_theta):
    """
    Calculate the success probability for a candidate based on the average theta values for their attributes.
    
    Args:
    - candidate_attributes: List of the candidate's attributes (e.g., ['Attribute1', 'Attribute3']).
    - firm_theta: Dictionary containing the theta values for each attribute for a specific firm.
    
    Returns:
    - success_prob: The success probability based on the average of the two theta values for the candidate's attributes.
    """
    theta_values = [firm_theta[attr] for attr in candidate_attributes]
    success_prob = np.mean(theta_values)  # Average of the theta values
    return success_prob

def save_state_multi_candidate(state_filename, llm_prompt, chosen_firms, candidate_choices, current_round, reward_history, estimate_highest_list):
    """Save the current state for multi-agent LLM"""
    state = {
        'llm_prompt': llm_prompt,
        'chosen_firms': chosen_firms,
        'candidate_choices': candidate_choices,
        'current_round': current_round,
        'reward_history': reward_history,
        'estimate_highest_list': estimate_highest_list
    }
    with open(state_filename, 'wb') as f:
        pickle.dump(state, f)


def load_state_multi_candidate(state_filename):
    """Load the saved state for multi-agent LLM"""
    with open(state_filename, 'rb') as f:
        state = pickle.load(f)
    return (state['llm_prompt'], state['chosen_firms'], state['candidate_choices'], state['current_round'], state['reward_history'], state['estimate_highest_list'])


def save_results_to_csv_multi_candidate(llm_choices, estimate_highest_list, filename):
    num_agents = len(llm_choices.keys())
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Dynamically generate header based on the number of agents
        header = ["Round"]
        for i in range(1, num_agents + 1):
            header.append(f"Candidate{i}_choice")
            header.append(f"Candidate{i}_reward")
            header.append(f"Candidate{i}_highest")

        # Write the header to the CSV
        writer.writerow(header)

        # Determine the number of rounds (assuming all agents have the same number of rounds)
        num_rounds = len(list(llm_choices.values())[0])

        # Write each round's data
        for round_idx in range(num_rounds):
            row = [round_idx]  # Start with the round number
            for i in range(1, num_agents + 1):
                agent_choice, agent_reward = llm_choices[f"Candidate{i}"][round_idx]
                agent_highest = estimate_highest_list[i-1][round_idx]
                row.extend([agent_choice, agent_reward, agent_highest])  # Add choice, reward, and highest for each agent

            # Write the row to the CSV
            writer.writerow(row)

def run_multi_candidate_hiring_experiment(turn_num, num_rounds, candidate_features, firm_theta, num_firms, state_filename, logger, csv_filename):
    num_candidates = len(candidate_features)
    
    # Try to load the state if it exists, otherwise start from scratch
    try:
        llm_prompt, chosen_firms, candidate_choices, current_round, reward_history, estimate_highest_list = load_state_multi_candidate(state_filename)
        logger.info("Resuming from saved state")
    except FileNotFoundError:
        llm_prompt = {f"Candidate{i+1}": [] for i in range(num_candidates)}
        estimate_highest_list = [[0] for _ in range(num_candidates)]
        chosen_firms = {f"Candidate{i+1}": [] for i in range(num_candidates)}
        candidate_choices = {f"Candidate{i+1}": [] for i in range(num_candidates)}
        reward_history = initialize_multi_llm_reward_history(num_firms)
        current_round = 0

        # Initialize with first choices
        for candidate_idx in range(num_candidates):
            candidate_name = f"Candidate{candidate_idx+1}"
            candidate_feature = candidate_features[candidate_name]
            tmp_prompt = {
                "role": "system", 
                "content": (
                    "You are one of several candidates tasked with maximizing the number of times you are hired by applying to one of 4 firms: Firm1, Firm2, Firm3, or Firm4. "
                    f"Your attributes are as follows: {candidate_feature}. "
                    "Each firm has an unknown hiring preference based on how well your attributes align with them, and the outcome follows a Bernoulli distribution (either hired or not hired, with unknown probabilities). "
                    "Your goal is to maximize the number of times you are hired over a series of applications by deciding which firm to apply to next, based on feedback provided after each round. "
                    "After each round, you’ll receive feedback on your choices and see the choices of other candidates. "
                    "You should explore all firms multiple times in the early rounds and exploit in the later rounds. "
                    "Let’s think step by step to make sure we make the best decision. "
                    "After deciding, provide your final answer within the tags <Answer>Firm</Answer>."
                )
            }
            llm_prompt[candidate_name].append(tmp_prompt)

            chosen_firm = candidate_idx+1
            chosen_firms[candidate_name].append(chosen_firm)
            candidate_choices[candidate_name].append((chosen_firm, 1))  # Initial choice
            reward_history = update_multi_llm_reward_history(reward_history, f"Firm{chosen_firm}", candidate_feature, [1,1])

    logger.info(f"==========Game Start========")
    print(f"==========Game Start========")

    llm_agent = initialize_clients_sync()

    # Loop through each round
    for round_idx in range(current_round, num_rounds):
        logger.info(f"========Round {round_idx+1}======")
        print(f"========Round {round_idx+1}======")
        tmp_success_list= {f"Candidate{i+1}": [] for i in range(num_candidates)}

        for candidate_idx in range(num_candidates):
            candidate_name = f"Candidate{candidate_idx+1}"
            candidate_feature = candidate_features[candidate_name]

            # Generate the LLM prompt
            prompt1 = {
                "role": "assistant",
                "content": f"Firm{candidate_choices[candidate_name][-1][0]}." if candidate_choices[candidate_name] else "Choose a firm."
            }

            # Estimate theta and success probabilities
            all_success_probabilities = []
            for firm_idx in range(1, num_firms + 1):
                firm_name = f"Firm{firm_idx}"
                all_prob = []
                for feature in candidate_feature:
                    trials = reward_history[firm_name][feature]['trials']
                    successes = reward_history[firm_name][feature]['successes']
                    success_rate = successes / trials if trials > 0 else 0
                    all_prob.append(success_rate)
                average_prob = np.mean(all_prob)
                all_success_probabilities.append(average_prob)

            # Find the firm with the highest estimated success probability
            estimate_highest = np.argmax(all_success_probabilities) + 1
            estimate_highest_list[candidate_idx].append(estimate_highest)

            # Generate the summary for this candidate
            summary_results = generate_candidate_summary(reward_history, candidate_name, candidate_feature)
            prompt2 = {
                "role": "user",
                "content": (
                    f"So far you have played {round_idx + 1} times. Your previous choices and rewards, along with shared information from other candidates, are summarized as follows:\n"
                    f"{summary_results}"
                    "Which firm will you choose next? Remember, YOU MUST provide your final answer within the tags <Answer>Firm</Answer> where Firm is one of Firm1, Firm2, Firm3, Firm4. Let’s think step by step to make the best choice."
                )
            }

            llm_prompt[candidate_name].append(prompt1)
            llm_prompt[candidate_name].append(prompt2)
            # print(f"{llm_prompt[candidate_name]}")

            # LLM selects a firm
            response_agent = llm_agent.chat.completions.create(
                messages=llm_prompt[candidate_name],
                model=MODEL_NAME,
                temperature=TEMPERATURE
            )
            llm_response = response_agent.choices[0].message.content.strip()
            logger.info(f"{candidate_name} Response: {[llm_response]}")
            chosen_firm = get_choice_prompt_candidate_hiring(llm_response)
            chosen_firms[candidate_name].append(chosen_firm)

            # Calculate the success for the chosen firm
            for feature in candidate_feature:
                tmp_success_prob = firm_theta[f"Firm{chosen_firm}"][feature]
                tmp_success = np.random.binomial(1, tmp_success_prob)
                tmp_success_list[candidate_name].append(tmp_success)

            # success_prob = calculate_success_probability(candidate_feature, firm_theta[f"Firm{chosen_firm}"])
            # success = np.random.binomial(1, success_prob)
            success = np.mean(tmp_success_list[candidate_name])

            # Store the choice and reward
            candidate_choices[candidate_name].append((chosen_firm, success))

        for candidate_idx in range(num_candidates):
            # Update the reward history for the firm
            candidate_name = f"Candidate{candidate_idx+1}"
            candidate_feature = candidate_features[candidate_name]
            chosen_firm, success = candidate_choices[candidate_name][-1]
            reward_history = update_multi_llm_reward_history(reward_history, f"Firm{chosen_firm}", candidate_feature, tmp_success_list[candidate_name])

            logger.info(f"{candidate_name} chose: Firm{chosen_firm}, success: {success}")
            print(f"{candidate_name} chose: Firm{chosen_firm}, success: {success}")

        # Save the state for future resumption
        save_state_multi_candidate(state_filename, llm_prompt, chosen_firms, candidate_choices, round_idx + 1, reward_history, estimate_highest_list)

    logger.info(f"==========Game End========")
    print(f"==========Game End========")

    # Save the results of each round to a CSV file
    save_results_to_csv_multi_candidate(candidate_choices, estimate_highest_list, csv_filename)
    print("CSV file stored.")



# Experiment setup
NUM_TURN = 10
num_rounds = 20
agent_num = 4

# scenario: bandit v.s. hiring
SCENARIO = "multibandit-candidate"
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

firm_theta = {
    'Firm1': {'Attribute1': 0.9, 'Attribute2': 0.1, 'Attribute3': 0.9, 'Attribute4': 0.1},
    'Firm2': {'Attribute1': 0.2, 'Attribute2': 0.8, 'Attribute3': 0.8, 'Attribute4': 0.2},
    'Firm3': {'Attribute1': 0.8, 'Attribute2': 0.2, 'Attribute3': 0.2, 'Attribute4': 0.8},
    'Firm4': {'Attribute1': 0.1, 'Attribute2': 0.9, 'Attribute3': 0.1, 'Attribute4': 0.9}
}

candidate_features = {
    'Candidate1': ['Attribute1', 'Attribute3'],  # Candidate1: X1, Y1
    'Candidate2': ['Attribute2', 'Attribute3'],  # Candidate2: X2, Y1
    'Candidate3': ['Attribute1', 'Attribute4'],  # Candidate3: X1, Y2
    'Candidate4': ['Attribute2', 'Attribute4']   # Candidate4: X2, Y2
}

# Configure logging to store LLM responses in a log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f'result_dir/log_{timestamp}_{TEST_TYPE}'
state_dir = f'result_dir/state_multi_hiring'
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

    run_multi_candidate_hiring_experiment(turn+1, num_rounds, candidate_features, firm_theta, 4, state_filename, logger, csv_filename)

    logger.info(f'Finished turn {turn + 1}')
    
    logger.removeHandler(file_handler)
    

# def inspect_pickle_file(filename):
#     with open(filename, 'rb') as f:
#         state = pickle.load(f)  # Load the pickled data
#     return state  # Return the loaded state to inspect
# state = inspect_pickle_file('/gbf2/bai-lab/llm-exp/result_dir/log_20240915_205447_single/bandit_state.pkl')
# pprint(state)