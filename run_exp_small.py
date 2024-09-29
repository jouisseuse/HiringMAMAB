import os
import sys
import asyncio
import json
import numpy as np
import itertools
from openai import OpenAI, AsyncOpenAI, DefaultHttpxClient, DefaultAsyncHttpxClient
import logging
from datetime import datetime
import ast
from collections import Counter
from scipy.stats import entropy
import httpx
import re
import time
import csv

NUM_ROUND = 40
NUM_TURN = 3
TIMEOUT = 180

'''
code for generating features
'''
def load_feature_distributions(file_path):
    """Load feature distribution template from a JSON file."""
    with open(file_path) as f:
        return json.load(f)

def generate_all_feature_combinations(features):
    """Generate all possible combinations of feature values."""
    all_combinations = []
    for feature, options in features.items():
        all_combinations.append([(feature, option) for option in options])
    # Create Cartesian product of all feature options
    return [dict(combination) for combination in itertools.product(*all_combinations)]

def encode_one_hot(features, feature_template):
    """Encode features as a one-hot vector based on the template."""
    one_hot_vector = []
    for feature, options in feature_template.items():
        for option in options:
            one_hot_vector.append(1 if option == features[feature] else 0)
    return np.array([one_hot_vector])

def set_up_candidates(candidate_feature_template):
    """Generate and encode all candidate feature combinations."""
    candidate_features = generate_all_feature_combinations(candidate_feature_template)
    encoded_candidates = []
    for feature in candidate_features:
        feature_vector = encode_one_hot(feature, candidate_feature_template)
        encoded_candidates.append({"features": feature, "encoded_features": feature_vector})
        print(f"[debug] candidate {feature_vector}")
    return encoded_candidates

def load_firm_vectors(file_path):
    """Load firm vectors from a text file, removing any non-numeric characters."""
    with open(file_path) as f:
        list_from_file = [ast.literal_eval(line.replace(' ', ', ')) for line in f]
    vectors = []
    for vector in list_from_file:
        vectors.append(np.array([vector]))
        print(f"[debug] firm {vector}")
    return vectors

def generate_firm_descriptions(vectors, firm_types):
    """Generate descriptions for firms based on their vectors."""
    descriptions = []
    for i, vector in enumerate(vectors):
        # description = f"F{i+1}: "
        description = [f"{firm_types[j]} ({vector[0][j]*100:.0f}%)" for j in range(len(firm_types))]
        descriptions.append(description)
    return descriptions

'''
code for LLM initialization
'''
# Function to initialize multiple AsyncOpenAI clients
def initialize_clients(num_clients):
    """Initialize a list of AsyncOpenAI clients."""
    return [AsyncOpenAI(
        # api_key=os.environ.get("API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        http_client=DefaultAsyncHttpxClient(
            proxies="http://127.0.0.1:7890",
            transport=httpx.HTTPTransport(local_address="0.0.0.0"),
        ),
    ) for _ in range(num_clients)]

# Function to initialize multiple OpenAI clients
def initialize_clients_sync(num_clients):
    """Initialize a list of AsyncOpenAI clients."""
    return [OpenAI(
        # api_key=os.environ.get("API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        http_client=DefaultHttpxClient(
            proxies="http://127.0.0.1:7890",
            transport=httpx.HTTPTransport(local_address="0.0.0.0"),
        ),
    ) for _ in range(num_clients)]

# Function to generate system prompts for initial role setting
def generate_initial_prompts_decentralized(candidates, firm_descriptions):
    candidate_system_prompts = []
    firm_system_prompts = []
    
    for i,candidate in enumerate(candidates):
        # candidate_description = (
        #     f"You are a candidate with [{candidate['features']['category#1']}, "
        #     f"{candidate['features']['category#2']}, {candidate['features']['category#3']}]. "
        #     # "This scenario is a Multi-Armed Bandit (MAB) problem where 
        #     "Your goal is to maximize your long-term success by selecting the firms to apply to over many rounds. "
        #     "Each round, pick a firm (F1-F4) to apply to and get feedback (hired/not). "
        #     "Hiring depends on how your features match the firm's (beta distribution). "
        #     "Balance exploration (trying new firms) and exploitation (applying to firms where you have previously succeeded). "
        #     "Use the previous round's results to choose the firm ID (e.g., F1)."
        # )
        # candidate_description = (
            # f"You are an intelligent candidate with [{candidate['features']['category#1']}, "
            # f"{candidate['features']['category#2']} "
        #     # "This scenario is a Multi-Armed Bandit (MAB) problem where 
        #     "Your goal is to maximize your *long-term* success by selecting the firms to apply to over many rounds. "
        #     "Hiring depends on how your features match the firm's (beta distribution). "
        #     "Each round, pick a firm (F1-F4) to apply to and get feedback (hired/not) to update the estimate. "
        #     # "Balance exploration (trying new firms) and exploitation (applying to firms where you have previously succeeded). "
        #     "Use the previous round's results(current estimate) to choose the firm ID (e.g., F1)."
        # )

        ### distributed
        # candidate_description = (
        #     f"You are a job candidate with [{candidate['features']['category#1']}, "
        #     f"{candidate['features']['category#2']}] in a hiring game. "
        #     "Your goal is to get hired as often as possible by choosing which firm (F1-F4) to apply to each round. "
        #     "After each round, you'll get feedback on whether you were hired, helping you make future decisions. "
        # )

        ### decentralized
        other_candidate = ""
        for j,cand in enumerate(candidates):
            if i != j:
                other_candidate += f"C{j+1}: [{cand['features']['category#1']}, {cand['features']['category#2']}]. "

        candidate_description = (
            f"You are a job candidate C{i+1} with [{candidate['features']['category#1']}, "
            f"{candidate['features']['category#2']}] in a hiring game. "
            f"Other candidates have the following features: {other_candidate}"
            "Your goal is to get hired as often as possible by choosing which firm (F1-F4) to apply to each round. "
            "After each round, you'll get feedback on all candidates hiring results, helping you make future decisions. "
        )

        ####
        print(f"[debug] candidate_description: {candidate_description}")
        candidate_system_prompts.append({"role": "system", "content": candidate_description})
    
    for i,firm_desc in enumerate(firm_descriptions):
        ### distributed
        # firm_description = (
        #     f"You are a firm with: {firm_desc}. "
        #     # "This scenario is a Multi-Armed Bandit (MAB) problem where 
        #     "Your goal is to maximize long-term performance by selecting the candidates over many rounds. "
        #     "Each round, you must select one candidate to hire based on their features and past performance. "
        #     "Performance is the sum of efforts between your features and the candidate's, with effort coefficients following a normal distribution. "
        #     "Balance exploration (trying new candidates) and exploitation (selecting candidates with proven performance). "
        #     "Use the previous round’s results and the current candidate pool to choose the optimal candidate ID (e.g., C1)."
        # )
        # firm_description = (
        #     f"You are an intelligent HR of firm with: {firm_desc}. "
        #     # "This scenario is a Multi-Armed Bandit (MAB) problem where 
        #     "Your goal is to maximize *long-term* performance by selecting the candidates over many rounds. "
        #     "Performance is the sum of efforts between your features and the candidate's, with effort coefficients following a normal distribution. "
        #     "Each round, you must select one candidate to hire based on their features and past performance and get feedback to update the estimate. "
        #     # "Balance exploration (trying new candidates) and exploitation (selecting candidates with proven performance). "
        #     "Use the previous round’s results(current estimate) and the current candidate pool to choose the optimal candidate ID (e.g., C1)."
        # )

        # firm_description = (
        #     f"You are an HR representative in a hiring game. Your firm is: {firm_desc}. "
        #     "Your goal is to improve your firm's performance by selecting the best candidate each round. "
        #     "After each round, you'll get feedback on the hired candidate's performance. "
        # )

        ### distributed
        other_firm = ""
        for j,f_desc in enumerate(firm_descriptions):
            if i != j:
                other_firm += f"F{j+1}: {f_desc}. "

        firm_description = (
            f"You are an HR representative in a hiring game. Your firm F{i+1} is: {firm_desc}. "
            f"Other firm has the following features: {other_firm}"
            # f"Candiates features: C1:[feature#1, feature#3]. C2:[feature#1, feature#4]. C3:[feature#2, feature#3]. C4:[feature#2, feature#4]."
            "Your goal is to improve your firm's performance by selecting the best candidate each round. "
            "After each round, you'll get feedback on the hired candidate's performance of all firms. "
        )

        firm_system_prompts.append({"role": "system", "content": firm_description})
    
    return candidate_system_prompts, firm_system_prompts

def generate_initial_prompts_distributed(candidates, firm_descriptions):
    candidate_system_prompts = []
    firm_system_prompts = []
    
    for i,candidate in enumerate(candidates):
        ### distributed
        candidate_description = (
            f"You are a job candidate with [{candidate['features']['category#1']}, "
            f"{candidate['features']['category#2']}] in a hiring game. "
            "Your goal is to get hired as often as possible by choosing which firm (F1-F4) to apply to each round. "
            "After each round, you'll get feedback on whether you were hired, helping you make future decisions. "
        )

        print(f"[debug] candidate_description: {candidate_description}")
        candidate_system_prompts.append({"role": "system", "content": candidate_description})
    
    for i,firm_desc in enumerate(firm_descriptions):
        ### distributed
        firm_description = (
            f"You are an HR representative in a hiring game. Your firm is: {firm_desc}. "
            "Your goal is to improve your firm's performance by selecting the best candidate each round. "
            "After each round, you'll get feedback on the hired candidate's performance. "
        )

        firm_system_prompts.append({"role": "system", "content": firm_description})
    
    return candidate_system_prompts, firm_system_prompts

# Function to update the theta for one firm
def update_firm(firm_vector, candidate_vector, reward, A, b):
    print(f"[debug] {firm_vector}, {candidate_vector}")
    features = (candidate_vector * firm_vector.T).reshape(-1)
    A += np.outer(features, features)
    b += reward * features
    theta = np.linalg.inv(A).dot(b)

    return theta, A, b

# Function to update the beta for one candidate
def update_candidate(firm_id, candidate_vector, reward, a, b):
    for i in range(4):
        if candidate_vector[0][i] > 0:
            if reward == 1:
                a[firm_id][i] += 1
            else:
                b[firm_id][i] += 1
    
    return a,b

# Function to generate candidate observation
def candidate_observation_distributed(a, b, candidate_vector):
    observed_results = ""

    for j in range(4):  # Outer loop for firms
        firm_result = f"F{j+1}: "
        grouped_features = {}

        for i in range(4):  # Inner loop for features
            if candidate_vector[0][i] > 0:  # Check if the feature is relevant
                success = a[j][i]
                failure = b[j][i]
                key = (success, failure)
                
                # Group features by their (success, failure) tuple
                if key not in grouped_features:
                    grouped_features[key] = []
                grouped_features[key].append(f"feature#{i+1}")

        feature_groups = []
        for (success, failure), features in grouped_features.items():
            feature_group = f"{', '.join(features)}: (success: {success}, failure: {failure})"
            feature_groups.append(feature_group)
        
        if feature_groups:
            firm_result += ", ".join(feature_groups)
            observed_results += firm_result + "\n"
    
    return observed_results

# Function to generate firm observation
def firm_observation_distributed(theta, A):
    observed_results = ""

    for k in range(len(theta)):
        i = k // 4
        j = k % 4
        firm_result = f"type#{j+1}_feature#{i+1}:(mean: {theta[k]}, std: {np.linalg.inv(A)[k,k]})"
        observed_results += firm_result + "\n"
    return observed_results


# Function to generate candidate summary string
def generate_candidate_summary(hire_summary_candidate, candidate_vectors, candidate_id=None):
    if candidate_id:  # If a candidate_id is provided, only output the summary for that candidate
        summary = hire_summary_candidate.get(candidate_id)
        if not summary:
            return f"C{candidate_id} not found."
        candidate_summary_str = f"C{summary['id']}:\n"
        for firm_id, stats in summary['firms'].items():
            candidate_summary_str += f"  F{firm_id} - Hired: {stats['hire_count']} times, Not Hired: {stats['not_hire_count']} times\n"
        return candidate_summary_str
    else:  # If no candidate_id is provided, output the summary for all candidates
        candidate_summary_str = "Candidate Hire Summary:\n"
        for candidate_id, summary in hire_summary_candidate.items():
            candidate_summary_str += f"C{summary['id']} [{candidate_vectors[candidate_id-1]['features']['category#1']},{candidate_vectors[candidate_id-1]['features']['category#2']}]:\n"
            for firm_id, stats in summary['firms'].items():
                candidate_summary_str += f"  F{firm_id} - Hired: {stats['hire_count']} times, Not Hired: {stats['not_hire_count']} times\n"
        return candidate_summary_str

# Function to generate firm summary string
def generate_firm_summary(hire_summary_firm, candidate_vectors, firm_descriptions, firm_id=None):
    if firm_id:  # If a firm_id is provided, only output the summary for that firm
        summary = hire_summary_firm.get(firm_id)
        if not summary:
            return f"F{firm_id} not found."
        firm_summary_str = f"F{summary['id']}:\n"
        for candidate_id, stats in summary['candidates'].items():
            firm_summary_str += f"  C{candidate_id} [{candidate_vectors[candidate_id-1]['features']['category#1']},{candidate_vectors[candidate_id-1]['features']['category#2']}] - Rewards: {stats['rewards']}\n"
        return firm_summary_str
    else:  # If no firm_id is provided, output the summary for all firms
        firm_summary_str = "\nFirm Rewards Summary:\n"
        for firm_id, summary in hire_summary_firm.items():
            firm_summary_str += f"F{summary['id']} {firm_descriptions[firm_id-1]}:\n"
            for candidate_id, stats in summary['candidates'].items():
                firm_summary_str += f"  C{candidate_id} [{candidate_vectors[candidate_id-1]['features']['category#1']},{candidate_vectors[candidate_id-1]['features']['category#2']}] - Rewards: {stats['rewards']}\n"
        return firm_summary_str


'''
code for other functions
'''
def matrix_print(matrix):
    if isinstance(matrix, np.float64):
        matrix = [0] * 4
    formatted_matrix =  np.array([f"{num:2d}" for num in matrix]) 
    return formatted_matrix

def matrix_print_2d(matrix):
    if isinstance(matrix, np.float64):
        matrix.append([0] * 4)
    formatted_matrix =  np.array([f"{num:2d}" for num in matrix[0]]) 
    
    return formatted_matrix 

# Calculate the entropy of each category
def cal_entropy(features):
    if len(features) < 2 :
        return 0.0
    data = np.array(features)
    gender = data[:, :2]
    race = data[:, 2:4]

    # 计算性别的熵
    gender_probs = np.mean(gender, axis=0)
    gender_entropy = entropy(gender_probs, base=2)

    # 计算人种的熵
    race_probs = np.mean(race, axis=0)
    race_entropy = entropy(race_probs, base=2)

    # 总的熵
    total_entropy = gender_entropy + race_entropy

    return total_entropy

# Calculate the entropy of a given set of vectors
def calculate_entropy(vector_set):
     # 将 numpy 数组转换为元组，因为元组是可哈希的
    tuple_vector_set = [tuple(vector) for vector in vector_set]
    
    # 计算每个向量出现的次数
    vector_counts = Counter(tuple_vector_set)
    
    # 总向量数
    total_vectors = len(vector_set)
    
    # 计算每个向量的概率
    probabilities = np.array(list(vector_counts.values())) / total_vectors
    
    # 使用 scipy 的 entropy 函数计算熵
    entropy_value = entropy(probabilities, base=2)  # base=2 表示以2为底的对数
    
    return entropy_value

def extract_scores_from_logprobs(choice_logprobs):
    # 初始化一个字典，用于存储1, 2, 3, 4的分数
    scores = {str(i): 0 for i in range(1, 5)}

    # 遍历ChoiceLogprobs内容
    for token_logprob in choice_logprobs.content[1].top_logprobs:
        token = token_logprob.token
        
        # 只处理 '1', '2', '3', '4' 的token
        if token in scores:
            scores[token] = np.exp(token_logprob.logprob)
            print(f"[debug] {token} {scores[token]}")

    return scores

def score_to_grade(score):
    if score > 4.6745:
        return 'A'
    elif score > 4:
        return 'B'
    elif score > 3.3255:
        return 'C'
    else:
        return 'D'

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


'''
code for iterative loop experiment (distributed & decentralized)
'''
# Main experiment loop
# def run_experiment_distributed_grade(candidate_file, firm_vector_file, logger, turn, log_dir):
#     ### feature generation
#     # Firm types
#     # firm_types = ["technology", "management", "service", "production"]
#     firm_types = ["type#1", "type#2"]
#     # Load firm vectors
#     firm_vectors = load_firm_vectors(firm_vector_file)
#     # Generate descriptions
#     firm_descriptions = generate_firm_descriptions(firm_vectors, firm_types)

#     # Load feature template and generate candidates
#     candidate_feature_template = load_feature_distributions(candidate_file)
#     candidate_vectors = set_up_candidates(candidate_feature_template)

#     ### initialization
#     # Initialize OpenAI clients
#     print(f"[DEBUG] Initialize OpenAI clients")
#     candidate_clients = initialize_clients_sync(4)
#     firm_clients = initialize_clients_sync(4)

#     # Initialize result tracking variables
#     # Define file paths for saving and restoring states
#     state_file = f"result_dir/state/game_state_distributed_{turn}.json"
#     # Default values for initialization variables
#     def initialize_variables():
#         candidate_system_prompts, firm_system_prompts = generate_initial_prompts_distributed(candidate_vectors, firm_descriptions)
        
#         all_application_features = {firm: [] for firm in range(4)}
#         all_offer_features = {firm: [] for firm in range(4)}
#         candidate_results = {
#             1: [1,1],
#             2: [2,1],
#             3: [3,1],
#             4: [4,1]
#         }  # (firm.id(1-4), 0/1]
#         firm_results = {
#             1: [1,0],
#             2: [2,0],
#             3: [3,0],
#             4: [4,0]
#         }  # (candidate.id, reward)
#         candidate_messages = [[candidate_system_prompts[i]] for i in range(4)]
#         firm_messages = [[firm_system_prompts[i]] for i in range(4)]

#         # initial hiring round
#         for i, firm in enumerate(firm_vectors):
#             selected_candidates = candidate_vectors[i]['encoded_features']
#             logger.debug(f"firm: {i} initsel_features {matrix_print(selected_candidates[0])}")
            
#             # update firm reward
#             theta = np.full((2, 4), 2)
#             reward = np.sum(theta * selected_candidates * firm.T) + 2.0
#             firm_results[i+1][1] = reward
#             all_offer_features[i].append(candidate_vectors[i]['encoded_features'][0])
        
#         current_round = 0
#         return all_application_features, all_offer_features, candidate_results, firm_results, candidate_messages, firm_messages, current_round

#     # restore states from file
#     if os.path.exists(state_file):
#         with open(state_file, "r") as f:
#             saved_state = json.load(f)
#         # Unpack recovery status
#         data_all_application_features = saved_state["all_application_features"]
#         all_application_features = {int(k): v for k, v in data_all_application_features.items()}
#         data_all_offer_features = saved_state["all_offer_features"]
#         all_offer_features = {int(k): v for k, v in data_all_offer_features.items()}
#         data_candidate_results = saved_state["candidate_results"]
#         candidate_results = {int(k): v for k, v in data_candidate_results.items()}
#         data_firm_results = saved_state["firm_results"]
#         firm_results = {int(k): v for k, v in data_firm_results.items()}
#         candidate_messages = saved_state["candidate_messages"]
#         firm_messages = saved_state["firm_messages"]
#         current_round = saved_state["current_round"]
#     else:
#         # initialize variables
#         all_application_features, all_offer_features, candidate_results, firm_results, candidate_messages, firm_messages, current_round = initialize_variables()

#     logger.debug("==================== Game Start =======================")
    
#     with open(f"{log_dir}/candidate_evaluation_{turn+1}.csv", "w", newline='') as candidate_csvfile, \
#          open(f"{log_dir}/firm_evaluation_{turn+1}.csv", "w", newline='') as firm_csvfile:

#         # init CSV writter
#         candidate_writer = csv.DictWriter(candidate_csvfile, fieldnames=["round", "agent"] + [str(i) for i in range(1, 5)])
#         firm_writer = csv.DictWriter(firm_csvfile, fieldnames=["round", "agent"] + [str(i) for i in range(1, 5)])

#         # write header
#         candidate_writer.writeheader()
#         firm_writer.writeheader()

#         ### iterate
#         # Iterate through multiple rounds
#         for round_num in range(current_round, NUM_ROUND):
#             logger.debug(f"--- Round {round_num + 1} ---")
#             print(f"--- Round {round_num + 1} ---")

#             ### Step 1: Candidates apply to firms
#             applications = {firm: [] for firm in range(4)}  # initialize applications
#             applications_features = {firm: [] for firm in range(4)}  # initialize applications
#             response_applications = []
#             # for i, candidate in enumerate(candidate_clients):
#             for i in range(4):
#                 status = "hired" if candidate_results[i+1][1] == 1 else "not hired"
#                 prompt1 = {
#                     "role": "assistant",
#                     "content": (f"F{candidate_results[i+1][0]}.")
#                 }
#                 prompt2 ={
#                     "role": "user", 
#                     "content": (
#                         f"In the last round, you applied to F{candidate_results[i+1][0]} and you were {status}. "
#                         "Choose a firm. Respond with the firm ID (e.g., F1)."
#                     )
#                 }
#                 candidate_messages[i].append(prompt1)
#                 candidate_messages[i].append(prompt2)
#                 candidate_response = candidate_clients[i].chat.completions.create(
#                     messages=candidate_messages[i],
#                     model="gpt-4o-mini",
#                     logprobs=True,
#                     max_tokens=2,
#                     top_logprobs=4
#                 )
#                 # print(f"[DEBUG] {candidate_response.choices[0].logprobs}")
#                 response_applications.append(candidate_response.choices[0].message.content.strip())
#                 # candidate_messages[i].append(candidate_response.choices[0].message)
#                 scores = extract_scores_from_logprobs(candidate_response.choices[0].logprobs)
#                 candidate_writer.writerow({"round": round_num, "agent": f"C{i+1}", **scores})
#                 print(f"[debug] candidate message: {candidate_messages[i]}")
#                 logger.debug(f"[DEBUG] candidate {i}, response {response_applications[i]}")
#                 time.sleep(10)


#             # process the response
#             for i, firm_name in enumerate(response_applications):
#                 # Extract firm index from firm name (assuming "F1", "F2", etc.)
#                 firm_index = int(re.findall(r'F(\d+)', firm_name)[-1]) - 1 if re.findall(r'F(\d+)', firm_name) else None
#                 # firm_index = np.random.randint(0,4)
#                 if firm_index < 0 or firm_index > 3:
#                     logger.debug(f"Candidate{i} apply for Firm out of index")
#                 else:
#                     logger.debug(f"Candidate{i} apply for Firm{firm_index}")
#                     print(f"Candidate{i} apply for Firm{firm_index}")
#                     selected_candidates = candidate_vectors[i]['features']
#                     candidate_feature = candidate_vectors[i]['encoded_features']
#                     applications_features[firm_index].append(f"C{i+1}:[{selected_candidates['category#1']}, {selected_candidates['category#2']}] ")
#                     applications[firm_index].append(i)
#                     all_application_features[firm_index].append(candidate_feature[0])
            
#             print(f"[DEBUG] {applications}")
                

#             ### Step 2: Firms select candidates
#             offers = {firm: [] for firm in range(4)} 
#             selected_candidates = {firm: [] for firm in range(4)}
#             for i in range(4):
#                 # has application
#                 if len(applications_features[i]) > 0: # skip 0/1
#                     candidate_pool = applications_features[i]
#                     # no hiring last round
#                     if firm_results[i+1][0] == 0:
#                         prompt = {
#                             "role": "user", 
#                             "content": (
#                                 f"In the last round, there were no candidates who applied, so you hired no one.  "
#                                 f"Here are the candidates who applied in this round: {candidate_pool}. "
#                                 "Choose a candidate."
#                                 # "Respond only with the ID (e.g., C1)."
#                             )
#                         }
#                         firm_messages[i].append(prompt)
#                     else:
#                         prompt1 = {
#                             "role": "assistant",
#                             "content": (f"C{firm_results[i+1][0]}")
#                         }
#                         prompt2 = {
#                             "role": "user", 
#                             "content": (
#                                 f"In the last round, you hired C{firm_results[i+1][0]} and their performance was {score_to_grade(firm_results[i+1][1])}. "
#                                 f"Here are the candidates who applied in this round: {candidate_pool}. "
#                                 "Choose a candidate."
#                                 # "Respond only with the ID (e.g., C1)."
#                             )
#                         }
#                         firm_messages[i].append(prompt1)
#                         firm_messages[i].append(prompt2)
#                     firm_responses = firm_clients[i].chat.completions.create(
#                         messages = firm_messages[i],
#                         model="gpt-4o-mini",
#                         logprobs=True,
#                         max_tokens=2,
#                         top_logprobs=4
#                     )
#                     selected_candidate = firm_responses.choices[0].message.content.strip()
#                     # firm_messages[i].append(firm_responses.choices[0].message)
#                     print(f"[debug] firm message: {firm_messages[i]}")
#                     selected_candidates[i].append(selected_candidate)
#                     scores = extract_scores_from_logprobs(firm_responses.choices[0].logprobs)
#                     firm_writer.writerow({"round": round_num, "agent": f"F{i+1}", **scores})
#                     time.sleep(10)
#                     logger.debug(f"[DEBUG] candidate {i}, select {selected_candidates[i]}")
#                     print(f"[DEBUG] candidate {i}, select {selected_candidates[i]}")
#                 else:
#                     # but not ask llm
#                     prompt1 = {
#                         "role": "assistant",
#                         "content": (f"C{firm_results[i+1][0]}")}
#                     prompt2 = {
#                         "role": "user", 
#                         "content": (
#                             f"In the last round, you hired C{firm_results[i+1][0]} and their performance was {score_to_grade(firm_results[i+1][1])}. "
#                             "In this round, there are no candidates who applied. You do not need to choose."
#                         )
#                     }
#                     firm_messages[i].append(prompt1)
#                     firm_messages[i].append(prompt2)
#                     firm_writer.writerow({"round": round_num, "agent": f"F{i+1}", "1":0, "2":0,"3":0, "4":0})


#             print(f"[DEBUG] {selected_candidates}")

#             # process the response
#             for i, candidate_list in selected_candidates.items():
#                 if candidate_list:
#                     candidate_index = int(re.findall(r'C(\d+)', candidate_list[0])[-1]) - 1 if re.findall(r'C(\d+)', candidate_list[0]) else None
#                     # candidate_index = np.random.randint(0,4)
#                     logger.debug(f"Firm{i} hires Candidate{candidate_index}, feature: {candidate_vectors[i]['encoded_features']}")
#                     offers[i].append(candidate_index)
#                     all_offer_features[i].append(candidate_vectors[candidate_index]['encoded_features'][0])
#                 else:
#                     logger.debug(f"Firm{i} hires no candidate")
                

#             ### step 3 Update results
#             for firm_id, candidate_list in applications.items():
#                 firm_feature = firm_vectors[firm_id]
#                 if candidate_list == []:
#                     firm_results[firm_id+1][0] = 0
#                     firm_results[firm_id+1][1] = 0
#                 else:
#                     for i in candidate_list:
#                         selected_candidates = candidate_vectors[i]['encoded_features']
#                         if i in offers[firm_id]:
#                             candidate_results[i+1][0] = firm_id+1
#                             candidate_results[i+1][1] = 1
#                             theta = np.full((2, 4), 2)
#                             reward = np.sum(theta * selected_candidates * firm_feature.T) + np.random.normal(0,1)
#                             firm_results[firm_id+1][0] = i+1
#                             firm_results[firm_id+1][1] = reward
#                             logger.debug(f"candidate: {i} applied_firm: {firm_feature} hired: 1")
#                         else:
#                             candidate_results[i+1][0] = firm_id+1
#                             candidate_results[i+1][1] = 0
#                             logger.debug(f"candidate: {i} applied_firm: {firm_feature} hired: 0")
                        
#             ### Overtime result output
#             logger.debug("---------------Offer Results-------------------")
#             entropies_offer1 = []
#             entropies_offer2 = []
#             for firm_id in range(4):
#                 logger.debug(f"firm: {firm_id} initsel_features {matrix_print(candidate_vectors[firm_id]['encoded_features'][0])}")
#                 logger.debug(f"firm: {firm_id} select_features  {matrix_print(np.sum(all_offer_features[firm_id], axis=0))}")
#                 logger.debug("------------------------------------------------")
#                 # cal entropy
#                 entropy1 = cal_entropy(all_offer_features[firm_id])
#                 entropies_offer1.append(entropy1)
#                 entropy2 = calculate_entropy(all_offer_features[firm_id])
#                 entropies_offer2.append(entropy2)

#             entropies1 = []
#             entropies2 = []
#             for firm_id, apply_feature in all_application_features.items():
#                 logger.debug(f"firm: {firm_id} initsel_features {matrix_print(candidate_vectors[firm_id]['encoded_features'][0])}")
#                 if apply_feature == []:
#                     entropies1.append(0.0)
#                     entropies2.append(0.0)
#                 else:
#                     logger.debug(f"firm: {firm_id} applied_features {matrix_print(np.sum(apply_feature, axis=0))}")
#                     # cal entropy
#                     entropy1 = cal_entropy(apply_feature)
#                     entropy2 = calculate_entropy(apply_feature)
#                     entropies1.append(entropy1)
#                     entropies2.append(entropy2)
#                 logger.debug("----------------------------------------------------")

#             logger.debug(f"offers entropy:{np.mean(entropies_offer1)}")
#             logger.debug(f"offers entropy all:{np.mean(entropies_offer2)}")
#             logger.debug(f"Applicants entropy:{np.mean(entropies1)}")
#             logger.debug(f"Applicants entropy all:{np.mean(entropies2)}")

#             current_round +=1
#             # restore the results
#             state_to_save = {
#                 "all_application_features": all_application_features,
#                 "all_offer_features": all_offer_features,
#                 "candidate_results": candidate_results,
#                 "firm_results": firm_results,
#                 "candidate_messages": candidate_messages,
#                 "firm_messages": firm_messages,
#                 "current_round": current_round
#             }
            
#             with open(state_file, "w") as f:
#                 json.dump(state_to_save, f, cls=NumpyEncoder)

#     logger.debug("==================== Game End =======================")

# def run_experiment_decentralized_grade(candidate_file, firm_vector_file, logger, turn, log_dir):
#     ### feature generation
#     # Firm types
#     # firm_types = ["technology", "management", "service", "production"]
#     firm_types = ["type#1", "type#2"]
#     # Load firm vectors
#     firm_vectors = load_firm_vectors(firm_vector_file)
#     # Generate descriptions
#     firm_descriptions = generate_firm_descriptions(firm_vectors, firm_types)

#     # Load feature template and generate candidates
#     candidate_feature_template = load_feature_distributions(candidate_file)
#     candidate_vectors = set_up_candidates(candidate_feature_template)

#     ### initialization
#     # Initialize OpenAI clients
#     print(f"[DEBUG] Initialize OpenAI clients")
#     candidate_clients = initialize_clients_sync(4)
#     firm_clients = initialize_clients_sync(4)

#     # Initialize result tracking variables
#     # Define file paths for saving and restoring states
#     state_file = f"result_dir/state/game_state_decentralized_{turn}.json"
#     # Default values for initialization variables
#     def initialize_variables():
#         candidate_system_prompts, firm_system_prompts = generate_initial_prompts_decentralized(candidate_vectors, firm_descriptions)
        
#         all_application_features = {firm: [] for firm in range(4)}
#         all_offer_features = {firm: [] for firm in range(4)}
#         candidate_results = {
#             1: [1,1],
#             2: [2,1],
#             3: [3,1],
#             4: [4,1]
#         }  # (firm.id(1-4), 0/1]
#         firm_results = {
#             1: [1,0],
#             2: [2,0],
#             3: [3,0],
#             4: [4,0]
#         }  # (candidate.id, reward)
#         candidate_messages = [[candidate_system_prompts[i]] for i in range(4)]
#         firm_messages = [[firm_system_prompts[i]] for i in range(4)]

#         # initial hiring round
#         for i, firm in enumerate(firm_vectors):
#             selected_candidates = candidate_vectors[i]['encoded_features']
#             logger.debug(f"firm: {i} initsel_features {matrix_print(selected_candidates[0])}")
            
#             # update firm reward
#             theta = np.full((2, 4), 2)
#             reward = np.sum(theta * selected_candidates * firm.T) + 2.0
#             firm_results[i+1][1] = reward
#             all_offer_features[i].append(candidate_vectors[i]['encoded_features'][0])
        
#         current_round = 0
#         return all_application_features, all_offer_features, candidate_results, firm_results, candidate_messages, firm_messages, current_round

#     # restore states from file
#     if os.path.exists(state_file):
#         with open(state_file, "r") as f:
#             saved_state = json.load(f)
#         # Unpack recovery status
#         data_all_application_features = saved_state["all_application_features"]
#         all_application_features = {int(k): v for k, v in data_all_application_features.items()}
#         data_all_offer_features = saved_state["all_offer_features"]
#         all_offer_features = {int(k): v for k, v in data_all_offer_features.items()}
#         data_candidate_results = saved_state["candidate_results"]
#         candidate_results = {int(k): v for k, v in data_candidate_results.items()}
#         data_firm_results = saved_state["firm_results"]
#         firm_results = {int(k): v for k, v in data_firm_results.items()}
#         candidate_messages = saved_state["candidate_messages"]
#         firm_messages = saved_state["firm_messages"]
#         current_round = saved_state["current_round"]
#     else:
#         # initialize variables
#         all_application_features, all_offer_features, candidate_results, firm_results, candidate_messages, firm_messages, current_round = initialize_variables()

#     logger.debug("==================== Game Start =======================")
#     with open(f"{log_dir}/candidate_evaluation_{turn+1}.csv", "w", newline='') as candidate_csvfile, \
#          open(f"{log_dir}/firm_evaluation_{turn+1}.csv", "w", newline='') as firm_csvfile:

#         # init CSV writter
#         candidate_writer = csv.DictWriter(candidate_csvfile, fieldnames=["round", "agent"] + [str(i) for i in range(1, 5)])
#         firm_writer = csv.DictWriter(firm_csvfile, fieldnames=["round", "agent"] + [str(i) for i in range(1, 5)])

#         # write header
#         candidate_writer.writeheader()
#         firm_writer.writeheader()    
#         ### iterate
#         # Iterate through multiple rounds
#         for round_num in range(current_round, NUM_ROUND):
#             logger.debug(f"--- Round {round_num + 1} ---")
#             print(f"--- Round {round_num + 1} ---")

#             ### Step 1: Candidates apply to firms
#             applications = {firm: [] for firm in range(4)}  # initialize applications
#             applications_features = {firm: [] for firm in range(4)}  # initialize applications
#             response_applications = []
#             # for i, candidate in enumerate(candidate_clients):
#             for i in range(4):
#                 other_cand_result = ""
#                 for j in range(4):
#                     if i !=j:
#                         other_status = "hired" if candidate_results[j+1][1] == 1 else "not hired"
#                         other_cand_result += f"C{j+1} applied to F{candidate_results[i+1][0]} and {other_status}. "

#                 status = "hired" if candidate_results[i+1][1] == 1 else "not hired"
#                 prompt1 = {
#                     "role": "assistant",
#                     "content": (f"F{candidate_results[i+1][0]}.")
#                 }
#                 prompt2 = {
#                     "role": "user", 
#                     "content": (
#                         f"In the last round, you applied to F{candidate_results[i+1][0]} and you were {status}. "
#                         f"Here are the results of other candidates: {other_cand_result}. "
#                         "Choose a firm. Respond with the firm ID (e.g., F1)."
#                     )
#                 }
#                 candidate_messages[i].append(prompt1)
#                 candidate_messages[i].append(prompt2)
#                 candidate_response = candidate_clients[i].chat.completions.create(
#                     messages=candidate_messages[i],
#                     model="gpt-4o-mini",
#                     logprobs=True,
#                     max_tokens=2,
#                     top_logprobs=4
#                 )
#                 response_applications.append(candidate_response.choices[0].message.content.strip())
#                 print(f"[debug] candidate message: {candidate_messages[i]}")
#                 logger.debug(f"[DEBUG] candidate {i}, response {response_applications[i]}")
#                 scores = extract_scores_from_logprobs(candidate_response.choices[0].logprobs)
#                 candidate_writer.writerow({"round": round_num, "agent": f"C{i+1}", **scores})
#                 time.sleep(10)


#             # process the response
#             for i, firm_name in enumerate(response_applications):
#                 # Extract firm index from firm name (assuming "F1", "F2", etc.)
#                 firm_index = int(re.findall(r'F(\d+)', firm_name)[-1]) - 1 if re.findall(r'F(\d+)', firm_name) else None
#                 # firm_index = np.random.randint(0,4)
#                 if firm_index < 0 or firm_index > 3:
#                     logger.debug(f"Candidate{i} apply for Firm out of index")
#                 else:
#                     logger.debug(f"Candidate{i} apply for Firm{firm_index}")
#                     print(f"Candidate{i} apply for Firm{firm_index}")
#                     selected_candidates = candidate_vectors[i]['features']
#                     candidate_feature = candidate_vectors[i]['encoded_features']
#                     applications_features[firm_index].append(f"C{i+1}:[{selected_candidates['category#1']}, {selected_candidates['category#2']}] ")
#                     applications[firm_index].append(i)
#                     all_application_features[firm_index].append(candidate_feature[0])
            
#             print(f"[DEBUG] {applications}")
                

#             ### Step 2: Firms select candidates
#             offers = {firm: [] for firm in range(4)} 
#             selected_candidates = {firm: [] for firm in range(4)}
#             for i in range(4):
#                 other_firm_result = ""
#                 for j in range(4):
#                     if i != j and firm_results[j+1][0] != 0:
#                         cand = candidate_vectors[firm_results[j+1][0]-1]
#                         other_f = f"[{cand['features']['category#1']}, {cand['features']['category#2']}]"
#                         other_firm_result  += f"F{j+1} hired C{firm_results[j+1][0]} with feature {other_f} and performance was {score_to_grade(firm_results[j+1][1])}."

#                 if len(applications_features[i]) > 0: # skip 0/1
#                     candidate_pool = applications_features[i]
#                     # no hiring last round
#                     if firm_results[i+1][0] == 0:
#                         prompt = {
#                             "role": "user", 
#                             "content": (
#                                 "In the last round, there were no candidates who applied, so you hired no one. "
#                                 f"Here are the results of other firms: {other_firm_result}."
#                                 f"Here are the candidates who applied in this round: {candidate_pool}. "
#                                 "Choose a candidate." 
#                                 # "Respond only with the ID (e.g., C1)."
#                             )
#                         }
#                         firm_messages[i].append(prompt)
#                     else:
#                         prompt1 = {
#                             "role": "assistant",
#                             "content": (f"C{firm_results[i+1][0]}")
#                         }
#                         prompt2 = {
#                             "role": "user", 
#                             "content": (
#                                 f"In the last round, you hired C{firm_results[i+1][0]} and performance was {score_to_grade(firm_results[i+1][1])}. "
#                                 f"Here are the results of other firms: {other_firm_result}"
#                                 f"Here are the candidates who applied in this round: {candidate_pool}. "
#                                 "Choose a candidate. Respond only with the ID (e.g., C1)."
#                             )
#                         }
#                         firm_messages[i].append(prompt1)
#                         firm_messages[i].append(prompt2)

#                     firm_responses = firm_clients[i].chat.completions.create(
#                         messages = firm_messages[i],
#                         model="gpt-4o-mini",
#                         logprobs=True,
#                         max_tokens=2,
#                         top_logprobs=4
#                     )
#                     selected_candidate = firm_responses.choices[0].message.content.strip()
#                     # firm_messages[i].append(firm_responses.choices[0].message)
#                     print(f"[debug] firm message: {firm_messages[i]}")
#                     selected_candidates[i].append(selected_candidate)
#                     scores = extract_scores_from_logprobs(firm_responses.choices[0].logprobs)
#                     firm_writer.writerow({"round": round_num, "agent": f"F{i+1}", **scores})
#                     time.sleep(10)
#                     logger.debug(f"[DEBUG] candidate {i}, select {selected_candidates[i]}")
#                     print(f"[DEBUG] candidate {i}, select {selected_candidates[i]}")
#                 else:
#                     # but do not ask
#                     prompt1 = {
#                         "role": "assistant",
#                         "content": (f"C{firm_results[i+1][0]}")
#                     }
#                     prompt = {
#                         "role": "user", 
#                         "content": (
#                             f"In the last round, you hired C{firm_results[i+1][0]} and performance was {score_to_grade(firm_results[i+1][1])}. "
#                             f"Here are the results of other firms: {other_firm_result}"
#                             "In this round, there are no candidates who applied. You do not need to choose."
#                         )
#                     }
#                     firm_messages[i].append(prompt1)
#                     firm_messages[i].append(prompt2)
#                     firm_writer.writerow({"round": round_num, "agent": f"F{i+1}", "1":0, "2":0,"3":0, "4":0})

#             print(f"[DEBUG] {selected_candidates}")

#             # process the response
#             for i, candidate_list in selected_candidates.items():
#                 if candidate_list:
#                     candidate_index = int(re.findall(r'C(\d+)', candidate_list[0])[-1]) - 1 if re.findall(r'C(\d+)', candidate_list[0]) else None
#                     # candidate_index = np.random.randint(0,4)
#                     logger.debug(f"Firm{i} hires Candidate{candidate_index}, feature: {candidate_vectors[i]['encoded_features']}")
#                     offers[i].append(candidate_index)
#                     all_offer_features[i].append(candidate_vectors[candidate_index]['encoded_features'][0])
#                 else:
#                     logger.debug(f"Firm{i} hires no candidate")
                

#             ### step 3 Update results
#             for firm_id, candidate_list in applications.items():
#                 firm_feature = firm_vectors[firm_id]
#                 if candidate_list == []:
#                     firm_results[firm_id+1][0] = 0
#                     firm_results[firm_id+1][1] = 0
#                 else:
#                     for i in candidate_list:
#                         selected_candidates = candidate_vectors[i]['encoded_features']
#                         if i in offers[firm_id]:
#                             candidate_results[i+1][0] = firm_id+1
#                             candidate_results[i+1][1] = 1
#                             theta = np.full((2, 4), 2)
#                             reward = np.sum(theta * selected_candidates * firm_feature.T) + np.random.normal(0,1)
#                             firm_results[firm_id+1][0] = i+1
#                             firm_results[firm_id+1][1] = reward
#                             logger.debug(f"candidate: {i} applied_firm: {firm_feature} hired: 1")
#                         else:
#                             candidate_results[i+1][0] = firm_id+1
#                             candidate_results[i+1][1] = 0
#                             logger.debug(f"candidate: {i} applied_firm: {firm_feature} hired: 0")
                    
#             ### Overtime result output
#             logger.debug("---------------Offer Results-------------------")
#             entropies_offer1 = []
#             entropies_offer2 = []
#             for firm_id in range(4):
#                 logger.debug(f"firm: {firm_id} initsel_features {matrix_print(candidate_vectors[firm_id]['encoded_features'][0])}")
#                 logger.debug(f"firm: {firm_id} select_features  {matrix_print(np.sum(all_offer_features[firm_id], axis=0))}")
#                 logger.debug("------------------------------------------------")
#                 # cal entropy
#                 entropy1 = cal_entropy(all_offer_features[firm_id])
#                 entropies_offer1.append(entropy1)
#                 entropy2 = calculate_entropy(all_offer_features[firm_id])
#                 entropies_offer2.append(entropy2)

#             entropies1 = []
#             entropies2 = []
#             for firm_id, apply_feature in all_application_features.items():
#                 logger.debug(f"firm: {firm_id} initsel_features {matrix_print(candidate_vectors[firm_id]['encoded_features'][0])}")
#                 if apply_feature == []:
#                     entropies1.append(0.0)
#                     entropies2.append(0.0)
#                 else:
#                     logger.debug(f"firm: {firm_id} applied_features {matrix_print(np.sum(apply_feature, axis=0))}")
#                     # cal entropy
#                     entropy1 = cal_entropy(apply_feature)
#                     entropy2 = calculate_entropy(apply_feature)
#                     entropies1.append(entropy1)
#                     entropies2.append(entropy2)
#                 logger.debug("----------------------------------------------------")

#             logger.debug(f"offers entropy:{np.mean(entropies_offer1)}")
#             logger.debug(f"offers entropy all:{np.mean(entropies_offer2)}")
#             logger.debug(f"Applicants entropy:{np.mean(entropies1)}")
#             logger.debug(f"Applicants entropy all:{np.mean(entropies2)}")
            
#             current_round += 1
#             # restore the round result
#             state_to_save = {
#                 "all_application_features": all_application_features,
#                 "all_offer_features": all_offer_features,
#                 "candidate_results": candidate_results,
#                 "firm_results": firm_results,
#                 "candidate_messages": candidate_messages,
#                 "firm_messages": firm_messages,
#                 "current_round": current_round
#             }
            
#             with open(state_file, "w") as f:
#                 json.dump(state_to_save, f, cls=NumpyEncoder)

    
#     logger.debug("==================== Game End =======================")


def run_experiment_distributed(candidate_file, firm_vector_file, logger, turn, log_dir):
    ### feature generation
    # Firm types
    # firm_types = ["technology", "management", "service", "production"]
    firm_types = ["type#1", "type#2"]
    # Load firm vectors
    firm_vectors = load_firm_vectors(firm_vector_file)
    # Generate descriptions
    firm_descriptions = generate_firm_descriptions(firm_vectors, firm_types)

    # Load feature template and generate candidates
    candidate_feature_template = load_feature_distributions(candidate_file)
    candidate_vectors = set_up_candidates(candidate_feature_template)

    ### initialization
    # Initialize OpenAI clients
    print(f"[DEBUG] Initialize OpenAI clients")
    candidate_clients = initialize_clients_sync(4)
    firm_clients = initialize_clients_sync(4)

    # Initialize result tracking variables
    # Define file paths for saving and restoring states
    state_file = f"result_dir/state/game_state_distributed_{turn}.json"
    # Default values for initialization variables
    def initialize_variables():
        candidate_system_prompts, firm_system_prompts = generate_initial_prompts_distributed(candidate_vectors, firm_descriptions)
        
        all_application_features = {firm: [] for firm in range(4)}
        all_offer_features = {firm: [] for firm in range(4)}
        candidate_results = {
            1: [1,1],
            2: [2,1],
            3: [3,1],
            4: [4,1]
        }  # (firm.id(1-4), 0/1]
        firm_results = {
            1: [1,0],
            2: [2,0],
            3: [3,0],
            4: [4,0]
        }  # (candidate.id, reward)
        candidate_messages = [[candidate_system_prompts[i]] for i in range(4)]
        firm_messages = [[firm_system_prompts[i]] for i in range(4)]

        # 初始化每个 candidate 的雇佣统计结构
        hire_summary_candidate = {
            candidate_id: {
                'id': candidate_id,
                'firms': {firm_id: {'hire_count': 0, 'not_hire_count': 0} for firm_id in range(1, 4 + 1)}
            }
            for candidate_id in range(1, 4 + 1)
        }

        # 初始化每个 firm 的收益统计结构
        hire_summary_firm = {
            firm_id: {
                'id': firm_id,
                'candidates': {candidate_id: {'rewards': []} for candidate_id in range(1, 4 + 1)}
            }
            for firm_id in range(1, 4 + 1)
        }

        def update_hire_summary(candidate_id, firm_id, hire_status, reward=0.0):
            # update candidate
            if hire_status == 1:
                hire_summary_candidate[candidate_id]['firms'][firm_id]['hire_count'] += 1
                # update firm
                hire_summary_firm[firm_id]['candidates'][candidate_id]['rewards'].append(reward)
            else:
                hire_summary_candidate[candidate_id]['firms'][firm_id]['not_hire_count'] += 1

        # initial hiring round
        for i, firm in enumerate(firm_vectors):
            selected_candidates = candidate_vectors[i]['encoded_features']
            logger.debug(f"firm: {i} initsel_features {matrix_print(selected_candidates[0])}")
            
            # update firm reward
            theta = np.full((2, 4), 2)
            reward = np.sum(theta * selected_candidates * firm.T) + 2.0
            firm_results[i+1][1] = reward
            all_offer_features[i].append(candidate_vectors[i]['encoded_features'][0])
            update_hire_summary(i+1, i+1, 1, reward)
        
        current_round = 0
        return all_application_features, all_offer_features, candidate_results, firm_results, candidate_messages, firm_messages, current_round, hire_summary_candidate, hire_summary_firm

    # restore states from file
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            saved_state = json.load(f)
        # Unpack recovery status
        data_all_application_features = saved_state["all_application_features"]
        all_application_features = {int(k): v for k, v in data_all_application_features.items()}
        data_all_offer_features = saved_state["all_offer_features"]
        all_offer_features = {int(k): v for k, v in data_all_offer_features.items()}
        data_candidate_results = saved_state["candidate_results"]
        candidate_results = {int(k): v for k, v in data_candidate_results.items()}
        data_firm_results = saved_state["firm_results"]
        firm_results = {int(k): v for k, v in data_firm_results.items()}
        candidate_messages = saved_state["candidate_messages"]
        firm_messages = saved_state["firm_messages"]
        current_round = saved_state["current_round"]
        hire_summary_candidate = saved_state["hire_summary_candidate"]
        hire_summary_firm = saved_state["hire_summary_firm"]
    else:
        # initialize variables
        all_application_features, all_offer_features, candidate_results, firm_results, candidate_messages, firm_messages, current_round, hire_summary_candidate, hire_summary_firm = initialize_variables()
    

    def update_hire_summary(candidate_id, firm_id, hire_status, reward=0.0):
        # update candidate
        if hire_status == 1:
            hire_summary_candidate[candidate_id]['firms'][firm_id]['hire_count'] += 1
            # update firm
            hire_summary_firm[firm_id]['candidates'][candidate_id]['rewards'].append(reward)
        else:
            hire_summary_candidate[candidate_id]['firms'][firm_id]['not_hire_count'] += 1

    logger.debug("==================== Game Start =======================")
    
    # with open(f"{log_dir}/candidate_evaluation_{turn+1}.csv", "w", newline='') as candidate_csvfile, \
        #  open(f"{log_dir}/firm_evaluation_{turn+1}.csv", "w", newline='') as firm_csvfile:

        # init CSV writter
        # candidate_writer = csv.DictWriter(candidate_csvfile, fieldnames=["round", "agent"] + [str(i) for i in range(1, 5)])
        # firm_writer = csv.DictWriter(firm_csvfile, fieldnames=["round", "agent"] + [str(i) for i in range(1, 5)])

        # write header
        # candidate_writer.writeheader()
        # firm_writer.writeheader()

        ### iterate
        # Iterate through multiple rounds
    for round_num in range(current_round, NUM_ROUND):
        logger.debug(f"--- Round {round_num + 1} ---")
        print(f"--- Round {round_num + 1} ---")

        ### Step 1: Candidates apply to firms
        applications = {firm: [] for firm in range(4)}  # initialize applications
        applications_features = {firm: [] for firm in range(4)}  # initialize applications
        response_applications = []
        # for i, candidate in enumerate(candidate_clients):
        for i in range(4):
            status = "hired" if candidate_results[i+1][1] == 1 else "not hired"
            prompt1 = {
                "role": "assistant",
                "content": (f"F{candidate_results[i+1][0]}.")
            }
            candidate_summary_str = generate_candidate_summary(hire_summary_candidate, candidate_vectors, i+1)
            prompt2 ={
                "role": "user", 
                "content": (
                    f"In the last round, you applied to F{candidate_results[i+1][0]} and you were {status}. "
                    f"Summary of hiring results across all rounds:\n{candidate_summary_str}\n"
                    "Choose a firm and briefly explain why." # Respond with the firm ID (e.g., F1)."
                )
            }
            candidate_messages[i].append(prompt1)
            candidate_messages[i].append(prompt2)
            candidate_response = candidate_clients[i].chat.completions.create(
                messages=candidate_messages[i],
                model="gpt-4o-mini",
                # logprobs=True,
                # max_tokens=2,
                # top_logprobs=4
            )
            # print(f"[DEBUG] {candidate_response.choices[0].logprobs}")
            response_applications.append(candidate_response.choices[0].message.content.strip())
            # logger.debug(f"[DEBUG] candidate {i}, response {candidate_response.choices[0].message.content}")
            # scores = extract_scores_from_logprobs(candidate_response.choices[0].logprobs)
            # candidate_writer.writerow({"round": round_num, "agent": f"C{i+1}", **scores})
            # print(f"[debug] candidate message: {candidate_messages[i]}")
            print(f"[debug] candidate message: {candidate_messages[i]}")
            logger.debug(f"[DEBUG] candidate {i}, response {[response_applications[i]]}")
            # time.sleep(10)


        # process the response
        for i, firm_name in enumerate(response_applications):
            # Extract firm index from firm name (assuming "F1", "F2", etc.)
            firm_index = int(re.findall(r'F(\d+)', firm_name)[-1]) - 1 if re.findall(r'F(\d+)', firm_name) else None
            # firm_index = np.random.randint(0,4)
            if firm_index < 0 or firm_index > 3:
                logger.debug(f"Candidate{i} apply for Firm out of index")
            else:
                logger.debug(f"Candidate{i} apply for Firm{firm_index}")
                print(f"Candidate{i} apply for Firm{firm_index}")
                selected_candidates = candidate_vectors[i]['features']
                candidate_feature = candidate_vectors[i]['encoded_features']
                applications_features[firm_index].append(f"C{i+1}:[{selected_candidates['category#1']}, {selected_candidates['category#2']}] ")
                applications[firm_index].append(i)
                all_application_features[firm_index].append(candidate_feature[0])
        
        print(f"[DEBUG] {applications}")
            

        ### Step 2: Firms select candidates
        offers = {firm: [] for firm in range(4)} 
        selected_candidates = {firm: [] for firm in range(4)}
        for i in range(4):
            # has application
            firm_summary_str = generate_firm_summary(hire_summary_firm, candidate_vectors, firm_descriptions, i+1)
            if len(applications_features[i]) > 0: # skip 0/1
                candidate_pool = applications_features[i]
                # no hiring last round
                if firm_results[i+1][0] == 0:
                    prompt = {
                        "role": "user", 
                        "content": (
                            f"In the last round, there were no candidates who applied, so you hired no one.  "
                            f"Summary of hiring results across all rounds:\n{firm_summary_str}\n"
                            f"Here are the candidates who applied in this round: {candidate_pool}. "
                            "Choose a candidate and briefly explain why."
                            #   Respond only with the ID (e.g., C1)."
                        )
                    }
                    firm_messages[i].append(prompt)
                else:
                    prompt1 = {
                        "role": "assistant",
                        "content": (f"C{firm_results[i+1][0]}")
                    }
                    prompt2 = {
                        "role": "user", 
                        "content": (
                            f"In the last round, you hired C{firm_results[i+1][0]} and their performance was {firm_results[i+1][1]}. "
                            f"Summary of hiring results across all rounds:\n{firm_summary_str}\n"
                            f"Here are the candidates who applied in this round: {candidate_pool}. "
                            "Choose a candidate and briefly explain why." # Respond only with the ID (e.g., C1)."
                        )
                    }
                    firm_messages[i].append(prompt1)
                    firm_messages[i].append(prompt2)
                firm_responses = firm_clients[i].chat.completions.create(
                    messages = firm_messages[i],
                    model="gpt-4o-mini",
                    # logprobs=True,
                    # max_tokens=2,
                    # top_logprobs=4
                )
                selected_candidate = firm_responses.choices[0].message.content.strip()
                # firm_messages[i].append(firm_responses.choices[0].message)
                print(f"[debug] firm message: {firm_messages[i]}")
                selected_candidates[i].append(selected_candidate)
                # scores = extract_scores_from_logprobs(firm_responses.choices[0].logprobs)
                # firm_writer.writerow({"round": round_num, "agent": f"F{i+1}", **scores})
                # time.sleep(10)
                logger.debug(f"[DEBUG] firm {i}, select {selected_candidates[i]}")
                print(f"[DEBUG] firm {i}, select {selected_candidates[i]}")
            else:
                # but not ask llm
                prompt1 = {
                    "role": "assistant",
                    "content": (f"C{firm_results[i+1][0]}")}
                prompt2 = {
                    "role": "user", 
                    "content": (
                        f"In the last round, you hired C{firm_results[i+1][0]} and their performance was {firm_results[i+1][1]}. "
                        f"Summary of hiring results across all rounds:\n{firm_summary_str}\n"
                        "In this round, there are no candidates who applied. You do not need to choose."
                    )
                }
                firm_messages[i].append(prompt1)
                firm_messages[i].append(prompt2)
                # firm_writer.writerow({"round": round_num, "agent": f"F{i+1}", "1":0, "2":0,"3":0, "4":0})


        print(f"[DEBUG] {selected_candidates}")

        # process the response
        for i, candidate_list in selected_candidates.items():
            if candidate_list:
                candidate_index = int(re.findall(r'C(\d+)', candidate_list[0])[-1]) - 1 if re.findall(r'C(\d+)', candidate_list[0]) else None
                # candidate_index = np.random.randint(0,4)
                logger.debug(f"Firm{i} hires Candidate{candidate_index}, feature: {candidate_vectors[i]['encoded_features']}")
                offers[i].append(candidate_index)
                all_offer_features[i].append(candidate_vectors[candidate_index]['encoded_features'][0])
            else:
                logger.debug(f"Firm{i} hires no candidate")
            

        ### step 3 Update results
        for firm_id, candidate_list in applications.items():
            firm_feature = firm_vectors[firm_id]
            if candidate_list == []:
                firm_results[firm_id+1][0] = 0
                firm_results[firm_id+1][1] = 0
            else:
                for i in candidate_list:
                    selected_candidates = candidate_vectors[i]['encoded_features']
                    if i in offers[firm_id]:
                        candidate_results[i+1][0] = firm_id+1
                        candidate_results[i+1][1] = 1
                        theta = np.full((2, 4), 2)
                        reward = np.sum(theta * selected_candidates * firm_feature.T) + np.random.normal(0,1)
                        firm_results[firm_id+1][0] = i+1
                        firm_results[firm_id+1][1] = reward
                        logger.debug(f"candidate: {i} applied_firm: {firm_feature} hired: 1")
                        update_hire_summary(i+1,firm_id+1,1,reward)
                    else:
                        candidate_results[i+1][0] = firm_id+1
                        candidate_results[i+1][1] = 0
                        logger.debug(f"candidate: {i} applied_firm: {firm_feature} hired: 0")
                        update_hire_summary(i+1,firm_id+1,0)
                    
        ### Overtime result output
        logger.debug("---------------Offer Results-------------------")
        entropies_offer1 = []
        entropies_offer2 = []
        for firm_id in range(4):
            logger.debug(f"firm: {firm_id} initsel_features {matrix_print(candidate_vectors[firm_id]['encoded_features'][0])}")
            logger.debug(f"firm: {firm_id} select_features  {matrix_print(np.sum(all_offer_features[firm_id], axis=0))}")
            logger.debug("------------------------------------------------")
            # cal entropy
            entropy1 = cal_entropy(all_offer_features[firm_id])
            entropies_offer1.append(entropy1)
            entropy2 = calculate_entropy(all_offer_features[firm_id])
            entropies_offer2.append(entropy2)

        entropies1 = []
        entropies2 = []
        for firm_id, apply_feature in all_application_features.items():
            logger.debug(f"firm: {firm_id} initsel_features {matrix_print(candidate_vectors[firm_id]['encoded_features'][0])}")
            if apply_feature == []:
                entropies1.append(0.0)
                entropies2.append(0.0)
            else:
                logger.debug(f"firm: {firm_id} applied_features {matrix_print(np.sum(apply_feature, axis=0))}")
                # cal entropy
                entropy1 = cal_entropy(apply_feature)
                entropy2 = calculate_entropy(apply_feature)
                entropies1.append(entropy1)
                entropies2.append(entropy2)
            logger.debug("----------------------------------------------------")

        logger.debug(f"offers entropy:{np.mean(entropies_offer1)}")
        logger.debug(f"offers entropy all:{np.mean(entropies_offer2)}")
        logger.debug(f"Applicants entropy:{np.mean(entropies1)}")
        logger.debug(f"Applicants entropy all:{np.mean(entropies2)}")

        current_round +=1
        # restore the results
        state_to_save = {
            "all_application_features": all_application_features,
            "all_offer_features": all_offer_features,
            "candidate_results": candidate_results,
            "firm_results": firm_results,
            "candidate_messages": candidate_messages,
            "firm_messages": firm_messages,
            "current_round": current_round,
            "hire_summary_candidate": hire_summary_candidate,
            "hire_summary_firm": hire_summary_firm
        }
        
        with open(state_file, "w") as f:
            json.dump(state_to_save, f, cls=NumpyEncoder)

    logger.debug("==================== Game End =======================")

def run_experiment_decentralized(candidate_file, firm_vector_file, logger, turn, log_dir):
    ### feature generation
    # Firm types
    # firm_types = ["technology", "management", "service", "production"]
    firm_types = ["type#1", "type#2"]
    # Load firm vectors
    firm_vectors = load_firm_vectors(firm_vector_file)
    # Generate descriptions
    firm_descriptions = generate_firm_descriptions(firm_vectors, firm_types)

    # Load feature template and generate candidates
    candidate_feature_template = load_feature_distributions(candidate_file)
    candidate_vectors = set_up_candidates(candidate_feature_template)

    ### initialization
    # Initialize OpenAI clients
    print(f"[DEBUG] Initialize OpenAI clients")
    candidate_clients = initialize_clients_sync(4)
    firm_clients = initialize_clients_sync(4)

    # Initialize result tracking variables
    # Define file paths for saving and restoring states
    state_file = f"result_dir/state/game_state_decentralized_{turn}.json"
    # Default values for initialization variables
    def initialize_variables():
        def update_hire_summary(candidate_id, firm_id, hire_status, reward=0.0):
            # update candidate
            if hire_status == 1:
                hire_summary_candidate[candidate_id]['firms'][firm_id]['hire_count'] += 1
                # update firm
                hire_summary_firm[firm_id]['candidates'][candidate_id]['rewards'].append(reward)
            else:
                hire_summary_candidate[candidate_id]['firms'][firm_id]['not_hire_count'] += 1
     
        candidate_system_prompts, firm_system_prompts = generate_initial_prompts_decentralized(candidate_vectors, firm_descriptions)
        
        all_application_features = {firm: [] for firm in range(4)}
        all_offer_features = {firm: [] for firm in range(4)}
        candidate_results = {
            1: [1,1],
            2: [2,1],
            3: [3,1],
            4: [4,1]
        }  # (firm.id(1-4), 0/1]
        firm_results = {
            1: [1,0],
            2: [2,0],
            3: [3,0],
            4: [4,0]
        }  # (candidate.id, reward)
        candidate_messages = [[candidate_system_prompts[i]] for i in range(4)]
        firm_messages = [[firm_system_prompts[i]] for i in range(4)]

        # 初始化每个 candidate 的雇佣统计结构
        hire_summary_candidate = {
            candidate_id: {
                'id': candidate_id,
                'firms': {firm_id: {'hire_count': 0, 'not_hire_count': 0} for firm_id in range(1, 4 + 1)}
            }
            for candidate_id in range(1, 4 + 1)
        }

        # 初始化每个 firm 的收益统计结构
        hire_summary_firm = {
            firm_id: {
                'id': firm_id,
                'candidates': {candidate_id: {'rewards': []} for candidate_id in range(1, 4 + 1)}
            }
            for firm_id in range(1, 4 + 1)
        }

        # initial hiring round
        for i, firm in enumerate(firm_vectors):
            selected_candidates = candidate_vectors[i]['encoded_features']
            logger.debug(f"firm: {i} initsel_features {matrix_print(selected_candidates[0])}")
            
            # update firm reward
            theta = np.full((2, 4), 2)
            reward = np.sum(theta * selected_candidates * firm.T) + 2.0
            firm_results[i+1][1] = reward
            all_offer_features[i].append(candidate_vectors[i]['encoded_features'][0])
            update_hire_summary(i+1, i+1, 1, reward)
        
        current_round = 0
        return all_application_features, all_offer_features, candidate_results, firm_results, candidate_messages, firm_messages, current_round, hire_summary_candidate, hire_summary_firm

    def update_hire_summary(candidate_id, firm_id, hire_status, reward=0.0):
        # update candidate
        if hire_status == 1:
            hire_summary_candidate[candidate_id]['firms'][firm_id]['hire_count'] += 1
            # update firm
            hire_summary_firm[firm_id]['candidates'][candidate_id]['rewards'].append(reward)
        else:
            hire_summary_candidate[candidate_id]['firms'][firm_id]['not_hire_count'] += 1
        

    # restore states from file
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            saved_state = json.load(f)
        # Unpack recovery status
        data_all_application_features = saved_state["all_application_features"]
        all_application_features = {int(k): v for k, v in data_all_application_features.items()}
        data_all_offer_features = saved_state["all_offer_features"]
        all_offer_features = {int(k): v for k, v in data_all_offer_features.items()}
        data_candidate_results = saved_state["candidate_results"]
        candidate_results = {int(k): v for k, v in data_candidate_results.items()}
        data_firm_results = saved_state["firm_results"]
        firm_results = {int(k): v for k, v in data_firm_results.items()}
        candidate_messages = saved_state["candidate_messages"]
        firm_messages = saved_state["firm_messages"]
        current_round = saved_state["current_round"]
        hire_summary_candidate = saved_state["hire_summary_candidate"]
        hire_summary_firm = saved_state["hire_summary_firm"]
    else:
        # initialize variables
        all_application_features, all_offer_features, candidate_results, firm_results, candidate_messages, firm_messages, current_round, hire_summary_candidate, hire_summary_firm = initialize_variables()

    logger.debug("==================== Game Start =======================")
    # with open(f"{log_dir}/candidate_evaluation_{turn+1}.csv", "w", newline='') as candidate_csvfile, \
        #  open(f"{log_dir}/firm_evaluation_{turn+1}.csv", "w", newline='') as firm_csvfile:

        # init CSV writter
        # candidate_writer = csv.DictWriter(candidate_csvfile, fieldnames=["round", "agent"] + [str(i) for i in range(1, 5)])
        # firm_writer = csv.DictWriter(firm_csvfile, fieldnames=["round", "agent"] + [str(i) for i in range(1, 5)])

        # write header
        # candidate_writer.writeheader()
        # firm_writer.writeheader()    
        ### iterate
        # Iterate through multiple rounds
    for round_num in range(current_round, NUM_ROUND):
        logger.debug(f"--- Round {round_num + 1} ---")
        print(f"--- Round {round_num + 1} ---")

        ### Step 1: Candidates apply to firms
        applications = {firm: [] for firm in range(4)}  # initialize applications
        applications_features = {firm: [] for firm in range(4)}  # initialize applications
        response_applications = []
        # for i, candidate in enumerate(candidate_clients):
        for i in range(4):
            other_cand_result = ""
            for j in range(4):
                if i !=j:
                    other_status = "hired" if candidate_results[j+1][1] == 1 else "not hired"
                    other_cand_result += f"C{j+1} applied to F{candidate_results[i+1][0]} and {other_status}. "

            status = "hired" if candidate_results[i+1][1] == 1 else "not hired"
            prompt1 = {
                "role": "assistant",
                "content": (f"F{candidate_results[i+1][0]}.")
            }
            candidate_summary_str = generate_candidate_summary(hire_summary_candidate, candidate_vectors)
            prompt2 = {
                "role": "user", 
                "content": (
                    f"In the last round, you applied to F{candidate_results[i+1][0]} and you were {status}. "
                    f"Here are the results of other candidates: {other_cand_result}. "
                    f"Summary of hiring results across all rounds:\n {candidate_summary_str} \n"
                    "Choose a firm and briefly explain why." # Respond with the firm ID (e.g., F1)."
                )
            }
            candidate_messages[i].append(prompt1)
            candidate_messages[i].append(prompt2)
            candidate_response = candidate_clients[i].chat.completions.create(
                messages=candidate_messages[i],
                model="gpt-4o-mini",
                # logprobs=True,
                # max_tokens=2,
                # top_logprobs=4
            )
            response_applications.append(candidate_response.choices[0].message.content.strip())
            print(f"[debug] candidate message: {candidate_messages[i]}")
            logger.debug(f"[DEBUG] candidate {i}, response {[response_applications[i]]}")
            # scores = extract_scores_from_logprobs(candidate_response.choices[0].logprobs)
            # candidate_writer.writerow({"round": round_num, "agent": f"C{i+1}", **scores})
            # time.sleep(10)


        # process the response
        for i, firm_name in enumerate(response_applications):
            # Extract firm index from firm name (assuming "F1", "F2", etc.)
            firm_index = int(re.findall(r'F(\d+)', firm_name)[-1]) - 1 if re.findall(r'F(\d+)', firm_name) else None
            # firm_index = np.random.randint(0,4)
            if firm_index < 0 or firm_index > 3:
                logger.debug(f"Candidate{i} apply for Firm out of index")
            else:
                logger.debug(f"Candidate{i} apply for Firm{firm_index}")
                print(f"Candidate{i} apply for Firm{firm_index}")
                selected_candidates = candidate_vectors[i]['features']
                candidate_feature = candidate_vectors[i]['encoded_features']
                applications_features[firm_index].append(f"C{i+1}:[{selected_candidates['category#1']}, {selected_candidates['category#2']}] ")
                applications[firm_index].append(i)
                all_application_features[firm_index].append(candidate_feature[0])
        
        print(f"[DEBUG] {applications}")
            

        ### Step 2: Firms select candidates
        offers = {firm: [] for firm in range(4)} 
        selected_candidates = {firm: [] for firm in range(4)}
        firm_summary_str = generate_firm_summary(hire_summary_firm, candidate_vectors, firm_descriptions)
        for i in range(4):
            # firm_summary_str = generate_firm_summary(hire_summary_firm, candidate_vectors, firm_descriptions)
            other_firm_result = ""
            for j in range(4):
                if i != j and firm_results[j+1][0] != 0:
                    cand = candidate_vectors[firm_results[j+1][0]-1]
                    other_f = f"[{cand['features']['category#1']}, {cand['features']['category#2']}]"
                    other_firm_result  += f"F{j+1} hired C{firm_results[j+1][0]} with feature {other_f} and performance was {firm_results[j+1][1]}."

            if len(applications_features[i]) > 0: # skip 0/1
                candidate_pool = applications_features[i]
                # no hiring last round
                if firm_results[i+1][0] == 0:
                    prompt = {
                        "role": "user", 
                        "content": (
                            "In the last round, there were no candidates who applied, so you hired no one. "
                            f"Here are the results of other firms: {other_firm_result}."
                            f"Summary of hiring results across all rounds:\n{firm_summary_str}\n"
                            f"Here are the candidates who applied in this round: {candidate_pool}. "
                            "Choose a candidate and briefly explain why." # Respond only with the ID (e.g., C1)."
                        )
                    }
                    firm_messages[i].append(prompt)
                else:
                    prompt1 = {
                        "role": "assistant",
                        "content": (f"C{firm_results[i+1][0]}")
                    }
                    prompt2 = {
                        "role": "user", 
                        "content": (
                            f"In the last round, you hired C{firm_results[i+1][0]} and performance was {firm_results[i+1][1]}. "
                            f"Here are the results of other firms: {other_firm_result}"
                            f"Summary of hiring results across all rounds:\n{firm_summary_str}\n"
                            f"Here are the candidates who applied in this round: {candidate_pool}. "
                            "Choose a candidate and briefly explain why." # Respond only with the ID (e.g., C1)."
                        )
                    }
                    firm_messages[i].append(prompt1)
                    firm_messages[i].append(prompt2)

                firm_responses = firm_clients[i].chat.completions.create(
                    messages = firm_messages[i],
                    model="gpt-4o-mini",
                    # logprobs=True,
                    # max_tokens=2,
                    # top_logprobs=4
                )
                selected_candidate = firm_responses.choices[0].message.content.strip()
                # firm_messages[i].append(firm_responses.choices[0].message)
                print(f"[debug] firm message: {firm_messages[i]}")
                selected_candidates[i].append(selected_candidate)
                # scores = extract_scores_from_logprobs(firm_responses.choices[0].logprobs)
                # firm_writer.writerow({"round": round_num, "agent": f"F{i+1}", **scores})
                # time.sleep(10)
                logger.debug(f"[DEBUG] firm {i}, select {selected_candidates[i]}")
                print(f"[DEBUG] firm {i}, select {selected_candidates[i]}")
            else:
                # but do not ask
                prompt1 = {
                    "role": "assistant",
                    "content": (f"C{firm_results[i+1][0]}")
                }
                prompt2 = {
                    "role": "user", 
                    "content": (
                        f"In the last round, you hired C{firm_results[i+1][0]} and performance was {firm_results[i+1][1]}. "
                        f"Here are the results of other firms: {other_firm_result}"
                        f"Summary of hiring results across all rounds:\n{firm_summary_str}\n"
                        "In this round, there are no candidates who applied. You do not need to choose."
                    )
                }
                firm_messages[i].append(prompt1)
                firm_messages[i].append(prompt2)
                # firm_writer.writerow({"round": round_num, "agent": f"F{i+1}", "1":0, "2":0,"3":0, "4":0})

        print(f"[DEBUG] {selected_candidates}")

        # process the response
        for i, candidate_list in selected_candidates.items():
            if candidate_list:
                candidate_index = int(re.findall(r'C(\d+)', candidate_list[0])[-1]) - 1 if re.findall(r'C(\d+)', candidate_list[0]) else None
                # candidate_index = np.random.randint(0,4)
                logger.debug(f"Firm{i} hires Candidate{candidate_index}, feature: {candidate_vectors[i]['encoded_features']}")
                offers[i].append(candidate_index)
                all_offer_features[i].append(candidate_vectors[candidate_index]['encoded_features'][0])
            else:
                logger.debug(f"Firm{i} hires no candidate")
            

        ### step 3 Update results
        for firm_id, candidate_list in applications.items():
            firm_feature = firm_vectors[firm_id]
            if candidate_list == []:
                firm_results[firm_id+1][0] = 0
                firm_results[firm_id+1][1] = 0
            else:
                for i in candidate_list:
                    selected_candidates = candidate_vectors[i]['encoded_features']
                    if i in offers[firm_id]:
                        candidate_results[i+1][0] = firm_id+1
                        candidate_results[i+1][1] = 1
                        theta = np.full((2, 4), 2)
                        reward = np.sum(theta * selected_candidates * firm_feature.T) + np.random.normal(0,1)
                        firm_results[firm_id+1][0] = i+1
                        firm_results[firm_id+1][1] = reward
                        logger.debug(f"candidate: {i} applied_firm: {firm_feature} hired: 1")
                        update_hire_summary(i+1,firm_id+1,1,reward)
                    else:
                        candidate_results[i+1][0] = firm_id+1
                        candidate_results[i+1][1] = 0
                        logger.debug(f"candidate: {i} applied_firm: {firm_feature} hired: 0")
                        update_hire_summary(i+1,firm_id+1,0)
                
        ### Overtime result output
        logger.debug("---------------Offer Results-------------------")
        entropies_offer1 = []
        entropies_offer2 = []
        for firm_id in range(4):
            logger.debug(f"firm: {firm_id} initsel_features {matrix_print(candidate_vectors[firm_id]['encoded_features'][0])}")
            logger.debug(f"firm: {firm_id} select_features  {matrix_print(np.sum(all_offer_features[firm_id], axis=0))}")
            logger.debug("------------------------------------------------")
            # cal entropy
            entropy1 = cal_entropy(all_offer_features[firm_id])
            entropies_offer1.append(entropy1)
            entropy2 = calculate_entropy(all_offer_features[firm_id])
            entropies_offer2.append(entropy2)

        entropies1 = []
        entropies2 = []
        for firm_id, apply_feature in all_application_features.items():
            logger.debug(f"firm: {firm_id} initsel_features {matrix_print(candidate_vectors[firm_id]['encoded_features'][0])}")
            if apply_feature == []:
                entropies1.append(0.0)
                entropies2.append(0.0)
            else:
                logger.debug(f"firm: {firm_id} applied_features {matrix_print(np.sum(apply_feature, axis=0))}")
                # cal entropy
                entropy1 = cal_entropy(apply_feature)
                entropy2 = calculate_entropy(apply_feature)
                entropies1.append(entropy1)
                entropies2.append(entropy2)
            logger.debug("----------------------------------------------------")

        logger.debug(f"offers entropy:{np.mean(entropies_offer1)}")
        logger.debug(f"offers entropy all:{np.mean(entropies_offer2)}")
        logger.debug(f"Applicants entropy:{np.mean(entropies1)}")
        logger.debug(f"Applicants entropy all:{np.mean(entropies2)}")
        
        current_round += 1
        # restore the round result
        state_to_save = {
            "all_application_features": all_application_features,
            "all_offer_features": all_offer_features,
            "candidate_results": candidate_results,
            "firm_results": firm_results,
            "candidate_messages": candidate_messages,
            "firm_messages": firm_messages,
            "current_round": current_round,
            "hire_summary_candidate": hire_summary_candidate,
            "hire_summary_firm": hire_summary_firm
        }
        
        with open(state_file, "w") as f:
            json.dump(state_to_save, f, cls=NumpyEncoder)


    logger.debug("==================== Game End =======================")



async def run_experiment_async_distributed(candidate_file, firm_vector_file, logger):
    ### feature generation
    # Firm types
    # firm_types = ["technology", "management", "service", "production"]
    firm_types = ["type#1", "type#2"]
    # Load firm vectors
    firm_vectors = load_firm_vectors(firm_vector_file)
    # Generate descriptions
    firm_descriptions = generate_firm_descriptions(firm_vectors, firm_types)

    # Load feature template and generate candidates
    candidate_feature_template = load_feature_distributions(candidate_file)
    candidate_vectors = set_up_candidates(candidate_feature_template)

    ### initialization
    # Initialize OpenAI clients
    print(f"[DEBUG] Initialize OpenAI clients")
    candidate_clients = initialize_clients(4)
    firm_clients = initialize_clients(4)

    # Generate initial system prompts
    candidate_system_prompts, firm_system_prompts = generate_initial_prompts(candidate_vectors, firm_descriptions)

    # Initialize result tracking variables
    estimate_beta_a_matrix = [np.ones((4, 4), dtype=int) for _ in range(4)]
    estimate_beta_b_matrix = [np.ones((4, 4), dtype=int) for _ in range(4)]
    theta_matrix = np.zeros((2, 4), dtype=float)
    estimate_theta = [theta_matrix.flatten() for _ in range(4)]
    estimate_A = [np.identity(2 * 4) for _ in range(4)]
    estimate_b = [np.zeros(2 * 4) for _ in range(4)]
    all_application_features = {firm: [] for firm in range(4)}
    all_offer_features = {firm: [] for firm in range(4)}
    
    # initial hiring round
    for i, firm in enumerate(firm_vectors):
        selected_candidates = candidate_vectors[i]['encoded_features']
        logger.debug(f"firm: {i} initsel_features {matrix_print(selected_candidates[0])}")
        # update firm theta
        theta = np.full((2, 4), 2)
        reward = np.sum(theta * selected_candidates * firm.T) + 2.0
        estimate_theta[i], estimate_A[i], estimate_b[i] = update_firm(firm, selected_candidates, reward, estimate_A[i], estimate_b[i])
        all_offer_features[i].append(candidate_vectors[i]['encoded_features'][0])
        # update candidate beta matrix
        estimate_beta_a_matrix[i], estimate_beta_b_matrix[i] = update_candidate(i, selected_candidates,1, estimate_beta_a_matrix[i], estimate_beta_b_matrix[i])
        
    logger.debug("==================== Game Start =======================")
        
    ### iterate
    # Iterate through multiple rounds
    for round_num in range(NUM_ROUND):
        logger.debug(f"--- Round {round_num + 1} ---")
        print(f"--- Round {round_num + 1} ---")

        ### Step 1: Candidates apply to firms
        applications = {firm: [] for firm in range(4)}  # initialize applications
        applications_features = {firm: [] for firm in range(4)}  # initialize applications
        candidate_tasks = []
        # for i, candidate in enumerate(candidate_clients):
        for i in range(4):
            current_candidates = candidate_vectors[i]['encoded_features']
            observed_summary = candidate_observation_distributed(estimate_beta_a_matrix[i], estimate_beta_b_matrix[i], current_candidates)
            # prompt = {
            #     "role": "user", 
            #     "content": (
            #         f"Round {round_num + 1}: Based on results: {observed_summary}. " #Last round you tried F{i}.
            #         # "This is a Multi-Armed Bandit (MAB) problem where your goal is to maximize long-term rewards by choosing the best firms to apply to over many rounds. "
            #         "Do some explorations. Trying estimated low firms with fewer trials, it may get unexpected high rewards." #and exploitation (applying to firms with estimated higher success probability)."
            #         "Please choose only one firm (F1-F4) and show the firm ID (e.g., F1) at the end."
            #         "Respond with only the firm ID (e.g., F1)."
            #     )
            # }
            prompt = {
                "role": "user", 
                "content": (
                    f"Round {round_num + 1}: Based on results: {observed_summary}. " #Last round you tried F{i}.
                    "Choose one firm to apply. Remember to explore to uncover the groundtruth and maximize your *long-term* success."
                    # "Please choose only one firm (F1-F4) and show the firm ID (e.g., F1) at the end."
                    "Respond with only the firm ID (e.g., F1)."
                )
            }
            

            candidate_tasks.append(candidate_clients[i].chat.completions.create(
                messages=[candidate_system_prompts[i], prompt],
                model="gpt-4o-mini"
            ))

        # Use asyncio.gather with timeout
        try:
            candidate_responses = await asyncio.wait_for(
                asyncio.gather(*candidate_tasks), timeout=TIMEOUT
            )
        except asyncio.TimeoutError:
            print(f"[ERROR] Tasks did not complete within {TIMEOUT} seconds.")
            # Handle the timeout situation, e.g., retry, skip, log, etc.
            candidate_responses = None # Or handle as appropriate
        
        response_applications = [response.choices[0].message.content.strip() for response in candidate_responses]
        logger.debug(f"[DEBUG] {response_applications}")

        # process the response
        for i, firm_name in enumerate(response_applications):
            logger.debug(f"[DEBUG] candidate {i} {firm_name}")
            # Extract firm index from firm name (assuming "F1", "F2", etc.)
            firm_index = int(re.findall(r'F(\d+)', firm_name)[-1]) - 1 if re.findall(r'F(\d+)', firm_name) else None
            # firm_index = np.random.randint(0,4)
            logger.debug(f"Candidate{i} apply for Firm{firm_index}")
            selected_candidates = candidate_vectors[i]['features']
            candidate_feature = candidate_vectors[i]['encoded_features']
            applications_features[firm_index].append(f"C{i+1}:[{selected_candidates['category#1']}, {selected_candidates['category#2']}] ")
            applications[firm_index].append(i)
            # print(firm_index, applications_features[firm_index])
            all_application_features[firm_index].append(candidate_feature[0])

        print(f"[DEBUG] {applications}")
            

        ### Step 2: Firms select candidates
        offers = {firm: [] for firm in range(4)} 
        selected_candidates = {firm: [] for firm in range(4)}
        firm_tasks = []
        index_tasks = []
        for i in range(4):
            if len(applications_features[i]) > 1: # skip 0/1
                candidate_pool = applications_features[i]
                observed_summary = firm_observation_distributed(estimate_theta[i], estimate_A[i])
                # prompt = {
                #     "role": "user", 
                #     "content": (
                #         f"Round {round_num + 1}: Applied Candidates: {candidate_pool}. " #Last round you tried C{i}.
                #         f"Based on previous results: {observed_summary}. In last round, you tried C{i}"
                #         # "This is a Multi-Armed Bandit (MAB) problem where your goal is to maximize long-term performance by hiring the best candidates over many rounds. "
                #         "Do some explorations. Try estimated lower candidates with fewer trials, it may get unexpected high rewards." # and exploitation (selecting candidates with proven performance). "
                #         "Please choose only one candidate from Applied Candidates and show the candidate ID (e.g., C1) at the end."
                #         "Respond with only the candidate ID (e.g., C1)."
                #     )
                # }
                prompt = {
                    "role": "user", 
                    "content": (
                        f"Round {round_num + 1}: Applied Candidates: {candidate_pool}. "
                        f"Based on previous results: {observed_summary}. "
                        "Choose one candidate from applied candidates to hire."
                        "Remember to explore to uncover the groundtruth and maximize your *long-term* performance."
                        # "Please choose only one candidate from Applied Candidates and show the candidate ID (e.g., C1) at the end."
                        "Respond with only the candidate ID (e.g., C1)."
                    )
                }
                firm_tasks.append(firm_clients[i].chat.completions.create(
                    messages=[firm_system_prompts[i], prompt],
                    model="gpt-4o-mini"
                ))
                index_tasks.append(i)
            elif len(applications_features[i]) == 1:
                selected_candidates[i].append(applications_features[i][0].split(":")[0])

        # Use asyncio.gather with timeout
        try:
            firm_responses = await asyncio.wait_for(
                asyncio.gather(*firm_tasks), timeout=TIMEOUT
            )
        except asyncio.TimeoutError:
            print(f"[ERROR] Tasks did not complete within {TIMEOUT} seconds.")
            # Handle the timeout situation, e.g., retry, skip, log, etc.
            firm_responses = None # Or handle as appropriate
        for i, response in enumerate(firm_responses):
            selected_candidate = response.choices[0].message.content.strip()
            logger.debug(f"[DEBUG] firm {i} {selected_candidate}")
            selected_candidates[index_tasks[i]].append(selected_candidate)

        print(f"[DEBUG] {selected_candidates}")

        # process the response
        for i, candidate_list in selected_candidates.items():
            if candidate_list:
                candidate_index = int(re.findall(r'C(\d+)', candidate_list[0])[-1]) - 1 if re.findall(r'C(\d+)', candidate_list[0]) else None
                # candidate_index = np.random.randint(0,4)
                logger.debug(f"Firm{i} hires Candidate{candidate_index}, feature: {candidate_vectors[i]['encoded_features']}")
                offers[i].append(candidate_index)
                all_offer_features[i].append(candidate_vectors[candidate_index]['encoded_features'][0])
            else:
                logger.debug(f"Firm{i} hires no candidate")
             

        ### step 3 Update results
        for firm_id, candidate_list in applications.items():
            firm_feature = firm_vectors[firm_id]
            for i in candidate_list:
                selected_candidates = candidate_vectors[i]['encoded_features']
                if i in offers[firm_id]:
                    estimate_beta_a_matrix[i], estimate_beta_b_matrix[i] = update_candidate(i, selected_candidates,1, estimate_beta_a_matrix[i], estimate_beta_b_matrix[i])
                    theta = np.full((2, 4), 1.5)
                    reward = np.sum(theta * selected_candidates * firm_feature.T) + np.random.normal(0,1)
                    estimate_theta[firm_id], estimate_A[firm_id], estimate_b[firm_id] = update_firm(firm_feature, selected_candidates, reward, estimate_A[firm_id], estimate_b[firm_id])
                    logger.debug(f"candidate: {i} applied_firm: {firm_feature} hired: 1")
                else:
                    estimate_beta_a_matrix[i], estimate_beta_b_matrix[i] = update_candidate(i, selected_candidates,0, estimate_beta_a_matrix[i], estimate_beta_b_matrix[i])
                    logger.debug(f"candidate: {i} applied_firm: {firm_feature} hired: 0")
                
        ### Overtime result output
        logger.debug("---------------Offer Results-------------------")
        entropies_offer1 = []
        entropies_offer2 = []
        for firm_id in range(4):
            logger.debug(f"firm: {firm_id} initsel_features {matrix_print(candidate_vectors[firm_id]['encoded_features'][0])}")
            logger.debug(f"firm: {firm_id} select_features  {matrix_print(np.sum(all_offer_features[firm_id], axis=0))}")
            logger.debug("------------------------------------------------")
            # cal entropy
            entropy1 = cal_entropy(all_offer_features[firm_id])
            entropies_offer1.append(entropy1)
            entropy2 = calculate_entropy(all_offer_features[firm_id])
            entropies_offer2.append(entropy2)

        entropies1 = []
        entropies2 = []
        for firm_id, apply_feature in all_application_features.items():
            logger.debug(f"firm: {firm_id} initsel_features {matrix_print(candidate_vectors[firm_id]['encoded_features'][0])}")
            if apply_feature == []:
                entropies1.append(0.0)
                entropies2.append(0.0)
            else:
                logger.debug(f"firm: {firm_id} applied_features {matrix_print(np.sum(apply_feature, axis=0))}")
                # cal entropy
                entropy1 = cal_entropy(apply_feature)
                entropy2 = calculate_entropy(apply_feature)
                entropies1.append(entropy1)
                entropies2.append(entropy2)
            logger.debug("----------------------------------------------------")

        logger.debug(f"offers entropy:{np.mean(entropies_offer1)}")
        logger.debug(f"offers entropy all:{np.mean(entropies_offer2)}")
        logger.debug(f"Applicants entropy:{np.mean(entropies1)}")
        logger.debug(f"Applicants entropy all:{np.mean(entropies2)}")

    
    logger.debug("==================== Game End =======================")

# def run_experiment_distributed(candidate_file, firm_vector_file, logger):
#     ### feature generation
#     # Firm types
#     # firm_types = ["technology", "management", "service", "production"]
#     firm_types = ["type#1", "type#2"]
#     # Load firm vectors
#     firm_vectors = load_firm_vectors(firm_vector_file)
#     # Generate descriptions
#     firm_descriptions = generate_firm_descriptions(firm_vectors, firm_types)

#     # Load feature template and generate candidates
#     candidate_feature_template = load_feature_distributions(candidate_file)
#     candidate_vectors = set_up_candidates(candidate_feature_template)

#     ### initialization
#     # Initialize OpenAI clients
#     print(f"[DEBUG] Initialize OpenAI clients")
#     candidate_clients = initialize_clients_sync(4)
#     firm_clients = initialize_clients_sync(4)

#     # Generate initial system prompts
#     candidate_system_prompts, firm_system_prompts = generate_initial_prompts(candidate_vectors, firm_descriptions)

#     # Initialize result tracking variables
#     estimate_beta_a_matrix = [np.ones((4, 4), dtype=int) for _ in range(4)]
#     estimate_beta_b_matrix = [np.ones((4, 4), dtype=int) for _ in range(4)]
#     theta_matrix = np.zeros((2, 4), dtype=float)
#     estimate_theta = [theta_matrix.flatten() for _ in range(4)]
#     estimate_A = [np.identity(2 * 4) for _ in range(4)]
#     estimate_b = [np.zeros(2 * 4) for _ in range(4)]
#     all_application_features = {firm: [] for firm in range(4)}
#     all_offer_features = {firm: [] for firm in range(4)}
    
#     # initial hiring round
#     for i, firm in enumerate(firm_vectors):
#         selected_candidates = candidate_vectors[i]['encoded_features']
#         logger.debug(f"firm: {i} initsel_features {matrix_print(selected_candidates[0])}")
#         # update firm theta
#         theta = np.full((2, 4), 1.5)
#         reward = np.sum(theta * selected_candidates * firm.T) + 2.0
#         estimate_theta[i], estimate_A[i], estimate_b[i] = update_firm(firm, selected_candidates, reward, estimate_A[i], estimate_b[i])
#         all_offer_features[i].append(candidate_vectors[i]['encoded_features'][0])
#         # update candidate beta matrix
#         estimate_beta_a_matrix[i], estimate_beta_b_matrix[i] = update_candidate(i, selected_candidates,1, estimate_beta_a_matrix[i], estimate_beta_b_matrix[i])
        
#     logger.debug("==================== Game Start =======================")
        
#     ### iterate
#     # Iterate through multiple rounds
#     for round_num in range(NUM_ROUND):
#         logger.debug(f"--- Round {round_num + 1} ---")
#         print(f"--- Round {round_num + 1} ---")

#         ### Step 1: Candidates apply to firms
#         applications = {firm: [] for firm in range(4)}  # initialize applications
#         applications_features = {firm: [] for firm in range(4)}  # initialize applications
#         response_applications = []
#         # for i, candidate in enumerate(candidate_clients):
#         for i in range(4):
#             current_candidates = candidate_vectors[i]['encoded_features']
#             observed_summary = candidate_observation_distributed(estimate_beta_a_matrix[i], estimate_beta_b_matrix[i], current_candidates)
#             prompt = {
#                 "role": "user", 
#                 "content": (
#                     f"Round {round_num + 1}: Based on results: {observed_summary}. " #Last round you tried F{i}.
#                     # "This is a Multi-Armed Bandit (MAB) problem where your goal is to maximize long-term rewards by choosing the best firms to apply to over many rounds. "
#                     "Do some explorations. Trying estimated low firms with fewer trials, it may get unexpected high rewards." #and exploitation (applying to firms with estimated higher success probability)."
#                     "Please choose only one firm (F1-F4) and show the firm ID (e.g., F1) at the end."
#                     # "Respond with only the firm ID (e.g., F1)."
#                 )
#             }
#             candidate_response = candidate_clients[i].chat.completions.create(
#                 messages=[candidate_system_prompts[i], prompt],
#                 model="gpt-4o-mini"
#             )
#             response_applications.append(candidate_response.choices[0].message.content.strip())
#             logger.debug(f"[DEBUG] candidate {i}, response {response_applications[i]}")
#             time.sleep(10)


#         # process the response
#         for i, firm_name in enumerate(response_applications):
#             # Extract firm index from firm name (assuming "F1", "F2", etc.)
#             firm_index = int(re.findall(r'F(\d+)', firm_name)[-1]) - 1 if re.findall(r'F(\d+)', firm_name) else None
#             # firm_index = np.random.randint(0,4)
#             if firm_index < 0 or firm_index > 3:
#                 logger.debug(f"Candidate{i} apply for Firm out of index")
#             else:
#                 logger.debug(f"Candidate{i} apply for Firm{firm_index}")
#                 print(f"Candidate{i} apply for Firm{firm_index}")
#                 selected_candidates = candidate_vectors[i]['features']
#                 candidate_feature = candidate_vectors[i]['encoded_features']
#                 applications_features[firm_index].append(f"C{i+1}:[{selected_candidates['category#1']}, {selected_candidates['category#2']}] ")
#                 applications[firm_index].append(i)
#                 # print(firm_index, applications_features[firm_index])
#                 all_application_features[firm_index].append(candidate_feature[0])
        
#         print(f"[DEBUG] {applications}")
            

#         ### Step 2: Firms select candidates
#         offers = {firm: [] for firm in range(4)} 
#         selected_candidates = {firm: [] for firm in range(4)}
#         for i in range(4):
#             if len(applications_features[i]) > 1: # skip 0/1
#                 candidate_pool = applications_features[i]
#                 observed_summary = firm_observation_distributed(estimate_theta[i], estimate_A[i])
#                 prompt = {
#                     "role": "user", 
#                     "content": (
#                         f"Round {round_num + 1}: Applied Candidates: {candidate_pool}. " #Last round you tried C{i}.
#                         f"Based on previous results: {observed_summary}. In last round, you tried C{i}"
#                         # "This is a Multi-Armed Bandit (MAB) problem where your goal is to maximize long-term performance by hiring the best candidates over many rounds. "
#                         "Do some explorations. Try estimated lower candidates with fewer trials, it may get unexpected high rewards." # and exploitation (selecting candidates with proven performance). "
#                         "Please choose only one candidate from Applied Candidates and show the candidate ID (e.g., C1) at the end. Please respond concise."
#                         # "Respond with only the candidate ID (e.g., C1)."
#                     )
#                 }
#                 firm_responses = firm_clients[i].chat.completions.create(
#                     messages=[firm_system_prompts[i], prompt],
#                     model="gpt-4o-mini"
#                 )
#                 selected_candidate = firm_responses.choices[0].message.content.strip()
#                 selected_candidates[i].append(selected_candidate)
#                 time.sleep(10)
#                 logger.debug(f"[DEBUG] candidate {i}, select {selected_candidates[i]}")
#                 print(f"[DEBUG] candidate {i}, select {selected_candidates[i]}")
#             elif len(applications_features[i]) == 1:
#                 selected_candidates[i].append(applications_features[i][0].split(":")[0])


#         print(f"[DEBUG] {selected_candidates}")

#         # process the response
#         for i, candidate_list in selected_candidates.items():
#             if candidate_list:
#                 candidate_index = int(re.findall(r'C(\d+)', candidate_list[0])[-1]) - 1 if re.findall(r'C(\d+)', candidate_list[0]) else None
#                 # candidate_index = np.random.randint(0,4)
#                 logger.debug(f"Firm{i} hires Candidate{candidate_index}, feature: {candidate_vectors[i]['encoded_features']}")
#                 offers[i].append(candidate_index)
#                 all_offer_features[i].append(candidate_vectors[candidate_index]['encoded_features'][0])
#             else:
#                 logger.debug(f"Firm{i} hires no candidate")
             

#         ### step 3 Update results
#         for firm_id, candidate_list in applications.items():
#             firm_feature = firm_vectors[firm_id]
#             for i in candidate_list:
#                 selected_candidates = candidate_vectors[i]['encoded_features']
#                 if i in offers[firm_id]:
#                     estimate_beta_a_matrix[i], estimate_beta_b_matrix[i] = update_candidate(i, selected_candidates,1, estimate_beta_a_matrix[i], estimate_beta_b_matrix[i])
#                     theta = np.full((2, 4), 1.5)
#                     reward = np.sum(theta * selected_candidates * firm_feature.T) + np.random.normal(0,1)
#                     estimate_theta[firm_id], estimate_A[firm_id], estimate_b[firm_id] = update_firm(firm_feature, selected_candidates, reward, estimate_A[firm_id], estimate_b[firm_id])
#                     logger.debug(f"candidate: {i} applied_firm: {firm_feature} hired: 1")
#                 else:
#                     estimate_beta_a_matrix[i], estimate_beta_b_matrix[i] = update_candidate(i, selected_candidates,0, estimate_beta_a_matrix[i], estimate_beta_b_matrix[i])
#                     logger.debug(f"candidate: {i} applied_firm: {firm_feature} hired: 0")
                
#         ### Overtime result output
#         logger.debug("---------------Offer Results-------------------")
#         entropies_offer1 = []
#         entropies_offer2 = []
#         for firm_id in range(4):
#             logger.debug(f"firm: {firm_id} initsel_features {matrix_print(candidate_vectors[firm_id]['encoded_features'][0])}")
#             logger.debug(f"firm: {firm_id} select_features  {matrix_print(np.sum(all_offer_features[firm_id], axis=0))}")
#             logger.debug("------------------------------------------------")
#             # cal entropy
#             entropy1 = cal_entropy(all_offer_features[firm_id])
#             entropies_offer1.append(entropy1)
#             entropy2 = calculate_entropy(all_offer_features[firm_id])
#             entropies_offer2.append(entropy2)

#         entropies1 = []
#         entropies2 = []
#         for firm_id, apply_feature in all_application_features.items():
#             logger.debug(f"firm: {firm_id} initsel_features {matrix_print(candidate_vectors[firm_id]['encoded_features'][0])}")
#             if apply_feature == []:
#                 entropies1.append(0.0)
#                 entropies2.append(0.0)
#             else:
#                 logger.debug(f"firm: {firm_id} applied_features {matrix_print(np.sum(apply_feature, axis=0))}")
#                 # cal entropy
#                 entropy1 = cal_entropy(apply_feature)
#                 entropy2 = calculate_entropy(apply_feature)
#                 entropies1.append(entropy1)
#                 entropies2.append(entropy2)
#             logger.debug("----------------------------------------------------")

#         logger.debug(f"offers entropy:{np.mean(entropies_offer1)}")
#         logger.debug(f"offers entropy all:{np.mean(entropies_offer2)}")
#         logger.debug(f"Applicants entropy:{np.mean(entropies1)}")
#         logger.debug(f"Applicants entropy all:{np.mean(entropies2)}")

    
#     logger.debug("==================== Game End =======================")


async def run_experiment_async_decentralized(candidate_file, firm_vector_file, logger):
    ### feature generation
    # Firm types
    # firm_types = ["technology", "management", "service", "production"]
    firm_types = ["type#1", "type#2", "type#3", "type#4"]
    # Load firm vectors
    firm_vectors = load_firm_vectors(firm_vector_file)
    # Generate descriptions
    firm_descriptions = generate_firm_descriptions(firm_vectors, firm_types)

    # Load feature template and generate candidates
    candidate_feature_template = load_feature_distributions(candidate_file)
    candidate_vectors = set_up_candidates(candidate_feature_template)

    ### initialization
    # Initialize OpenAI clients
    print(f"[DEBUG] Initialize OpenAI clients")
    candidate_clients = initialize_clients(4)
    firm_clients = initialize_clients(4)

    # Generate initial system prompts
    candidate_system_prompts, firm_system_prompts = generate_initial_prompts(candidate_vectors, firm_descriptions)

    # Initialize result tracking variables
    estimate_beta_a_matrix = np.ones((4, 4), dtype=int)
    estimate_beta_b_matrix = np.ones((4, 4), dtype=int)
    theta_matrix = np.zeros((2, 4), dtype=float)
    estimate_theta = theta_matrix.flatten()
    estimate_A = np.identity(4 * 2)
    estimate_b = np.zeros(4 * 2)
    all_application_features = {firm: [] for firm in range(4)}
    all_offer_features = {firm: [] for firm in range(4)}
    
    # initial hiring round
    for i, firm in enumerate(firm_vectors):
        selected_candidates = candidate_vectors[i]['encoded_features']
        logger.debug(f"firm: {i} initsel_features {matrix_print(selected_candidates[0])}")
        # update firm theta
        theta = np.full((2, 4), 1.5)
        reward = np.sum(theta * selected_candidates * firm.T) + 2.0
        estimate_theta, estimate_A, estimate_b = update_firm(firm, selected_candidates, reward, estimate_A, estimate_b)
        all_offer_features[i].append(candidate_vectors[i]['encoded_features'][0])
        # update candidate beta matrix
        estimate_beta_a_matrix, estimate_beta_b_matrix = update_candidate(i, selected_candidates,1, estimate_beta_a_matrix, estimate_beta_b_matrix)
        
    logger.debug("==================== Game Start =======================")
        
    ### iterate
    # Iterate through multiple rounds
    for round_num in range(NUM_ROUND):
        logger.debug(f"--- Round {round_num + 1} ---")
        print(f"--- Round {round_num + 1} ---")

        ### Step 1: Candidates apply to firms
        applications = {firm: [] for firm in range(4)}  # initialize applications
        applications_features = {firm: [] for firm in range(4)}  # initialize applications
        candidate_tasks = []
        # for i, candidate in enumerate(candidate_clients):
        for i in range(4):
            current_candidates = candidate_vectors[i]['encoded_features']
            observed_summary = candidate_observation_distributed(estimate_beta_a_matrix, estimate_beta_b_matrix, current_candidates)
            # prompt = {
            #     "role": "user", 
            #     "content": (
            #         f"Round {round_num + 1}: Based on results: {observed_summary}. " #Last round you tried F{i}.
            #         # "This is a Multi-Armed Bandit (MAB) problem where your goal is to maximize long-term rewards by choosing the best firms to apply to over many rounds. "
            #         "Do some explorations. Trying estimated low firms with fewer trials, it may get unexpected high rewards." #and exploitation (applying to firms with estimated higher success probability)."
            #         "Please choose only one firm (F1-F4) and show the firm ID (e.g., F1) at the end."
            #         "Respond with only the firm ID (e.g., F1)."
            #     )
            # }
            prompt = {
                "role": "user", 
                "content": (
                    f"Round {round_num + 1}: Based on results: {observed_summary}. " #Last round you tried F{i}.
                    "Choose one firm to apply. Remember to explore to uncover the groundtruth and maximize your *long-term* success."
                    # "Please choose only one firm (F1-F4) and show the firm ID (e.g., F1) at the end."
                    "Respond with only the firm ID (e.g., F1)."
                )
            }
            candidate_tasks.append(candidate_clients[i].chat.completions.create(
                messages=[candidate_system_prompts[i], prompt],
                model="gpt-4o"
            ))

        # Use asyncio.gather with timeout
        try:
            candidate_responses = await asyncio.wait_for(
                asyncio.gather(*candidate_tasks), timeout=TIMEOUT
            )
        except asyncio.TimeoutError:
            print(f"[ERROR] Tasks did not complete within {TIMEOUT} seconds.")
            # Handle the timeout situation, e.g., retry, skip, log, etc.
            candidate_responses = None # Or handle as appropriate
        
        response_applications = [response.choices[0].message.content.strip() for response in candidate_responses]
        logger.debug(f"[DEBUG] {response_applications}")

        # process the response
        for i, firm_name in enumerate(response_applications):
            logger.debug(f"[DEBUG] candidate {i} {firm_name}")
            # Extract firm index from firm name (assuming "F1", "F2", etc.)
            firm_index = int(re.findall(r'F(\d+)', firm_name)[-1]) - 1 if re.findall(r'F(\d+)', firm_name) else None
            # firm_index = np.random.randint(0,4)
            if firm_index < 0 or firm_index > 59:
                logger.debug(f"Candidate{i} apply for Firm out of index")
            else:
                logger.debug(f"Candidate{i} apply for Firm{firm_index}")
                selected_candidates = candidate_vectors[i]['features']
                candidate_feature = candidate_vectors[i]['encoded_features']
                applications_features[firm_index].append(f"C{i+1}:[{selected_candidates['category#1']}, {selected_candidates['category#2']}] ")
                applications[firm_index].append(i)
                # print(firm_index, applications_features[firm_index])
                all_application_features[firm_index].append(candidate_feature[0])

        print(f"[DEBUG] {applications}")
            

        ### Step 2: Firms select candidates
        offers = {firm: [] for firm in range(4)} 
        selected_candidates = {firm: [] for firm in range(4)}
        firm_tasks = []
        index_tasks = []
        for i in range(4):
            if len(applications_features[i]) > 1: # skip 0/1
                candidate_pool = applications_features[i]
                observed_summary = firm_observation_distributed(estimate_theta, estimate_A)
                # prompt = {
                #     "role": "user", 
                #     "content": (
                #         f"Round {round_num + 1}: Applied Candidates: {candidate_pool}. " #Last round you tried C{i}.
                #         f"Based on previous results: {observed_summary}. In last round, you tried C{i}"
                #         # "This is a Multi-Armed Bandit (MAB) problem where your goal is to maximize long-term performance by hiring the best candidates over many rounds. "
                #         "Do some explorations. Try estimated lower candidates with fewer trials, it may get unexpected high rewards." # and exploitation (selecting candidates with proven performance). "
                #         "Please choose only one candidate from Applied Candidates and show the candidate ID (e.g., C1) at the end."
                #         "Respond with only the candidate ID (e.g., C1)."
                #     )
                # }
                prompt = {
                    "role": "user", 
                    "content": (
                        f"Round {round_num + 1}: Applied Candidates: {candidate_pool}. "
                        f"Based on previous results: {observed_summary}. "
                        "Choose one candidate from applied candidates to hire."
                        "Remember to explore to uncover the groundtruth and maximize your *long-term* performance."
                        # "Please choose only one candidate from Applied Candidates and show the candidate ID (e.g., C1) at the end."
                        "Respond with only the candidate ID (e.g., C1)."
                    )
                }
                firm_tasks.append(firm_clients[i].chat.completions.create(
                    messages=[firm_system_prompts[i], prompt],
                    model="gpt-4o"
                ))
                index_tasks.append(i)
            elif len(applications_features[i]) == 1:
                selected_candidates[i].append(applications_features[i][0].split(":")[0])

        # Use asyncio.gather with timeout
        try:
            firm_responses = await asyncio.wait_for(
                asyncio.gather(*firm_tasks), timeout=TIMEOUT
            )
        except asyncio.TimeoutError:
            print(f"[ERROR] Tasks did not complete within {TIMEOUT} seconds.")
            # Handle the timeout situation, e.g., retry, skip, log, etc.
            firm_responses = None # Or handle as appropriate
        for i, response in enumerate(firm_responses):
            selected_candidate = response.choices[0].message.content.strip()
            logger.debug(f"[DEBUG] firm {i} {selected_candidate}")
            selected_candidates[index_tasks[i]].append(selected_candidate)

        print(f"[DEBUG] {selected_candidates}")

        # process the response
        for i, candidate_list in selected_candidates.items():
            if candidate_list:
                candidate_index = int(re.findall(r'C(\d+)', candidate_list[0])[-1]) - 1 if re.findall(r'C(\d+)', candidate_list[0]) else None
                # candidate_index = np.random.randint(0,4)
                logger.debug(f"Firm{i} hires Candidate{candidate_index}, feature: {candidate_vectors[i]['encoded_features']}")
                offers[i].append(candidate_index)
                all_offer_features[i].append(candidate_vectors[candidate_index]['encoded_features'][0])
            else:
                logger.debug(f"Firm{i} hires no candidate")
             

        ### step 3 Update results
        for firm_id, candidate_list in applications.items():
            firm_feature = firm_vectors[firm_id]
            for i in candidate_list:
                selected_candidates = candidate_vectors[i]['encoded_features']
                if i in offers[firm_id]:
                    theta = np.full((2, 4), 1.5)
                    reward = np.sum(theta * selected_candidates * firm_feature.T) + np.random.normal(0,1)
                    estimate_theta, estimate_A, estimate_b = update_firm(firm_feature, selected_candidates, reward, estimate_A, estimate_b)
                    logger.debug(f"candidate: {i} applied_firm: {firm_feature} hired: 1")
                    for j in range(4):
                        search_string = f"feature#{j+1}"
                        if any(search_string in feature for feature in applications_features[firm_id]):
                            if selected_candidates[0][j] > 0:
                                estimate_beta_a_matrix[firm_id][j] +=1
                            else:
                                estimate_beta_b_matrix[firm_id][j] +=1
                else:
                    # estimate_beta_a_matrix[i], estimate_beta_b_matrix[i] = update_candidate(i, selected_candidates,0, estimate_beta_a_matrix[i], estimate_beta_b_matrix[i])
                    logger.debug(f"candidate: {i} applied_firm: {firm_feature} hired: 0")
                
        ### Overtime result output
        logger.debug("---------------Offer Results-------------------")
        entropies_offer1 = []
        entropies_offer2 = []
        for firm_id in range(4):
            logger.debug(f"firm: {firm_id} initsel_features {matrix_print(candidate_vectors[firm_id]['encoded_features'][0])}")
            logger.debug(f"firm: {firm_id} select_features  {matrix_print(np.sum(all_offer_features[firm_id], axis=0))}")
            logger.debug("------------------------------------------------")
            # cal entropy
            entropy1 = cal_entropy(all_offer_features[firm_id])
            entropies_offer1.append(entropy1)
            entropy2 = calculate_entropy(all_offer_features[firm_id])
            entropies_offer2.append(entropy2)

        entropies1 = []
        entropies2 = []
        for firm_id, apply_feature in all_application_features.items():
            logger.debug(f"firm: {firm_id} initsel_features {matrix_print(candidate_vectors[firm_id]['encoded_features'][0])}")
            if apply_feature == []:
                entropies1.append(0.0)
                entropies2.append(0.0)
            else:
                logger.debug(f"firm: {firm_id} applied_features {matrix_print(np.sum(apply_feature, axis=0))}")
                # cal entropy
                entropy1 = cal_entropy(apply_feature)
                entropy2 = calculate_entropy(apply_feature)
                entropies1.append(entropy1)
                entropies2.append(entropy2)
            logger.debug("----------------------------------------------------")

        logger.debug(f"offers entropy:{np.mean(entropies_offer1)}")
        logger.debug(f"offers entropy all:{np.mean(entropies_offer2)}")
        logger.debug(f"Applicants entropy:{np.mean(entropies1)}")
        logger.debug(f"Applicants entropy all:{np.mean(entropies2)}")

    
    logger.debug("==================== Game End =======================")


# def run_experiment_decentralized():
#     pass

### Run the experiment

# Get input parameters and set the number of iterations
input_mode = sys.argv[1]  # 'distributed' or 'decentralized'

# Select the appropriate function based on the input mode
if input_mode == 'distributed':
    experiment_function = run_experiment_distributed
elif input_mode == 'distributed_async':
    experiment_function = run_experiment_async_distributed
elif input_mode == 'distributed_grade':
    experiment_function = run_experiment_distributed_grade
elif input_mode == 'decentralized':
    experiment_function = run_experiment_decentralized
elif input_mode == 'decentralized_async':
    experiment_function = run_experiment_async_decentralized
elif input_mode == 'decentralized_grade':
    experiment_function = run_experiment_decentralized_grade
else:
    raise ValueError("Invalid input mode. Use 'distributed' or 'decentralized'.")

# Create a new directory with a timestamp and input mode to store log files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f'result_dir/log_{timestamp}_{input_mode}'
os.makedirs(log_dir, exist_ok=True)

current_turn = 0
# Main loop to run the experiment multiple times
for turn in range(current_turn, NUM_TURN):
    # Generate a log file name for each iteration
    log_filename = os.path.join(log_dir, f'{turn+1}.log')

    # Set up a logger specifically for this turn
    logger = logging.getLogger(f'{input_mode}_Run_{turn+1}')
    logger.setLevel(logging.DEBUG)
    
    # Create a file handler that logs to the specific file
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    
    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    
    # Log the start of the experiment
    logger.info(f'Starting turn {turn+1} with mode {input_mode}')
    
    # Run the experiment
    experiment_function('candidate-symbol-small.json', 'firms_vector-small.txt',logger, turn, log_dir)
    
    # Log the end of the experiment
    logger.info(f'Finished turn {turn}')
    
    # Remove handlers after logging is complete to prevent accumulation of handlers
    logger.removeHandler(file_handler)
    file_handler.close()
