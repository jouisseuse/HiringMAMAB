import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

# Define the parameters for the LLM algorithm
theta_params = [0.1, 0.3, 0.5, 0.9]
K = 4
T = 60
CONFIDENCE_LEVEL = 0.95
Z = 1.96  # Z-score for 0.95 confidence interval

# Read the data
input_dir = sys.argv[1]
output_dir = os.path.join(input_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

# get all csv files in the directory
csv_files = [f for f in os.listdir(input_dir) if f.endswith('_results.csv')]

all_metrics = []

def process_file(csv_file, theta_params):
    turnnum = csv_file.split('_')[0]
    df = pd.read_csv(os.path.join(input_dir, csv_file))

    csv_file_overall = os.path.join(output_dir, f"{turnnum}_overall_arms.png")
    csv_file_round = os.path.join(output_dir, f"{turnnum}_round_arms.png")
    csv_file_regret = os.path.join(output_dir, f"{turnnum}_regret_LLM.png")
    csv_file_explore_exploit = os.path.join(output_dir, f"{turnnum}_llm_explore_exploit.png")
    csv_file_explore_exploit_rounds = os.path.join(output_dir, f"{turnnum}_llm_explore_exploit_rounds.png")

    ### 1. Plot cumulative arm selection for each algorithm (bar chart)
    cumulative_selection = pd.DataFrame({
        'LLM': df['LLM'].value_counts(),
        'UCB': df['UCB'].value_counts(),
        'ThompsonSampling': df['ThompsonSampling'].value_counts(),
        'EGreedy': df['EGreedy'].value_counts()
    }).fillna(0)

    # Transpose the dataframe so each strategy has its own group of arms (A1, A2, A3, A4)
    cumulative_selection = cumulative_selection.T

    # Create the bar chart
    cumulative_selection.plot(kind='bar', stacked=False, figsize=(10, 6))
    plt.title(f'Cumulative Arm Selection for Each Strategy (Turn {turnnum})')
    plt.xlabel('Strategy')
    plt.ylabel('Cumulative Selections')
    plt.xticks(rotation=0)
    plt.legend(title='Arms', loc='upper right')
    plt.tight_layout()

    # Show the plot
    plt.savefig(csv_file_overall)
    plt.close()

    ### 2. Plot the choices over time (line plot)
    strategies = ['LLM', 'UCB', 'ThompsonSampling', 'EGreedy']
    df.plot(x='Round', y=strategies, kind='line', marker='o', figsize=(10, 6))
    plt.title(f'Arm Choices Over Time (Turn {turnnum})')
    plt.xlabel('Round')
    plt.ylabel('Chosen Arm')
    plt.legend(title='Algorithm')
    plt.tight_layout()

    # Show the second plot
    plt.savefig(csv_file_round)
    plt.close()

    for strategy in strategies:
        df.plot(x='Round', y=strategy, kind='line', marker='o', figsize=(10, 6))
        plt.title(f'Arm Choices Over Time for {strategy} (Turn {turnnum})')
        plt.xlabel('Round')
        plt.ylabel('Chosen Arm')
        plt.legend(title='Algorithm')
        plt.tight_layout()

        # Show the second plot
        csv_file_name = os.path.join(output_dir, f"{turnnum}_round_arms_{strategy}.png")
        plt.savefig(csv_file_name)
        plt.close()

    ### 3. calculate regret
    def calculate_regret(df, strategy_column, theta_params):
        optimal_reward = max(theta_params)
        df[f'{strategy_column}_Expected_Reward'] = df[strategy_column].apply(lambda x: theta_params[x - 1])
        df[f'{strategy_column}_Regret'] = optimal_reward - df[f'{strategy_column}_Expected_Reward']
        df[f'{strategy_column}_Cumulative_Regret'] = df[f'{strategy_column}_Regret'].cumsum()

    calculate_regret(df, 'LLM', theta_params)
    plt.figure(figsize=(10, 6))
    plt.plot(df['Round'], df['LLM_Cumulative_Regret'], marker='o', label='LLM')
    plt.title(f'Cumulative Regret Over Time for LLM (Turn {turnnum})')
    plt.xlabel('Round')
    plt.ylabel('Cumulative Regret')
    plt.legend(title='LLM')
    plt.tight_layout()
    plt.savefig(csv_file_regret)
    plt.close()

    ### 4. indicator of exploration and exploitation
    arm_history = {i: [] for i in range(1, len(theta_params) + 1)}
    initial_estimates = {i: 0.0 for i in range(1, len(theta_params) + 1)}
    arm_selection_count = {i: 0 for i in range(1, len(theta_params) + 1)}

    # update the estimates based on the history of rewards
    def update_estimates(arm_history, initial_estimates, chosen_arm, reward):
        arm_selection_count[chosen_arm] += 1
        n = arm_selection_count[chosen_arm]

        old_value = initial_estimates[chosen_arm]
        new_value = old_value + (reward - old_value) / n
        initial_estimates[chosen_arm] = new_value

        return initial_estimates

    # iterate through each row of the dataframe and update the estimates
    explore_exploit = []

    for index, row in df.iterrows():
        llm_choice = row['LLM']
        reward = row['LLM Reward']
        estimates = update_estimates(arm_history, initial_estimates, llm_choice, reward)
        highest_estimated_arm = max(estimates, key=estimates.get)

        # check if LLM chose the highest estimated arm
        if llm_choice == highest_estimated_arm:
            decision = 'Exploitation'
            decision_numeric = 1
        else:
            decision = 'Exploration'
            decision_numeric = 0

        explore_exploit.append({
            'Round': row['Round'],
            'LLM_Choice': llm_choice,
            'Highest_Estimated_Arm': highest_estimated_arm,
            'LLM_Decision': decision,
            'LLM_Decision_Numeric': decision_numeric,
            'A1_Estimate': estimates[1],
            'A2_Estimate': estimates[2],
            'A3_Estimate': estimates[3],
            'A4_Estimate': estimates[4]
        })

        arm_history[llm_choice].append(row['LLM Reward'])

    explore_exploit_df = pd.DataFrame(explore_exploit)

    # save the updated dataframe to a new CSV file
    df = df.merge(explore_exploit_df[['Round', 'Highest_Estimated_Arm', 'LLM_Decision', 'LLM_Decision_Numeric', 'A1_Estimate', 'A2_Estimate', 'A3_Estimate', 'A4_Estimate']], on='Round')
    output_csv_file = os.path.join(output_dir, f"{turnnum}_llm_exploration_exploitation.csv")
    df.to_csv(output_csv_file, index=False)

    # visualization for bar chart
    plt.figure(figsize=(10, 6))
    explore_counts = df['LLM_Decision'].value_counts()
    explore_counts.plot(kind='bar', color=['#1f77b4', '#ff7f0e'], rot=0)
    plt.title(f'LLM Decisions: Exploration vs Exploitation (Turn {turnnum})')
    plt.xlabel('Decision Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(csv_file_explore_exploit)
    plt.close()

    # visualization for line chart
    plt.figure(figsize=(10, 6))
    plt.plot(df['Round'], df['LLM_Decision_Numeric'], marker='o', linestyle='-', color='#1f77b4')
    plt.title(f'Exploration vs Exploitation Decisions Over Time (Turn {turnnum})')
    plt.xlabel('Round')
    plt.ylabel('Decision (0 = Exploration, 1 = Exploitation)')
    plt.yticks([0, 1], ['Exploration', 'Exploitation'])
    plt.tight_layout()
    plt.savefig(csv_file_explore_exploit_rounds)
    plt.close()

def process_file_ref(csv_file, theta_params, K):
    # 提取 turnnum 作为前缀
    turnnum = csv_file.split('_')[0]
    df = pd.read_csv(os.path.join(input_dir, csv_file))

    # 输出文件的路径
    output_csv_file = os.path.join(output_dir, f"{turnnum}_metrics.csv")

    # 初始化统计数据
    total_rounds = len(df)
    total_reward = 0
    greedy_choices = 0
    # failure_choices = 0
    # minfrac_choices = 0
    arm_counts = {i: 0 for i in range(1, K + 1)}  # 记录每个臂的选择次数
    estimated_rewards = {i: 0.0 for i in range(1, K + 1)}  # 初始化每个臂的估计值为 0.5
    arm_rewards = {i: [] for i in range(1, K + 1)}  # 用于记录每个臂的历史奖励
    suff_fail_list = [] 

    metrics = []

    # 1. 计算 Suffix Failure Frequency (SFF)
    suff_fail_list = calculate_suff_fail(df, T)

    # 遍历每一轮并计算指标
    for index, row in df.iterrows():
        print(index)
        llm_choice = row['LLM']
        reward = row['LLM Reward']
        total_reward += reward
        arm_counts[llm_choice] += 1
        arm_rewards[llm_choice].append(reward)

        # update
        for arm in arm_rewards:
            if arm_rewards[arm]:
                estimated_rewards[arm] = np.mean(arm_rewards[arm]) 

        # 2. 计算 MinFrac
        min_fraction = min(arm_counts.values()) / (index + 1) * K

        # 3. 计算 GreedyFrac
        highest_estimated_arm = max(estimated_rewards, key=estimated_rewards.get)
        greedy = 1 if llm_choice == highest_estimated_arm else 0
        greedy_choices += greedy

        # 4. 计算 Average Reward
        cumulative_reward = total_reward / (index + 1)

        # 保存每轮的统计数据
        metrics.append({
            'Round': row['Round'],
            'Suffix Failure Frequency': suff_fail_list[index],
            'MinFrac': min_fraction,
            'GreedyFrac': greedy_choices / (index + 1),
            'Average Reward': cumulative_reward
        })

    # 转换为 DataFrame 并保存到新的 CSV 文件
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_csv_file, index=False)

    time_averaged_reward = cumulative_reward
    all_time_averaged_rewards.append(time_averaged_reward)


    # 计算 Uniform-like Failure MinFrac(t)
    # minfrac_values = calculate_minfrac(df,K)

    # 可视化这些指标
    plt.figure(figsize=(12, 8))

    # Suffix Failure Frequency (SFF) 绘图
    plt.subplot(2, 2, 1)
    plt.plot(metrics_df['Round'], metrics_df['Suffix Failure Frequency'], marker='o')
    plt.title(f'Suffix Failure Frequency (Turn {turnnum})')
    plt.xlabel('Round')
    plt.ylabel('SFF')

    # # MinFrac 绘图
    plt.subplot(2, 2, 2)
    plt.plot(metrics_df['Round'], metrics_df['MinFrac'], marker='o')
    plt.title(f'MinFrac (Turn {turnnum})')
    plt.xlabel('Round')
    plt.ylabel('MinFrac')

    # MinFrac(t) for uniform-like failures 绘图
    # plt.subplot(2, 2, 2)
    # plt.plot(range(1, total_rounds + 1), minfrac_values, marker='o')
    # plt.title(f'Uniform-like Failure MinFrac(t) (Turn {turnnum})')
    # plt.xlabel('Round')
    # plt.ylabel('K · MinFrac(t)')

    # GreedyFrac 绘图
    plt.subplot(2, 2, 3)
    plt.plot(metrics_df['Round'], metrics_df['GreedyFrac'], marker='o')
    plt.title(f'GreedyFrac (Turn {turnnum})')
    plt.xlabel('Round')
    plt.ylabel('GreedyFrac')

    # Average Reward 绘图
    plt.subplot(2, 2, 4)
    plt.plot(metrics_df['Round'], metrics_df['Average Reward'], marker='o')
    plt.title(f'Average Reward (Turn {turnnum})')
    plt.xlabel('Round')
    plt.ylabel('Average Reward')

    # 保存图像文件
    plt.tight_layout()
    output_image_file = os.path.join(output_dir, f"{turnnum}_metrics.png")
    plt.savefig(output_image_file)
    plt.close()

    all_metrics.append(pd.DataFrame(metrics))

def rescale_and_compute_median():
    """
    Rescale the time-averaged total rewards across all replicates to [0, 1],
    and compute the median of the rescaled values.

    Args:
        all_time_averaged_rewards (list): A list of time-averaged total rewards for all replicates.

    Returns:
        float: The rescaled median of the time-averaged total rewards.
    """
    # Step 1: Compute ∆ (delta) which is the range of Φ(R) values across all replicates
    # print(all_time_averaged_rewards)
    min_reward = np.min(all_time_averaged_rewards)
    max_reward = np.max(all_time_averaged_rewards)
    delta = max_reward - min_reward
    # print(max_reward,min_reward,delta)

    if delta == 0:
        # If all replicates have the same time-averaged reward, no rescaling is needed
        rescaled_rewards = np.zeros_like(all_time_averaged_rewards)
    else:
        # Step 2: Rescale the time-averaged rewards to [0, 1]
        rescaled_rewards = (all_time_averaged_rewards - min_reward) / delta

    # Step 3: Compute the median of the rescaled values
    median_rescaled_reward = np.median(rescaled_rewards)
    # print(rescaled_rewards)

    return median_rescaled_reward

def calculate_median_reward(reward_matrix, T):
    """
    Calculate the rescaled MedianReward for the time-averaged total reward.

    Args:
        reward_matrix: A 2D array where each row is a replicate and each column is the reward for that round.
        delta: The deviation ∆ for the expected value range.
        T: Total number of rounds.

    Returns:
        median_reward: The rescaled median of the time-averaged total reward.
    """
    # Step 1: Calculate Φ(R) for each replicate (Time-averaged total reward for each replicate)
    phi_R = np.mean(reward_matrix, axis=1)  # Average reward per round for each replicate

    # Step 2: Rescale Φ(R) from [1/2 - ∆/2, 1/2 + ∆/2] to [0, 1]
    delta = np.max(phi_R) - np.min(phi_R)
    if delta == 0:
        # If all replicates have the same time-averaged reward, no rescaling is needed
        rescaled_phi_R = np.zeros_like(phi_R)  # All values should be zero since they map to [0, 0]
    else:
        # Step 3: Rescale Φ(R) from [min(Φ(R)), max(Φ(R))] to [0, 1]
        rescaled_phi_R = (phi_R - np.min(phi_R)) / delta
    
    # Step 3: Calculate the median of the rescaled values
    median_reward = np.median(rescaled_phi_R)
    
    return median_reward

# 计算每一轮的 Suffix Failure (SuffFail(t, R)) 对应的0/1值
def calculate_suff_fail(df, T):
    """
    Calculate SuffFail(t, R) for each round t and replicate R.
    SuffFail(t, R) is 1 if the best arm is never chosen from round t to T.
    
    Args:
        df: DataFrame for a single replicate with columns ['Round', 'LLM'].
        T: Total number of rounds.
        
    Returns:
        suff_fail_values: List of SuffFail(t, R) for each round t.
    """
    suff_fail_values = [0] * (T+1)
    best_arm = np.argmax(theta_params) + 1
    
    # Iterate over each round t
    for t in range(0, T + 1):
        # Check if the best arm was ever chosen in rounds [t, T]
        t_half = t // 2
        pulled_best_arm = (df.loc[t_half:t, 'LLM'] == best_arm).any()
        suff_fail = 1 if not pulled_best_arm else 0
        suff_fail_values[t] = suff_fail
    
    return suff_fail_values

# 计算 MinFrac(t) 用于 Uniform-like failures
def calculate_minfrac(df,K):
    total_rounds = len(df)
    arm_counts = {i: 0 for i in range(1, K + 1)}  # Initialize counts for each arm

    minfrac_values = []

    # Iterate through each round
    for index, row in df.iterrows():
        llm_choice = row['LLM']
        arm_counts[llm_choice] += 1

        # Calculate the fraction of times each arm was chosen
        arm_fractions = {arm: count / (index + 1) for arm, count in arm_counts.items()}

        # Find MinFrac(t, R) = min(arm_fractions)
        minfrac_t_r = min(arm_fractions.values())

        # Rescale MinFrac(t, R) by K
        rescaled_minfrac_t_r = K * minfrac_t_r

        # Store the rescaled value for the current round
        minfrac_values.append(rescaled_minfrac_t_r)

    return minfrac_values

def calculate_aggregate_metrics(all_metrics):
    # 合并所有 replicates 的数据
    combined_metrics = pd.concat(all_metrics, axis=0).groupby('Round').agg(['mean', 'std', 'count'])

    # 计算 0.95 置信区间
    for metric in ['Suffix Failure Frequency', 'MinFrac', 'GreedyFrac', 'Average Reward']:
        mean_col = (metric, 'mean')
        std_col = (metric, 'std')
        count_col = (metric, 'count')

        combined_metrics[(metric, 'ci_lower')] = combined_metrics[mean_col] - Z * (combined_metrics[std_col] / np.sqrt(combined_metrics[count_col]))
        combined_metrics[(metric, 'ci_upper')] = combined_metrics[mean_col] + Z * (combined_metrics[std_col] / np.sqrt(combined_metrics[count_col]))

    print(f"Suffix Failure Frequency: {combined_metrics[('Suffix Failure Frequency', 'mean')][T]}")
    print(f"MinFrac: {np.mean(combined_metrics[('MinFrac', 'mean')])}")
    print(f"GreedyFrac: {combined_metrics[('GreedyFrac', 'mean')][T]}")
    print(f"Median Average Reward: {rescale_and_compute_median()}")

    return combined_metrics

# 绘制带有置信区间的图表
def plot_with_confidence_intervals(combined_metrics, output_dir):
    plt.figure(figsize=(12, 8))

    # Suffix Failure Frequency (SFF) 绘图
    plt.subplot(2, 2, 1)
    plt.plot(combined_metrics.index, combined_metrics[('Suffix Failure Frequency', 'mean')], marker='o', color='b')
    plt.title('Suffix Failure Frequency')
    plt.xlabel('Round')
    plt.ylabel('SFF')

    # MinFrac 绘图
    plt.subplot(2, 2, 2)
    plt.plot(combined_metrics.index, combined_metrics[('MinFrac', 'mean')], marker='o')
    plt.fill_between(combined_metrics.index, combined_metrics[('MinFrac', 'ci_lower')],
                     combined_metrics[('MinFrac', 'ci_upper')], alpha=0.3)
    plt.title('MinFrac (0.95 CI)')
    plt.xlabel('Round')
    plt.ylabel('MinFrac')

    # GreedyFrac 绘图
    plt.subplot(2, 2, 3)
    plt.plot(combined_metrics.index, combined_metrics[('GreedyFrac', 'mean')], marker='o')
    plt.fill_between(combined_metrics.index, combined_metrics[('GreedyFrac', 'ci_lower')],
                     combined_metrics[('GreedyFrac', 'ci_upper')], alpha=0.3)
    plt.title('GreedyFrac (0.95 CI)')
    plt.xlabel('Round')
    plt.ylabel('GreedyFrac')

    # Average Reward 绘图
    plt.subplot(2, 2, 4)
    plt.plot(combined_metrics.index, combined_metrics[('Average Reward', 'mean')], marker='o')
    plt.fill_between(combined_metrics.index, combined_metrics[('Average Reward', 'ci_lower')],
                     combined_metrics[('Average Reward', 'ci_upper')], alpha=0.3)
    plt.title('Average Reward (0.95 CI)')
    plt.xlabel('Round')
    plt.ylabel('Average Reward')

    plt.tight_layout()

    # 保存图像文件
    output_image_file = os.path.join(output_dir, 'aggregate_metrics_with_confidence_intervals.png')
    plt.savefig(output_image_file)
    plt.close()

all_time_averaged_rewards = []
for csv_file in csv_files:
    process_file(csv_file, theta_params)
    process_file_ref(csv_file, theta_params,K)


combined_metrics = calculate_aggregate_metrics(all_metrics)
plot_with_confidence_intervals(combined_metrics, output_dir)
