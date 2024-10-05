import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

# Define the parameters for the LLM algorithm
theta_params = [0.1, 0.3, 0.5, 0.9]
K = 4
T = 20
CONFIDENCE_LEVEL = 0.95
Z = 1.96  # Z-score for 0.95 confidence interval

# Read the data
input_dir = sys.argv[1]
output_dir = os.path.join(input_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

# get all csv files in the directory
csv_files = [f for f in os.listdir(input_dir) if f.endswith('_results.csv')]

all_metrics = []
def process_file_ref(csv_file, K):
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
    arm_rewards = {i: [] for i in range(1, K + 1)}  # 用于记录每个臂的历史奖励
    suff_fail_list = [] 

    metrics = []

    # 1. 计算 Suffix Failure Frequency (SFF)
    suff_fail_list = calculate_suff_fail(df, T)

    # 遍历每一轮并计算指标
    for index, row in df.iterrows():
        print(index)
        llm_choice = row['LLM Choice']
        reward = row['LLM Reward']
        total_reward += reward
        arm_counts[llm_choice] += 1
        arm_rewards[llm_choice].append(reward)

        # 2. 计算 MinFrac
        min_fraction = min(arm_counts.values()) / (index + 1) * K

        # 3. 计算 GreedyFrac
        highest_estimated_arm = row['Estimated highest']
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
    median_rescaled_reward = np.median(all_time_averaged_rewards)
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
    best_arm = 1
    
    # Iterate over each round t
    for t in range(0, T + 1):
        # Check if the best arm was ever chosen in rounds [t, T]
        t_half = t // 2
        pulled_best_arm = (df.loc[t_half:t, 'LLM Choice'] == best_arm).any()
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
        llm_choice = row['LLM Choice']
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
    # process_file(csv_file, theta_params)
    process_file_ref(csv_file,K)


combined_metrics = calculate_aggregate_metrics(all_metrics)
plot_with_confidence_intervals(combined_metrics, output_dir)
