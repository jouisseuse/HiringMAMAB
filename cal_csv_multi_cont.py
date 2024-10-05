import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import sys

# 定义算法的参数
theta_params = [0.1, 0.3, 0.5, 0.9]
K = 4  # number of arms
T = 20  # number of rounds
CONFIDENCE_LEVEL = 0.95
Z = 1.96  # Z-score for 0.95 confidence interval

# 处理所有 agent 的文件
input_dir = sys.argv[1]
output_dir = os.path.join(input_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

# 获取所有结果 CSV 文件
csv_files = [f for f in os.listdir(input_dir) if f.endswith('_results.csv')]

# 计算每个 agent 的指标
def calculate_agent_metrics(df, agent_name, theta_params, K, T):
    metrics = []
    total_reward = 0
    greedy_choices = 0
    arm_counts = {i: 0 for i in range(1, K + 1)}
    # estimated_rewards = []  # 初始化每个臂的估计值
    arm_rewards = {i: [] for i in range(1, K + 1)}

    # 计算 Suffix Failure Frequency (SFF)
    suff_fail_list = calculate_suff_fail(df, agent_name, T)

    # 遍历每一轮并计算指标
    for index, row in df.iterrows():
        agent_choice = row[f'{agent_name}_choice']
        reward = row[f'{agent_name}_reward']
        total_reward += reward
        arm_counts[agent_choice] += 1
        arm_rewards[agent_choice].append(reward)

        # 更新每个臂的估计值
        # for arm in arm_rewards:
        #     if arm_rewards[arm]:
        #         estimated_rewards[arm] = np.mean(arm_rewards[arm])

        # 计算 MinFrac
        min_fraction = min(arm_counts.values()) / (index + 1) * K

        # 计算 GreedyFrac
        # estimated_rewards = [row['A1_estimate'],row['A2_estimate'],row['A3_estimate'],row['A4_estimate']]
        highest_estimated_arm = row[f'{agent_name}_highest'] 
        greedy = 1 if agent_choice == highest_estimated_arm else 0
        greedy_choices += greedy

        # 计算 Average Reward
        cumulative_reward = total_reward / (index + 1)
        all_time_averaged_rewards[agent_name].append(cumulative_reward)

        # 保存每轮的统计数据
        metrics.append({
            'Round': row['Round'],
            f'{agent_name}_Suffix Failure Frequency': suff_fail_list[index],
            f'{agent_name}_MinFrac': min_fraction,
            f'{agent_name}_GreedyFrac': greedy_choices / (index + 1),
            f'{agent_name}_Average Reward': cumulative_reward
        })

    return metrics

# 计算 Suffix Failure Frequency
def calculate_suff_fail(df, agent_name, T):
    suff_fail_values = [0] * (T+1)

    best_arm = best_arms[agent_name]
    
    # Iterate over each round t
    for t in range(0, T + 1):
        # Check if the best arm was ever chosen in rounds [t, T]
        t_half = t // 2
        pulled_best_arm = (df.loc[t_half:t, f'{agent_name}_choice'] == best_arm).any()
        suff_fail = 1 if not pulled_best_arm else 0
        suff_fail_values[t] = suff_fail
    
    return suff_fail_values

# 处理每个文件，并针对每个 agent 计算指标
def process_multi_agent_file(csv_file, theta_params, K, T):
    turnnum = csv_file.split('_')[0]
    df = pd.read_csv(os.path.join(input_dir, csv_file))

    # 保存所有 agent 的数据
    metrics_list = []
    agents = [f'Firm{i+1}' for i in range(4)]  # 假设有 4 个 agent: Firm1, Firm2, Firm3, Firm4

    # 分别处理每个 agent 的指标
    for agent in agents:
        metrics = calculate_agent_metrics(df, agent, theta_params, K, T)
        agent_metrics_df = pd.DataFrame(metrics)

        # 确保只保留一个 'Round' 列
        if 'Round' in agent_metrics_df.columns:
            agent_metrics_df = agent_metrics_df.drop(columns=['Round'], errors='ignore')
        
        # 将计算结果添加到 metrics_list
        metrics_list.append(agent_metrics_df)

    # 合并所有 agents 的 metrics 到一个 DataFrame 中
    combined_metrics_df = pd.concat(metrics_list, axis=1)
    combined_metrics_df.insert(0, 'Round', df['Round'])
    output_csv_file = os.path.join(output_dir, f"{turnnum}_multi_agent_metrics.csv")
    combined_metrics_df.to_csv(output_csv_file, index=False)

    return combined_metrics_df

# 汇总所有 agent 的 metrics
def calculate_aggregate_metrics(all_metrics, agents):
    # 合并所有 metrics 数据并按 'Round' 分组计算 mean、std 和 count
    combined_metrics = pd.concat(all_metrics, axis=0).groupby('Round').agg(['mean', 'std', 'count']).reset_index()

    # 展平列名，使其变为单层列名
    # combined_metrics.columns = ['_'.join(col).strip() if col[1] else col[0] for col in combined_metrics.columns.values]

    # 对每个 agent 的每个指标计算置信区间
    for agent in agents:
        for metric in [f'{agent}_Suffix Failure Frequency', f'{agent}_MinFrac', f'{agent}_GreedyFrac', f'{agent}_Average Reward']:
            mean_col = (metric, 'mean')
            std_col = (metric, 'std')
            count_col = (metric, 'count')

            # 计算置信区间
            combined_metrics[(metric, 'ci_lower')] = combined_metrics[mean_col] - Z * (combined_metrics[std_col] / np.sqrt(combined_metrics[count_col]))
            combined_metrics[(metric, 'ci_upper')] = combined_metrics[mean_col] + Z * (combined_metrics[std_col] / np.sqrt(combined_metrics[count_col]))

        print(f"{agent}:")
        print(f"Suffix Failure Frequency: {combined_metrics[(f'{agent}_Suffix Failure Frequency', 'mean')][T]}")
        print(f"MinFrac: {np.mean(combined_metrics[(f'{agent}_MinFrac', 'mean')])}")
        print(f"GreedyFrac: {combined_metrics[(f'{agent}_GreedyFrac', 'mean')][T]}")
        print(f"Median Average Reward: {rescale_and_compute_median(agent)}")

    return combined_metrics

def rescale_and_compute_median(agent_name):
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
    min_reward = np.min(all_time_averaged_rewards[agent_name])
    max_reward = np.max(all_time_averaged_rewards[agent_name])
    delta = max_reward - min_reward
    # print(max_reward,min_reward,delta)

    if delta == 0:
        # If all replicates have the same time-averaged reward, no rescaling is needed
        rescaled_rewards = np.zeros_like(all_time_averaged_rewards[agent_name])
    else:
        # Step 2: Rescale the time-averaged rewards to [0, 1]
        rescaled_rewards = (all_time_averaged_rewards[agent_name] - min_reward) / delta

    # Step 3: Compute the median of the rescaled values
    median_rescaled_reward = np.median(rescaled_rewards)
    # print(rescaled_rewards)

    return median_rescaled_reward

# 绘制带有置信区间的图表
def plot_with_confidence_intervals(combined_metrics, agents, output_dir):
    plt.figure(figsize=(12, 8))

    # 遍历每个 agent 绘制指标
    for i, agent in enumerate(agents, 1):
        plt.subplot(2, 2, i)
        plt.plot(combined_metrics.index, combined_metrics[(f'{agent}_Average Reward', 'mean')], marker='o')
        plt.fill_between(combined_metrics.index, combined_metrics[(f'{agent}_Average Reward', 'ci_lower')],
                         combined_metrics[(f'{agent}_Average Reward', 'ci_upper')], alpha=0.3)
        plt.title(f'{agent} - Average Reward with Confidence Interval')
        plt.xlabel('Round')
        plt.ylabel('Average Reward')

    plt.tight_layout()

    # 保存图像文件
    output_image_file = os.path.join(output_dir, 'multi_agent_metrics_with_confidence_intervals.png')
    plt.savefig(output_image_file)
    plt.close()


def plot_agent_metrics(metrics_df, agent_name, output_dir):
    plt.figure(figsize=(12, 8))

   # Suffix Failure Frequency (SFF) 绘图
    plt.subplot(2, 2, 1)
    plt.plot(metrics_df['Round'], metrics_df[(f'{agent_name}_Suffix Failure Frequency', 'mean')], marker='o', color='b')
    plt.title('Suffix Failure Frequency')
    plt.xlabel('Round')
    plt.ylabel('SFF')

    # MinFrac 绘图
    plt.subplot(2, 2, 2)
    plt.plot(metrics_df['Round'], metrics_df[(f'{agent_name}_MinFrac', 'mean')], marker='o')
    plt.fill_between(metrics_df['Round'], metrics_df[(f'{agent_name}_MinFrac', 'ci_lower')],
                     metrics_df[(f'{agent_name}_MinFrac', 'ci_upper')], alpha=0.3)
    plt.title('MinFrac (0.95 CI)')
    plt.xlabel('Round')
    plt.ylabel('MinFrac')

    # GreedyFrac 绘图
    plt.subplot(2, 2, 3)
    plt.plot(metrics_df['Round'], metrics_df[(f'{agent_name}_GreedyFrac', 'mean')], marker='o')
    plt.fill_between(metrics_df['Round'], metrics_df[(f'{agent_name}_GreedyFrac', 'ci_lower')],
                     metrics_df[(f'{agent_name}_GreedyFrac', 'ci_upper')], alpha=0.3)
    plt.title('GreedyFrac (0.95 CI)')
    plt.xlabel('Round')
    plt.ylabel('GreedyFrac')

    # Average Reward 绘图
    plt.subplot(2, 2, 4)
    plt.plot(metrics_df['Round'], metrics_df[(f'{agent_name}_Average Reward', 'mean')], marker='o')
    plt.fill_between(metrics_df['Round'], metrics_df[(f'{agent_name}_Average Reward', 'ci_lower')],
                     metrics_df[(f'{agent_name}_Average Reward', 'ci_upper')], alpha=0.3)
    plt.title('Average Reward (0.95 CI)')
    plt.xlabel('Round')
    plt.ylabel('Average Reward')

    # 保存图像文件
    plt.tight_layout()
    output_image_file = os.path.join(output_dir, f"{agent_name}_metrics.png")
    plt.savefig(output_image_file)
    plt.close()



# 处理每个文件的绘图
def process_plot_for_agents(metrics_df, output_dir):
    agents = [f'Firm{i+1}' for i in range(4)]  # 假设有 4 个 agent
    for agent in agents:
        plot_agent_metrics(metrics_df, agent, output_dir)

# 处理每个 CSV 文件并计算多 agent 指标
agents = [f'Firm{i+1}' for i in range(4)]  # 假设有 4 个 agent

best_arms = {
    'Firm1': 1,
    'Firm2': 1,
    'Firm3': 4,
    'Firm4': 4
}

all_time_averaged_rewards = { agent : [] for agent in agents}
all_metrics = []
for csv_file in csv_files:
    combined_metrics_df = process_multi_agent_file(csv_file, theta_params, K, T)
    all_metrics.append(combined_metrics_df)

# 计算汇总指标
combined_metrics = calculate_aggregate_metrics(all_metrics, agents)
# print(combined_metrics.columns)

# 绘制带有置信区间的图表
process_plot_for_agents(combined_metrics, output_dir)