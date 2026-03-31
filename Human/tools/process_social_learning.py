# filepath: process_social_learning.py

import os
import sys
import pandas as pd
from collections import defaultdict, Counter
import json

def compute_group_stats(df_all, current_index):
    try:
        current_index = int(current_index)
    except ValueError:
        return None, set()

    if current_index == 0:
        return None, set()

    prev_df = df_all[df_all['RoundIndex'].astype(int) == current_index - 1]

    majority_choice = None
    if not prev_df.empty:
        choice_counts = prev_df['decision'].dropna().value_counts()
        if not choice_counts.empty:
            majority_choice = choice_counts.idxmax()

    cumulative_rewards = df_all[df_all['RoundIndex'].astype(int) < current_index]
    reward_sums = cumulative_rewards.groupby('decision')['score'].sum()
    if not reward_sums.empty:
        max_reward = reward_sums.max()
        best_choices = set(reward_sums[reward_sums == max_reward].index)
    else:
        best_choices = set()
    
    return majority_choice, best_choices

def label_participant_social(df_participant, df_all):
    rows = []

    for i, row in df_participant.iterrows():
        round_num = row['RoundIndex']
        choice = row['decision']
        reward = row['score']

        last_choice = df_participant.iloc[i - 1]['decision'] if i > 0 else None
        last_reward = df_participant.iloc[i - 1]['score'] if i > 0 else None

        majority_choice, best_choices = compute_group_stats(df_all, round_num)
        
        if pd.isna(choice):
            judgment = "N/A"
        elif choice == majority_choice:
            judgment = "Majority-Biased"
        elif choice in best_choices:
            judgment = "Social-Exploit"
        else:
            judgment = "Contrarian"

        if last_choice is None or pd.isna(choice):
            pattern = "N/A"
        elif choice == last_choice and last_reward > 0:
            pattern = "Win-Stay"
        elif choice == last_choice and last_reward <= 0:
            pattern = "Lose-Stay"
        elif choice != last_choice and last_reward <= 0:
            pattern = "Lose-Shift"
        else:
            pattern = "Switch-After-Win"

        strategy_label = f"{judgment}+{pattern}"

        rows.append({
            "Round": round_num,
            "Choice": choice,
            "Reward": reward,
            "Last Choice": last_choice,
            "Last Reward": last_reward,
            "Majority Choice Prev": majority_choice,
            "Cumulative Best Choices": ", ".join(best_choices),
            "Judgment": judgment,
            "Pattern": pattern,
            "Strategy": strategy_label
        })

    return pd.DataFrame(rows)

def process_folder_social(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    input_path = os.path.join(input_folder, "playerRound.csv")
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    df_all = pd.read_csv(input_path)
    df_all['roundIDLastChangedAt'] = pd.to_datetime(df_all['roundIDLastChangedAt'])
    df_all = df_all.sort_values(by=['playerID', 'roundIDLastChangedAt'])

    df_all['RoundIndex'] = df_all.groupby('playerID').cumcount()

    for pid, df_part in df_all.groupby('playerID'):
        processed = label_participant_social(df_part.reset_index(drop=True), df_all)
        output_path = os.path.join(output_folder, f"{pid}_labeled.csv")
        processed.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

def process_all_subfolders_social(base_input_dir):
    for subfolder in os.listdir(base_input_dir):
        full_subfolder_path = os.path.join(base_input_dir, subfolder)
        if os.path.isdir(full_subfolder_path):
            output_path = os.path.join(full_subfolder_path, "analysis")
            process_folder_social(full_subfolder_path, output_path)

if __name__ == "__main__":
    base_input_dir = sys.argv[1]
    process_all_subfolders_social(base_input_dir)
