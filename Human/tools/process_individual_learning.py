# filepath: process_individual_learning.py

import os
import sys
import pandas as pd
from collections import defaultdict
import json

def compute_best_options(history):
    success_count = defaultdict(int)
    failure_count = defaultdict(int)
    success_ratio = defaultdict(lambda: [0, 0])

    for choice, reward in history:
        if pd.isna(choice):
            continue
        if reward > 0:
            success_count[choice] += 1
        else:
            failure_count[choice] += 1
        success_ratio[choice][0] += reward
        success_ratio[choice][1] += 1

    best_by_success = set()
    best_by_failure = set()
    best_by_ratio = set()

    if success_count:
        max_success = max(success_count.values())
        best_by_success = {k for k, v in success_count.items() if v == max_success}

    if failure_count:
        min_failure = min(failure_count.values())
        best_by_failure = {k for k, v in failure_count.items() if v == min_failure}

    ratio_values = {k: (v[0] / v[1]) if v[1] > 0 else -1 for k, v in success_ratio.items()}
    if ratio_values:
        max_ratio = max(ratio_values.values())
        best_by_ratio = {k for k, v in ratio_values.items() if v == max_ratio}

    return best_by_success, best_by_ratio, best_by_failure, dict(success_count), dict(failure_count), {k: (v[0] / v[1]) if v[1] > 0 else 0 for k, v in success_ratio.items()}

def label_participant(df_participant):
    rows = []
    history = []

    for i, row in df_participant.iterrows():
        round_num = row['RoundIndex']
        choice = row['decision']
        reward = row['score']

        last_choice = df_participant.iloc[i - 1]['decision'] if i > 0 else None
        last_reward = df_participant.iloc[i - 1]['score'] if i > 0 else None

        best_succ_set, best_ratio_set, best_fail_set, scount, fcount, sratio = compute_best_options(history)

        # Judgment strategy
        if pd.isna(choice):
            judgment = "N/A"
        elif choice in best_succ_set or choice in best_ratio_set or choice in best_fail_set:
            judgment = "Exploit"
        else:
            judgment = "Explore"

        # Behavioral pattern
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
            "Judgment": judgment,
            "Pattern": pattern,
            "Strategy": strategy_label,
            "Choice": choice,
            "Reward": reward,
            "Last Choice": last_choice,
            "Last Reward": last_reward,
            "Best highest success": ", ".join(best_succ_set),
            "Best highest ratio": ", ".join(best_ratio_set),
            "Best lowest failures": ", ".join(best_fail_set),
            "Success Count": json.dumps(scount),
            "Failure Count": json.dumps(fcount),
            "Success Ratio": json.dumps(sratio)
        })

        history.append((choice, reward))

    return pd.DataFrame(rows)

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    input_path = os.path.join(input_folder, "playerRound.csv")
    df = pd.read_csv(input_path)
    df['roundIDLastChangedAt'] = pd.to_datetime(df['roundIDLastChangedAt'])
    df = df.sort_values(by=['playerID', 'roundIDLastChangedAt'])

    df['RoundIndex'] = df.groupby('playerID').cumcount()

    for pid, group in df.groupby('playerID'):
        processed = label_participant(group.reset_index(drop=True))
        output_path = os.path.join(output_folder, f"{pid}_labeled.csv")
        processed.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

def process_all_subfolders(base_input_dir):
    for subfolder in os.listdir(base_input_dir):
        full_subfolder_path = os.path.join(base_input_dir, subfolder)
        if os.path.isdir(full_subfolder_path):
            output_path = os.path.join(full_subfolder_path, "analysis")
            process_folder(full_subfolder_path, output_path)

if __name__ == "__main__":
    base_input_dir = sys.argv[1]
    process_all_subfolders(base_input_dir)


