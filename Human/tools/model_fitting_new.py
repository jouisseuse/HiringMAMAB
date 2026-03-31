import os
import sys
import pandas as pd
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial

ARM_LIST = ["Indigo", "Pink", "Black", "Purple", "Slate", "Sky Blue", "Orange", "Crimson", "Amber", "Bright Green"]
ARM_MAP = {color: i for i, color in enumerate(ARM_LIST)}


def compute_own_stats(history):
    success = defaultdict(int)
    failure = defaultdict(int)
    for choice, reward in history:
        if pd.isna(choice):
            continue
        arm = ARM_MAP.get(choice, -1)
        if arm == -1:
            continue
        if reward > 0:
            success[arm] += 1
        else:
            failure[arm] += 1
    return success, failure


def compute_group_stats(group_df, round_num):
    round_df = group_df[group_df['RoundIndex'] < round_num]
    success = defaultdict(int)
    failure = defaultdict(int)
    for _, row in round_df.iterrows():
        arm = ARM_MAP.get(row['decision'], -1)
        if arm == -1:
            continue
        if row['score'] > 0:
            success[arm] += 1
        else:
            failure[arm] += 1
    return success, failure


def process_file(file_path, output_path, condition):
    df = pd.read_csv(file_path)
    df = df.sort_values(by='roundIDLastChangedAt')
    df['RoundIndex'] = df.groupby('playerID').cumcount()

    all_long = []
    for pid, group in df.groupby('playerID'):
        group = group.reset_index(drop=True)
        history = []
        for i in range(len(group)):
            row = group.loc[i]
            round_num = int(row['RoundIndex'])
            choice_color = row['decision']
            reward = row['score']

            arm_choice = ARM_MAP.get(choice_color, -1)
            if arm_choice == -1:
                continue

            prev_choice = ARM_MAP.get(group.loc[i - 1]['decision'], -1) if i > 0 else -1
            prev_reward = group.loc[i - 1]['score'] if i > 0 else 0

            own_succ, own_fail = compute_own_stats(history)
            group_succ, group_fail = compute_group_stats(df, round_num)

            for arm in range(10):
                all_long.append({
                    'participant_id': pid,
                    'round': round_num,
                    'arm': arm,
                    'choice': arm_choice,
                    'reward': reward,
                    'own_success': own_succ.get(arm, 0),
                    'own_failure': own_fail.get(arm, 0),
                    'group_success': group_succ.get(arm, 0),
                    'group_failure': group_fail.get(arm, 0),
                    'prev_choice': prev_choice,
                    'prev_reward': prev_reward,
                    'condition': condition
                })

            history.append((choice_color, reward))

    df_long = pd.DataFrame(all_long)
    df_long.to_csv(output_path, index=False)
    print(f"✅ Saved: {output_path}")


def process_subfolder(base_input_dir, subfolder):
    full_path = os.path.join(base_input_dir, subfolder)
    if not os.path.isdir(full_path):
        return

    file_path = os.path.join(full_path, "playerRound.csv")
    if not os.path.exists(file_path):
        return

    condition = 'social' if 'social' in subfolder.lower() else 'asocial'
    output_file = os.path.join(full_path, f"model_{subfolder}.csv")
    try:
        process_file(file_path, output_file, condition)
    except Exception as e:
        print(f"❌ Failed on {subfolder}: {e}")


def process_all_folders_parallel(base_input_dir, max_workers=4):
    subfolders = [f for f in os.listdir(base_input_dir)
                  if os.path.isdir(os.path.join(base_input_dir, f))]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(partial(process_subfolder, base_input_dir), subfolders)


if __name__ == "__main__":
    base_input_dir = sys.argv[1]
    max_workers = os.cpu_count()  # use all available cores
    process_all_folders_parallel(base_input_dir, max_workers=max_workers)