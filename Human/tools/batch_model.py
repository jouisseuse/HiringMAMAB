import os
import sys
import pandas as pd
import numpy as np

def thompson_sampled_best_arms(successes, failures):
    arms = list(set(successes.keys()) | set(failures.keys()))
    if not arms:
        return set()

    samples = {
        arm: np.random.beta(successes.get(arm, 0) + 1, failures.get(arm, 0) + 1)
        for arm in arms
    }

    max_value = max(samples.values())
    best_arms = {arm for arm, val in samples.items() if val == max_value}
    return best_arms

def compute_strategy_shares(df_long):
    results = []

    for (pid, rnd), group in df_long.groupby(['participant_id', 'round']):
        if group.empty:
            continue

        chosen_arm = int(group['choice'].iloc[0])
        prev_choice = group['prev_choice'].iloc[0]
        prev_reward = group['prev_reward'].iloc[0]

        # Build arm → success/failure dictionaries
        own_success = group.set_index('arm')['own_success'].to_dict()
        own_failure = group.set_index('arm')['own_failure'].to_dict()
        group_success = group.set_index('arm')['group_success'].to_dict()
        group_failure = group.set_index('arm')['group_failure'].to_dict()

        # Thompson Sampling
        best_own = thompson_sampled_best_arms(own_success, own_failure)
        best_group = thompson_sampled_best_arms(group_success, group_failure)

        is_own = int(chosen_arm in best_own)
        is_group = int(chosen_arm in best_group)
        is_prev = int(prev_choice == chosen_arm and prev_reward > 0)
        is_others = int(max(is_own, is_group, is_prev) == 0)

        results.append({
            'participant_id': pid,
            'round': rnd,
            'is_own': is_own,
            'is_group': is_group,
            'is_prev': is_prev,
            'is_others': is_others
        })

    df_flagged = pd.DataFrame(results)
    if df_flagged.empty:
        return pd.DataFrame()

    summary = (
        df_flagged
        .groupby('participant_id')[['is_own', 'is_group', 'is_prev', 'is_others']]
        .mean()
        .rename(columns={
            'is_own': 'own_share',
            'is_group': 'group_share',
            'is_prev': 'prev_share',
            'is_others': 'others_share'
        })
        .reset_index()
    )

    summary['n'] = df_flagged.groupby('participant_id').size().values
    return summary

def process_all_model_files(input_dir, output_file):
    all_results = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.startswith("model_") and file.endswith(".csv"):
                path = os.path.join(root, file)
                print(f"📄 Processing: {path}")
                try:
                    df = pd.read_csv(path)
                    result = compute_strategy_shares(df)
                    if not result.empty:
                        result['experiment'] = file.replace("model_", "").replace(".csv", "")
                        all_results.append(result)
                except Exception as e:
                    print(f"❌ Failed to process {file}: {e}")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        mean_row = final_df[['own_share', 'group_share', 'prev_share', 'others_share']].mean()
        mean_row['experiment'] = 'AVERAGE'
        mean_row['participant_id'] = 'ALL'
        mean_row['n'] = final_df['n'].mean()

        print("\n📊 Average Strategy Proportions:")
        print(mean_row[['own_share', 'group_share', 'prev_share', 'others_share']])

        final_df = pd.concat([final_df, pd.DataFrame([mean_row])], ignore_index=True)
        final_df.to_csv(output_file, index=False)
        print(f"✅ Saved summary to {output_file}")
    else:
        print("⚠️ No valid model_*.csv files were processed.")

if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_file = "processed_strategy_summary_no.csv"
    process_all_model_files(input_dir, output_file)