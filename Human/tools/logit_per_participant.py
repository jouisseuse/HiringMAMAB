# logit_per_participant.py

import os
import sys
import pandas as pd
import statsmodels.api as sm

def prepare_data(df):
    df = df.copy()
    df['prev_choice_same'] = (df['arm'] == df['prev_choice']).astype(int)
    df['win_stay'] = df['prev_choice_same'] * df['prev_reward']
    return df

def fit_model_for_participant(df_sub, pid, experiment):
    df_sub = prepare_data(df_sub)
    df_choice = df_sub[df_sub['choice'] == 1].copy()

    # Skip participants with insufficient data
    if df_choice.empty or df_choice['arm'].nunique() < 2:
        return []

    try:
        y = df_choice['arm'].astype(int)
        X = df_choice[[
            'own_success', 'own_failure',
            'group_success', 'group_failure',
            'prev_choice_same', 'win_stay'
        ]]
        X = sm.add_constant(X)

        model = sm.MNLogit(y, X)
        result = model.fit(disp=False)

        summary = result.summary2().tables[1]
        rows = []

        for idx in summary.index:
            if "[" in idx and "]" in idx:
                param_name, arm_str = idx.split("[")
                arm = int(arm_str.rstrip("]"))

                rows.append({
                    "experiment": experiment,
                    "participant_id": pid,
                    "arm": arm,
                    "parameter": param_name.strip(),
                    "coef": summary.loc[idx, "Coef."],
                    "pval": summary.loc[idx, "P>|z|"]
                })
        return rows

    except Exception as e:
        print(f"❌ Fail on participant {pid} in {experiment}: {e}")
        return []

def analyze_all(base_folder):
    results = []
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.startswith("model_") and file.endswith(".csv"):
                file_path = os.path.join(root, file)
                experiment = file.replace("model_", "").replace(".csv", "")
                print(f"📄 Processing: {file_path}")
                try:
                    df = pd.read_csv(file_path)
                    for pid, df_sub in df.groupby('participant_id'):
                        rows = fit_model_for_participant(df_sub, pid, experiment)
                        results.extend(rows)
                except Exception as e:
                    print(f"❌ Error loading {file_path}: {e}")
    return pd.DataFrame(results)

if __name__ == "__main__":
    folder = sys.argv[1]
    df_all = analyze_all(folder)
    df_all.to_csv("summary_logit_per_participant.csv", index=False)
    print("✅ Saved: summary_logit_per_participant.csv")