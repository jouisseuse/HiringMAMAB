import os
import pandas as pd
import sys
import statsmodels.formula.api as smf

def read_entropy_data(main_folder, label):
    """Read merged_output.csv from each sub-experiment under a root folder and attach a condition label."""
    all_data = []
    for subfolder in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, subfolder)
        csv_path = os.path.join(folder_path, "merged_output.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["Communication"] = label
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def merge_and_clean(com_path, nocom_path):
    com_df = read_entropy_data(com_path, 1)
    nocom_df = read_entropy_data(nocom_path, 0)
    combined = pd.concat([com_df, nocom_df], ignore_index=True)
    cleaned = combined.dropna(subset=["entropy", "age", "gender", "race", "education", "politicalOrientation"])
    return cleaned

def run_regression(df):
    # Cast to category for dummy encoding in OLS
    for col in ["gender", "race", "education", "politicalOrientation"]:
        df[col] = df[col].astype("category")

    # OLS regression
    model = smf.ols("entropy ~ Communication + age + gender + race + education + politicalOrientation", data=df).fit()
    print(model.summary())

com_folder = sys.argv[1]
nocom_folder = sys.argv[2]

data = merge_and_clean(com_folder, nocom_folder)
run_regression(data)
