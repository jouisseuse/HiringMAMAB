import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import seaborn as sns
import scipy.stats as st
import statsmodels.formula.api as smf

def load_experiment_data(folder_path):
    """Recursively load all playerRound_processed.csv files from a folder."""
    file_paths = glob.glob(os.path.join(folder_path, "**/playerRound_processed.csv"), recursive=True)
    experiment_data = [pd.read_csv(file) for file in file_paths]
    return experiment_data

# Define candidates and optimal arm
options = [
    "Crimson", "Bright Green", "Amber", "Purple", "Sky Blue", "Pink", "Indigo", "Slate", "Orange", "Black"
]
optimal_arm = "Bright Green"
marker_size = 3

def calculate_optimal_arm_rate(data, condition):
    """Compute the optimal arm selection rate and 95% CI per round."""
    num_rounds = min(len(df) for df in data)
    optimal_rates = []

    for round_num in range(num_rounds):
        round_choices = []
        for df in data:
            if round_num < len(df):
                round_choices.append(df.iloc[round_num].value_counts(normalize=True).get(optimal_arm, 0))

        mean_rate = np.mean(round_choices)
        ci_lower, ci_upper = (st.t.interval(0.95, len(round_choices) - 1, loc=mean_rate, scale=st.sem(round_choices))
                              if len(round_choices) > 1 else (mean_rate, mean_rate))
        optimal_rates.append([round_num + 1, mean_rate, ci_lower, ci_upper, condition])

    return pd.DataFrame(optimal_rates, columns=["Round", "Mean", "Lower Bound", "Upper Bound", "Condition"])

def analyze_statistical_significance(df_comm, df_no_comm):
    """Compute OLS coefficient b, 95% CI, and p-value."""
    combined_df = pd.concat([df_comm, df_no_comm], ignore_index=True)
    model = smf.ols("Mean ~ C(Condition)", data=combined_df).fit()

    try:
        b_value = model.params["C(Condition)[T.Communication]"]
        conf_int = model.conf_int().loc["C(Condition)[T.Communication]"]
        p_value = model.pvalues["C(Condition)[T.Communication]"]
        print(f"b = {b_value:.4f}, 95% CI = [{conf_int[0]:.4f}, {conf_int[1]:.4f}], p = {p_value:.4f}")
    except KeyError:
        print("OLS model error: Could not find 'C(Condition)[T.Communication]'. Check category names.")


def plot_optimal_arm_rate(df_comm, df_no_comm, save_path):
    """Plot and save the optimal arm rate trend."""
    sns.set_theme(style="white")
    plt.figure(figsize=(4, 3))
    ax = plt.gca()

    plt.plot(df_comm["Round"], df_comm["Mean"], label="Social learning", linestyle='-', linewidth=1.8, color="#1a80bb")
    plt.fill_between(df_comm["Round"], df_comm["Lower Bound"], df_comm["Upper Bound"], color="#1a80bb", alpha=0.2)

    plt.plot(df_no_comm["Round"], df_no_comm["Mean"], label="Individual learning", linestyle='-', linewidth=1.8, color="#ea801c")
    plt.fill_between(df_no_comm["Round"], df_no_comm["Lower Bound"], df_no_comm["Upper Bound"], color="#ea801c", alpha=0.2)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Fix y-axis range to 0-1
    ax.set_ylim(0, 1)

    plt.xlabel("Rounds", fontsize=12, fontname="Arial")
    plt.ylabel("Optimal Arm Rate", fontsize=12, fontname="Arial")
    plt.xticks(fontsize=10, fontname="Arial")
    plt.yticks(fontsize=10, fontname="Arial")
    # plt.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.55, 0), ncol=2)
    plt.savefig(os.path.join(save_path, "human-optimal_arm_rate.pdf"), dpi=300, bbox_inches="tight", transparent=True, format="pdf")
    plt.close()

def main():
    comm_folder = sys.argv[1]
    no_comm_folder = sys.argv[2]

    comm_data = load_experiment_data(comm_folder)
    no_comm_data = load_experiment_data(no_comm_folder)

    save_path = comm_folder
    os.makedirs(save_path, exist_ok=True)

    df_optimal_comm = calculate_optimal_arm_rate(comm_data, "Communication")
    df_optimal_no_comm = calculate_optimal_arm_rate(no_comm_data, "No-Communication")

    df_optimal_comm.to_csv(os.path.join(save_path, "optimal_arm_rate_comm.csv"), index=False)
    df_optimal_no_comm.to_csv(os.path.join(save_path, "optimal_arm_rate_no_comm.csv"), index=False)

    plot_optimal_arm_rate(df_optimal_comm[:-1], df_optimal_no_comm[:-1], save_path)
    # analyze_statistical_significance(df_optimal_comm, df_optimal_no_comm)

    print(f"Optimal arm rate analysis completed. Results saved in {save_path}")

if __name__ == "__main__":
    main()
