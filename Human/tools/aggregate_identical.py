import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.stats as st
import sys
import math
import statsmodels.formula.api as smf
import seaborn as sns

def aggregate_entropy(directory):
    """
    Traverse a directory, aggregate all overall_entropy.csv and average_entropy.csv files,
    and compute the per-round mean and 95% CI.
    """
    overall_files = glob.glob(os.path.join(directory, "**/overall_entropy.csv"), recursive=True)
    average_files = glob.glob(os.path.join(directory, "**/average_entropy.csv"), recursive=True)

    print(len(overall_files))
    def process_files(file_list, label):
        all_rounds_data = {}

        # Read all CSVs and collect entropy values
        for file in file_list:
            df = pd.read_csv(file)

            # Ensure CSV contains the expected column
            entropy_column = "Overall Entropy" if "Overall Entropy" in df.columns else "Average Entropy"
            if entropy_column not in df.columns:
                print(f"Warning: {file} does not contain '{entropy_column}'. Skipping.")
                continue

            # Collect entropy values per round
            for round_num, entropy_value in enumerate(df[entropy_column]):
                if round_num not in all_rounds_data:
                    all_rounds_data[round_num] = []
                all_rounds_data[round_num].append(entropy_value)

        # No valid data found
        if not all_rounds_data:
            print(f"No valid '{label}' data found in {directory}")
            return pd.DataFrame(columns=["Round", "Mean Entropy", "Lower Bound", "Upper Bound"])

        # Compute mean and 95% CI per round
        final_rounds_data = []
        for round_num, entropy_values in all_rounds_data.items():
            mean_entropy = np.mean(entropy_values)

            # Compute 95% CI
            if len(entropy_values) > 1:
                se_adjusted = max(st.sem(entropy_values), 0.01)  # prevent SE from being too small
                ci_lower, ci_upper = st.t.interval(0.95, len(entropy_values) - 1, loc=mean_entropy, scale=se_adjusted)
            else:
                ci_lower, ci_upper = mean_entropy - 0.05, mean_entropy + 0.05  # minimum-width fallback CI

            # Clip CI to valid entropy range
            ci_lower, ci_upper = max(0, ci_lower), min(3.32, ci_upper)  # max entropy for 10 equally probable arms

            # Append results
            final_rounds_data.append((round_num + 1, mean_entropy, ci_lower, ci_upper))

        # Build DataFrame
        result_df = pd.DataFrame(final_rounds_data, columns=["Round", "Mean Entropy", "Lower Bound", "Upper Bound"])

        # Save to CSV
        csv_path = os.path.join(directory, f"aggregated_{label.replace(' ', '_').lower()}.csv")
        result_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

        return result_df

    # Process both entropy file types
    overall_result = process_files(overall_files, "Overall Entropy")
    average_result = process_files(average_files, "Average Entropy")

    return overall_result, average_result

def aggregate_last_round_entropy(directory):
    """
    Traverse a directory, extract the last-round entropy from all sessions,
    for use in Strip Plot + Point Plot (Communication vs No-Communication).
    """
    overall_files = glob.glob(os.path.join(directory, "**/overall_entropy.csv"), recursive=True)
    average_files = glob.glob(os.path.join(directory, "**/average_entropy.csv"), recursive=True)

    print(f"Found {len(overall_files)} overall_entropy.csv files")
    print(f"Found {len(average_files)} average_entropy.csv files")

    def extract_last_round_entropy(file_list):
        """Read multiple session CSVs and extract the last-round entropy value."""
        last_round_data = []

        for file in file_list:
            df = pd.read_csv(file)

            # Ensure CSV contains expected column
            entropy_column = "Overall Entropy" if "Overall Entropy" in df.columns else "Average Entropy"
            if entropy_column not in df.columns:
                print(f"Warning: {file} does not contain '{entropy_column}'. Skipping.")
                continue

            # Extract last-round entropy
            last_round_entropy = df[entropy_column].dropna().iloc[-1]
            print(file, last_round_entropy)

            # Collect data
            last_round_data.append(last_round_entropy)

        # Return as list
        return last_round_data

    # Extract last-round entropy from all sessions
    last_round = extract_last_round_entropy(overall_files)

    # Return results
    return last_round

def plot_entropy_comparison(comm_result, no_comm_result, title, filename, directory, num_arms):
    """
    Plot entropy comparison curves for Communication vs No-Communication.
    """
    largest_entropy = 3.32  # max entropy for 10 equally probable arms

    # White background, no grid
    sns.set_theme(style="white")

    # Color palette
    comm_color = "#1a80bb"  # blue
    no_comm_color = "#ea801c"  # orange

    # Create canvas
    plt.figure(figsize=(4, 3))
    ax = plt.gca()

    # Plot Communication condition
    plt.plot(
        comm_result["Round"], comm_result["Mean Entropy"],
        label="Communication", color=comm_color, linestyle="-", linewidth=1.8
    )
    plt.fill_between(
        comm_result["Round"], comm_result["Lower Bound"], comm_result["Upper Bound"],
        color=comm_color, alpha=0.2, label="95% CI"
    )

    # Plot No-Communication condition
    plt.plot(
        no_comm_result["Round"], no_comm_result["Mean Entropy"],
        label="No-Communication", color=no_comm_color, linestyle="-", linewidth=1.8
    )
    plt.fill_between(
        no_comm_result["Round"], no_comm_result["Lower Bound"], no_comm_result["Upper Bound"],
        color=no_comm_color, alpha=0.2, label="95% CI"
    )

    # Fix y-axis range
    ax.set_ylim(0, largest_entropy)

    # Remove top and right spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Ensure 3.32 appears on y-axis ticks
    y_ticks = ax.get_yticks()
    if largest_entropy not in y_ticks:
        y_ticks = np.append(y_ticks, largest_entropy)
        y_ticks = np.sort(y_ticks)
    ax.set_yticks(y_ticks[:-1])

    # Format y-axis tick labels
    ax.set_yticklabels([f"{tick:.2f}" for tick in y_ticks[:-1]])

    # Set axis labels
    ax.set_xlabel("Rounds", fontsize=12, fontname="Arial")
    ax.set_ylabel("Entropy", fontsize=12, fontname="Arial")

    # Tick font settings
    plt.xticks(fontsize=10, fontname="Arial")
    plt.yticks(fontsize=10, fontname="Arial")

    ax.tick_params(axis="both", direction="out", length=3, width=0.8)

    # Legend (commented out)
    # plt.legend(frameon=False, fontsize=14)
    # legend = plt.legend(
    #     frameon=False, fontsize=9, loc="lower center",
    #     bbox_to_anchor=(0.55, 0), ncol=2
    # )
    # for text in legend.get_texts():
    #     text.set_color("black")  # ensure legend text is black

    # Save PDF
    plot_path = os.path.join(directory, filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", transparent=True, format="pdf")
    print(f"Saved enhanced entropy comparison plot at {plot_path}")


def plot_last_round_boxplot(comm_result, no_comm_result, output_dir):
    """
    Strip Plot + Point Plot (Nature-style, vertical orientation) with OLS regression output.
    """
    largest_entropy = math.log2(10)

    # Assemble DataFrame
    combined_data = [(entropy, "Communication") for entropy in comm_result]
    combined_data.extend([(entropy, "No-Communication") for entropy in no_comm_result])
    combined_df = pd.DataFrame(combined_data, columns=["Entropy", "Condition"])

    # OLS regression
    combined_df["Condition"] = combined_df["Condition"].str.strip()
    print(combined_df)
    model = smf.ols("Entropy ~ C(Condition, Treatment(reference='No-Communication'))", data=combined_df).fit()

    print("Model Parameters:", model.params.keys())
    b_key = "C(Condition, Treatment(reference='No-Communication'))[T.Communication]"

    if b_key in model.params:
        b_value = model.params[b_key]
        conf_int = model.conf_int().loc[b_key]
        p_value = model.pvalues[b_key]
        print(f"b = {b_value:.4f}, 95% CI = [{conf_int[0]:.4f}, {conf_int[1]:.4f}], p = {p_value:.4f}")
    else:
        print("OLS model error: Key not found in model parameters. Check Condition column values.")

    mean_no_comm = combined_df[combined_df["Condition"] == "No-Communication"]["Entropy"].mean()
    mean_comm = combined_df[combined_df["Condition"] == "Communication"]["Entropy"].mean()
    reduction = mean_no_comm - mean_comm
    percentage = (reduction / mean_no_comm) * 100 if mean_no_comm != 0 else 0
    print(f"Entropy reduction: {reduction:.4f}, Percentage reduction: {percentage:.2f}%")

    # Nature journal style
    sns.set_theme(style="whitegrid")

    # Create canvas
    plt.figure(figsize=(4, 3))
    ax = plt.gca()
    color_palette = {"Communication": "#1a80bb", "No-Communication": "#ea801c"}

    # Jittered data points
    sns.stripplot(
        x="Condition", y="Entropy", data=combined_df,
        jitter=True, size=8, alpha=0.2,
        palette=color_palette, ax=ax
    )

    # Mean and CI overlay
    sns.pointplot(
        x="Condition", y="Entropy", data=combined_df,
        capsize=0.0, errwidth=2.0, join=True,
        palette=color_palette, markers="o", scale=1.2, ax=ax
    )

    # Fix y-axis range
    ax.set_ylim(0, largest_entropy)

    # X-axis labels
    ax.set_xlabel("")
    ax.set_xticklabels(["Social learning", "Individual learning"], fontsize=12, fontname="Arial")

    # Y-axis settings
    ax.set_ylabel("Entropy", fontsize=12, fontname="Arial")
    ax.set_yticks(np.linspace(0, largest_entropy, 6))
    ax.set_yticklabels([f"{tick:.1f}" for tick in np.linspace(0, largest_entropy, 6)], fontsize=10, fontname="Arial")

    # Remove top and right spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.tick_params(axis="both", direction="out", length=3, width=0.8)

    # Save PDF
    output_path = os.path.join(output_dir, "human_entropy_strip_catplot.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")

    # Show plot
    plt.show()

    print(f"Saved entropy plot at {output_path}")

if __name__ == "__main__":
    comm_dir = sys.argv[1]
    no_comm_dir = sys.argv[2]

    comm_overall, comm_average = aggregate_entropy(comm_dir)
    no_comm_overall, no_comm_average = aggregate_entropy(no_comm_dir)

    comm_last = aggregate_last_round_entropy(comm_dir)
    no_comm_last = aggregate_last_round_entropy(no_comm_dir)

    output_dir = os.path.dirname(comm_dir)

    plot_entropy_comparison(comm_overall, no_comm_overall, "Overall Entropy Comparison", "human-overall_entropy_comparison.pdf", output_dir, 10)
    # plot_entropy_comparison(comm_average, no_comm_average, "Average Entropy Comparison", "average_entropy_comparison.png", output_dir, 10)

    print("Comparison plots saved to:", output_dir)
    plot_last_round_boxplot(comm_last, no_comm_last, output_dir)
