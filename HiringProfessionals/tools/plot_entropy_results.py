import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import sys
import math
import seaborn as sns

def calculate_entropy(frequencies):
    total = sum(frequencies)
    probabilities = [freq / total for freq in frequencies]
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    return entropy

def process_csv(input_csv):
    # Load CSV
    df = pd.read_csv(input_csv, index_col=0)

    # Drop first and last rows (instruction/practice rounds)
    df = df.iloc[1:-1]

    # Create output directory
    output_dir = os.path.join(os.path.dirname(input_csv), "figures")
    os.makedirs(output_dir, exist_ok=True)

    # Candidate university names
    options = [
      "Trinity College Dublin",
      "The University of Western Australia",
      "University of Glasgow",
      "Heidelberg University",
      "University of Adelaide",
      "University of Leeds",
      "University of Southampton",
      "University of Sheffield",
      "University of Nottingham",
      "Karlsruhe Institute of Technology"
    ]

    # Overall analysis: cumulative entropy of candidate selections
    overall_entropy = []
    cumulative_choices = pd.Series(0, index=options, dtype=int)

    for round_id in df.index:
        round_choices = df.loc[round_id].dropna()
        cumulative_choices = cumulative_choices.add(round_choices.value_counts(), fill_value=0)
        overall_entropy.append(calculate_entropy(cumulative_choices))

    print(cumulative_choices)

    # Save entropy data
    entropy_data = pd.DataFrame({
        "Round": range(1, len(overall_entropy) + 1),
        "Overall Entropy": overall_entropy
    })
    entropy_csv_path = os.path.join(os.path.dirname(input_csv), "overall_entropy.csv")
    entropy_data.to_csv(entropy_csv_path, index=False)

    # Bar chart: cumulative selection proportions
    total_count = cumulative_choices.sum()
    arm_labels = cumulative_choices.index.tolist()
    arm_counts = cumulative_choices.values

    # Compute selection percentages
    arm_percentages = [count / total_count * 100 for count in arm_counts]
    average_percentage = np.mean(arm_percentages)

    # Flag if any arm exceeds 70% selection rate
    if any(p > 70 for p in arm_percentages):
        flag_file = os.path.join(output_dir, "flag.txt")
        with open(flag_file, "w") as f:
            f.write("Triggered: One arm exceeds 50% selection rate.\n")
        print(f"Flag file created at: {flag_file}")

    # Apply Seaborn style
    sns.set(style="whitegrid", palette="pastel", font_scale=1.1)

    # Color: darker for above-average, lighter for below-average
    colors = ['#4575b4' if percent >= average_percentage else '#91bfdb' for percent in arm_percentages]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw bars
    bars = ax.bar(arm_labels, arm_percentages, color=colors, edgecolor='black', width=0.7)

    # Title and axis labels
    ax.set_title("Proportion of Candidate Selections", fontsize=16, weight='bold', pad=20)
    ax.set_xlabel("Candidates", fontsize=13, labelpad=10)
    ax.set_ylabel("Selection Percentage (%)", fontsize=13, labelpad=10)

    # Average line
    ax.axhline(average_percentage, color='#d73027', linestyle='--', linewidth=2.5, label=f'Average: {average_percentage:.2f}%')
    ax.legend()

    # Value labels on bars
    for bar, percentage in zip(bars, arm_percentages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{percentage:.1f}%", ha="center", va="bottom", fontsize=11, weight='bold', color='black')

    # Entropy annotation
    entropy_text = f"Entropy: {overall_entropy[-1]:.2f}"
    ax.text(0.3, 0.7, entropy_text, transform=ax.transAxes, fontsize=14,
    weight='bold', color='black', ha='right', va='top', bbox=dict(facecolor='white', edgecolor='black'))

    # Despine and add grid
    sns.despine(left=True, bottom=True)
    ax.grid(True, axis='y', color='gray', linestyle='--', linewidth=0.6, alpha=0.5)

    # Save figure
    plt.savefig(os.path.join(output_dir, "candidate_distribution.png"), dpi=600, bbox_inches="tight")
    plt.close()

    # Cumulative entropy trend plot
    plt.style.use("default")
    plt.figure()
    plt.plot(range(1, len(overall_entropy) + 1), overall_entropy, label="Overall Entropy")
    plt.xlabel("Round")
    plt.ylabel("Cumulative Entropy")
    plt.title("Overall Cumulative Entropy Trend")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "overall_entropy_trend.png"))
    plt.close()

    # Per-player analysis
    player_entropies = {}
    average_entropies = []

    for player in df.columns:
        player_entropy = []
        player_choices = pd.Series(dtype=int)

        for round_id in df.index:
            choice = df.at[round_id, player]
            if pd.notna(choice):
                player_choices = player_choices.add(pd.Series([1], index=[choice]), fill_value=0)
            player_entropy.append(calculate_entropy(player_choices))

        player_entropies[player] = player_entropy

        # Save per-player entropy trend plot
        plt.figure()
        plt.plot(range(1, len(player_entropy) + 1), player_entropy, label=f"Player {player} Entropy")
        plt.xlabel("Round")
        plt.ylabel("Cumulative Entropy")
        plt.title(f"Player {player} Cumulative Entropy Trend")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"player_{player}_entropy_trend.png"))
        plt.close()

    # Compute average entropy per round across players
    for round_idx in range(len(df.index)):
        round_entropies = [player_entropies[player][round_idx] for player in df.columns]
        average_entropies.append(np.nanmean(round_entropies))

    # Save average entropy data
    avg_entropy_data = pd.DataFrame({
        "Round": range(1, len(average_entropies) + 1),
        "Average Entropy": average_entropies
    })
    avg_entropy_csv_path = os.path.join(os.path.dirname(input_csv), "average_entropy.csv")
    avg_entropy_data.to_csv(avg_entropy_csv_path, index=False)

    print(f"Entropy data saved: {entropy_csv_path}, {avg_entropy_csv_path}")

    # Average entropy trend plot
    plt.figure()
    plt.plot(range(1, len(average_entropies) + 1), average_entropies, label="Average Entropy")
    plt.xlabel("Round")
    plt.ylabel("Average Entropy")
    plt.title("Average Entropy Trend Across Players")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "average_entropy_trend.png"))
    plt.close()

    # Per-player candidate choices scatter plot
    for player in df.columns:
        player_data = df[player].reset_index()
        player_data.columns = ["Round", "Candidate"]

        # Fill missing choices
        player_data["Candidate"] = player_data["Candidate"].fillna("No Choice").astype(str)

        plt.figure()
        plt.scatter(player_data["Round"], player_data["Candidate"], label=f"Player {player} Choices")
        plt.xlabel("Round")
        plt.ylabel("Candidate")
        plt.title(f"Player {player} Candidate Choices")
        plt.savefig(os.path.join(output_dir, f"player_{player}_choices.png"))
        plt.close()

if __name__ == "__main__":
    input_csv = sys.argv[1]
    process_csv(input_csv)
    print(f"Analysis complete. All figures saved to {os.path.join(os.path.dirname(input_csv), 'figures')}")
