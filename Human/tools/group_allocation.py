import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import sys

def process_group_allocations(input_csv):
    # Load CSV
    df = pd.read_csv(input_csv)

    # Create output figures directory
    output_dir = os.path.join(os.path.dirname(input_csv), "figures")
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each player's allocation data
    for idx, row in df.iterrows():
        player_id = row.get("id")
        allocations_json = row.get("groupAllocations")

        # Skip players with no allocation data
        if pd.isna(allocations_json):
            continue

        # Parse allocation JSON
        try:
            allocations = json.loads(allocations_json)
        except json.JSONDecodeError:
            print(f"Invalid JSON format for player {player_id}, skipping.")
            continue

        # Plot pie chart
        labels = allocations.keys()
        sizes = allocations.values()

        plt.figure(figsize=(8, 6))
        plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, colors=["skyblue", "pink", "limegreen", "slateblue", "indigo", "orange", "red", "cyan", "yellow", "purple"])
        plt.title(f"Group Allocations for Player {player_id}")

        # Save figure
        plt.savefig(os.path.join(output_dir, f"player_{player_id}_allocations.png"))
        plt.close()

def process_exit_survey(input_csv):
    # Load CSV
    df = pd.read_csv(input_csv)

    # Set output path
    output_dir = os.path.dirname(input_csv)
    output_csv = os.path.join(output_dir, "exit_survey_results.csv")

    # Collect survey rows
    survey_data = []

    for idx, row in df.iterrows():
        player_id = row.get("id")
        survey_json = row.get("exitSurvey")

        # Skip players with no survey data
        if pd.isna(survey_json):
            continue

        # Parse survey JSON
        try:
            survey = json.loads(survey_json)
            survey["playerID"] = player_id
            survey_data.append(survey)
        except json.JSONDecodeError:
            print(f"Invalid JSON format for player {player_id}, skipping.")
            continue

    # Save survey results to CSV
    survey_df = pd.DataFrame(survey_data)
    survey_df.to_csv(output_csv, index=False)

    print(f"Survey results saved to: {output_csv}")

if __name__ == "__main__":
    input_csv = sys.argv[1]
    process_group_allocations(input_csv)
    process_exit_survey(input_csv)
    print(f"Figures saved to: {os.path.join(os.path.dirname(input_csv), 'figures')}")
