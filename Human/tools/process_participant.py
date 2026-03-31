import pandas as pd
import numpy as np
from scipy.stats import entropy
import os
from pathlib import Path
import sys

# === Step 1: Compute entropy for player choices ===
def compute_entropy(choices):
    counts = choices.value_counts(dropna=True)
    probabilities = counts / counts.sum()
    return entropy(probabilities, base=2)

# === Step 2: Resolve fallback demographic fields ===
def resolve_field(row, field, alt_field):
    return row[field] if pd.notna(row[field]) and row[field] != "" else row[alt_field]

# === Step 3: Process a single experiment folder ===
def process_experiment_folder(folder_path: Path):
    round_file = folder_path / "playerRound_processed.csv"
    survey_file = folder_path / "exit_survey_results.csv"

    if not round_file.exists() or not survey_file.exists():
        print(f"Skipping {folder_path}, missing required files.")
        return

    # Read and compute entropy
    round_df = pd.read_csv(round_file, skiprows=0)
    round_df.set_index(round_df.columns[0], inplace=True)
    player_choices = round_df.transpose()
    entropy_series = player_choices.apply(compute_entropy, axis=1)
    entropy_df = entropy_series.reset_index()
    entropy_df.columns = ['playerID', 'entropy']

    # Read and clean survey
    survey_df = pd.read_csv(survey_file)
    survey_df['gender'] = survey_df.apply(lambda r: resolve_field(r, 'gender', 'genderOther'), axis=1)
    survey_df['race'] = survey_df.apply(lambda r: resolve_field(r, 'race', 'raceOther'), axis=1)
    survey_df['education'] = survey_df.apply(lambda r: resolve_field(r, 'education', 'educationOther'), axis=1)
    selected_survey = survey_df[['playerID', 'age', 'gender', 'race', 'education', 'politicalOrientation']]

    # Merge and export
    merged_df = pd.merge(entropy_df, selected_survey, on='playerID', how='left')
    output_file = folder_path / "merged_output.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")

# === Step 4: Process all subfolders ===
def process_all_experiments(main_folder: str):
    root = Path(main_folder)

    for subdir in root.iterdir():
        if subdir.is_dir():
            process_experiment_folder(subdir)

# Example usage:
input_dir = sys.argv[1]
process_all_experiments(input_dir)