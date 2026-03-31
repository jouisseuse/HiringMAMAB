import pandas as pd
import os
from pathlib import Path
import sys

# === Helper to resolve fallback fields ===
def resolve_field(row, field, alt_field):
    return row[field] if pd.notna(row[field]) and row[field] != "" else row[alt_field]

# === Process all experiment folders recursively ===
def process_all_experiments(main_folders):
    all_survey_data = []

    for main_folder in main_folders:
        for root_dir, dirs, files in os.walk(main_folder):
            root_path = Path(root_dir)
            survey_file = root_path / "exit_survey_results.csv"

            if not survey_file.exists():
                continue

            survey_df = pd.read_csv(survey_file)
            survey_df['gender'] = survey_df.apply(lambda r: resolve_field(r, 'gender', 'genderOther'), axis=1)
            survey_df['race'] = survey_df.apply(lambda r: resolve_field(r, 'race', 'raceOther'), axis=1)
            survey_df['education'] = survey_df.apply(lambda r: resolve_field(r, 'education', 'educationOther'), axis=1)
            survey_df = survey_df[['age', 'gender', 'race', 'education', 'politicalOrientation']]
            all_survey_data.append(survey_df)

    if not all_survey_data:
        print("No valid survey data found.")
        return

    full_df = pd.concat(all_survey_data, ignore_index=True)
    full_df = full_df.dropna(subset=['age'])
    full_df['age'] = pd.to_numeric(full_df['age'], errors='coerce')
    full_df = full_df.dropna(subset=['age'])

    # Age stats
    age_mean = full_df['age'].mean()
    age_min = full_df['age'].min()
    age_max = full_df['age'].max()

    summary = {
        'age_mean': [age_mean],
        'age_min': [age_min],
        'age_max': [age_max],
    }

    for field in ['gender', 'race', 'education', 'politicalOrientation']:
        proportions = full_df[field].value_counts(normalize=True)
        for value, proportion in proportions.items():
            key = f"{field}__{value}"
            summary[key] = [proportion]

    summary_df = pd.DataFrame(summary)
    output_file = Path("summary_output.csv")
    summary_df.to_csv(output_file, index=False)
    print(f"Saved summary: {output_file}")

# === Entry Point ===
if __name__ == '__main__':
    input_dirs = sys.argv[1:]
    process_all_experiments(input_dirs)
