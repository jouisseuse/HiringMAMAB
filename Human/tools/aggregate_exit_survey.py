import sys
import os
import pandas as pd


def find_csv_files(root_dirs, target_filename="exit_survey_results.csv"):
    for root_dir in root_dirs:
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file == target_filename:
                    yield os.path.join(dirpath, file)


def aggregate_exit_surveys(root_dirs, output_file="aggregated_exit_survey_asocial.csv"):
    csv_files = list(find_csv_files(root_dirs))
    if not csv_files:
        print("No exit_survey_results.csv files found.")
        return

    all_data = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, usecols=["keyFactors", "strategy"])
            all_data.append(df)
        except Exception as e:
            print(f"Skipping {file}: {e}")

    if not all_data:
        print("No valid data collected.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    print(f"Aggregated data saved to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <folder1> [<folder2> ...]")
        sys.exit(1)

    input_dirs = sys.argv[1:]
    aggregate_exit_surveys(input_dirs)