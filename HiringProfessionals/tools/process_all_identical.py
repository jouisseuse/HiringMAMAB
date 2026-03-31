import os
import subprocess
import sys

def process_all_files(directory):
    """
    Recursively process all experiment files under directory:
    1. For each playerRound.csv: check if playerRound_processed.csv already exists.
    2. For each playerRound_processed.csv: run plot_entropy_results.py.
    3. player.csv files (group allocation) are currently disabled.
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    processed_files = []
    player_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)

            if file == "playerRound.csv":
                print(f"Processing: {file_path}")
                # subprocess.run(["python3", "process_round_results.py", file_path], check=True)
                processed_file_path = os.path.join(root, "playerRound_processed.csv")
                if os.path.exists(processed_file_path):
                    processed_files.append(processed_file_path)

            elif file == "player.csv":
                player_files.append(file_path)

    # Plot entropy for all processed round files
    for processed_file in processed_files:
        print(f"Plotting entropy results: {processed_file}")
        subprocess.run(["python3", "plot_entropy_results.py", processed_file], check=True)

    # Group allocation processing (currently disabled)
    # for player_file in player_files:
    #     print(f"Processing group allocation: {player_file}")
    #     subprocess.run(["python3", "group_allocation.py", player_file], check=True)

if __name__ == "__main__":
    directory = sys.argv[1]
    process_all_files(directory)
