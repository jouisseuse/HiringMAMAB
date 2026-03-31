import os
import subprocess
import sys

def process_all_player_rounds(directory):
    """Recursively find all playerRound.csv files and run process_different_to_log.py on each."""
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    player_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file == "playerRound.csv":
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")

                subprocess.run(["python3", "process_different_to_log.py", file_path], check=True)

            elif file == "player.csv":
                file_path = os.path.join(root, file)
                player_files.append(file_path)

    # Process player.csv files for group allocations
    for player_file in player_files:
        print(f"Processing group allocation: {player_file}")
        subprocess.run(["python3", "group_allocation.py", player_file], check=True)

if __name__ == "__main__":
    directory = input("Enter the directory path containing playerRound.csv files: ").strip()
    process_all_player_rounds(directory)
