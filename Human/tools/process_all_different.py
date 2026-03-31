import os
import subprocess

def process_all_player_rounds(directory):
    """
    Recursively traverse the input directory and its subdirectories,
    convert all playerRound.csv files to log format with process_different_to_log.py,
    and run group_allocation.py on all player.csv files.
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    player_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file == "playerRound.csv":
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")

                # Convert to log format
                subprocess.run(["python3", "process_different_to_log.py", file_path], check=True)

            elif file == "player.csv":
                file_path = os.path.join(root, file)
                player_files.append(file_path)

    # Run group allocation on all player.csv files
    for player_file in player_files:
        print(f"Group allocation: {player_file}")
        subprocess.run(["python3", "group_allocation.py", player_file], check=True)

if __name__ == "__main__":
    directory = input("Enter the directory path containing playerRound.csv files: ").strip()
    process_all_player_rounds(directory)
