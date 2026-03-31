import os
import sys
import pandas as pd
import json

def process_csv(input_csv, output_json):
    # Map option names to numeric indices
    options = [
        "Crimson", "Bright Green", "Amber", "Purple", "Sky Blue", "Pink", "Indigo", "Slate", "Orange", "Black"
    ]
    option_mapping = {option: idx for idx, option in enumerate(options)}

    # Load CSV
    df = pd.read_csv(input_csv)

    # Fill missing decisions
    df['decision'] = df['decision'].fillna('')

    # Map option names to numeric indices
    df['decision'] = df['decision'].map(option_mapping).fillna(-1).astype(int)

    # Parse timestamp column
    df['roundIDLastChangedAt'] = pd.to_datetime(df['roundIDLastChangedAt'], format='%Y-%m-%dT%H:%M:%S.%fZ')

    # Sort by timestamp and player ID
    df_sorted = df.sort_values(by=['roundIDLastChangedAt', 'playerID'], ascending=[True, True])

    # Pivot: rows=rounds, columns=players, values=decisions/scores
    pivot_choices = df_sorted.pivot(index='roundID', columns='playerID', values='decision')
    pivot_scores = df_sorted.pivot(index='roundID', columns='playerID', values='score')

    # Preserve original ordering
    round_ids = df_sorted['roundID'].unique()
    player_ids = df_sorted['playerID'].unique()
    pivot_choices = pivot_choices.loc[round_ids, player_ids].fillna(-1).astype(int)
    pivot_scores = pivot_scores.loc[round_ids, player_ids].fillna(0).astype(int)

    # Drop first and last rounds (introduction/warmup)
    round_ids = round_ids[1:-1]

    # Build JSON structure
    results = []
    for round_id in round_ids:
        choices = pivot_choices.loc[round_id].tolist()
        rewards = pivot_scores.loc[round_id].tolist()
        results.append({"choices": choices, "rewards": rewards})

    # Write JSON
    with open(output_json, 'w') as json_file:
        json.dump({"results": results}, json_file, indent=4)

    print(f"Processed data saved to: {output_json}")

if __name__ == "__main__":
    input_csv = sys.argv[1]
    base_name, _ = os.path.splitext(input_csv)
    output_json = f"{base_name}.log"

    process_csv(input_csv, output_json)
