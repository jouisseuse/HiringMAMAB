import csv
import pandas as pd
import sys
import os

def process_csv(input_csv, output_csv):
    # Load CSV
    df = pd.read_csv(input_csv)

    # Fill missing decisions with empty string
    df['decision'] = df['decision'].fillna('')

    # Parse timestamp column
    df['roundIDLastChangedAt'] = pd.to_datetime(df['roundIDLastChangedAt'], format='%Y-%m-%dT%H:%M:%S.%fZ')

    # Sort by timestamp and player ID
    df_sorted = df.sort_values(by=['roundIDLastChangedAt', 'playerID'], ascending=[True, True])

    # Pivot: rows=rounds, columns=players, values=decisions
    pivot_table = df_sorted.pivot(index='roundID', columns='playerID', values='decision')

    # Preserve original ordering
    pivot_table = pivot_table.loc[df_sorted['roundID'].unique(), df_sorted['playerID'].unique()]

    # Write to CSV
    pivot_table.to_csv(output_csv)

if __name__ == "__main__":
    input_csv = sys.argv[1]
    base_name, ext = os.path.splitext(input_csv)
    output_csv = f"{base_name}_processed{ext}"

    process_csv(input_csv, output_csv)
    print(f"Processed data saved to: {output_csv}")