# file: process_csv.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


FILTER_GAME_ID = "01KDNWHDJA5A4FAN49KSNS5QFK"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter out rows from a CSV by gameID."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to output cleaned CSV file",
    )
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def filter_game_id(df: pd.DataFrame, game_id: str) -> pd.DataFrame:
    if "gameID" not in df.columns:
        raise KeyError("Required column 'gameID' not found in CSV")

    original_count = len(df)
    df_filtered = df[df["gameID"] != game_id]
    removed = original_count - len(df_filtered)

    print(f"Removed {removed} rows where gameID == {game_id}")
    return df_filtered


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()

    try:
        df = load_csv(args.input)
        df = filter_game_id(df, FILTER_GAME_ID)
        write_csv(df, args.output)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Cleaned CSV written to: {args.output}")


if __name__ == "__main__":
    main()