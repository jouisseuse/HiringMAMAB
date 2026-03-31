# file: scripts/process_group_allocations.py
"""Process group allocation results from player.csv files into group_allocation.csv per folder.

- Input: root folder(s). Recursively find all `player.csv` files.
- For each `player.csv`, read `id` and `groupAllocations` columns.
- Parse `groupAllocations` JSON-like strings (handles doubled quotes from CSV escaping).
- Output: `group_allocation.csv` in the same folder, columns:
  id, Sky Blue, Slate, Purple, Indigo, Amber, Crimson, Pink, Black, Bright Green, Orange, entropy
- Skip rows where groupAllocations is empty.
- Append a final row `average` with per-column means (including entropy mean).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Dict, Iterable, List, Optional

import pandas as pd

GROUPS = [
    "Sky Blue",
    "Slate",
    "Purple",
    "Indigo",
    "Amber",
    "Crimson",
    "Pink",
    "Black",
    "Bright Green",
    "Orange",
]


def _find_player_csvs(root_dirs: Iterable[str]) -> List[str]:
    files: List[str] = []
    for root in root_dirs:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn == "player.csv":
                    files.append(os.path.join(dirpath, fn))
    return files


def _parse_allocations(raw: str) -> Dict[str, int]:
    if not raw or pd.isna(raw):
        return {}
    s = str(raw).strip()
    if not s:
        return {}
    # Normalize doubled quotes → single quotes
    s = s.replace("''", '"').replace('""', '"')
    if s and s[0] == "'":
        s = s.replace("'", '"')
    s = re.sub(r",\s*}\s*$", "}", s)

    try:
        data = json.loads(s)
    except Exception:
        try:
            import ast
            data = ast.literal_eval(s)
        except Exception:
            return {}

    out: Dict[str, int] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                out[str(k)] = int(v)
            except Exception:
                continue
    return out


def _entropy(counts: List[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p, 2)
    return ent


def process_player_csv(path: str) -> Optional[str]:
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception as e:
        print(f"Skip {path}: {e}")
        return None

    if not set(["id", "groupAllocations"]).issubset(df.columns):
        print(f"Skip {path}: missing required columns 'id' and 'groupAllocations'")
        return None

    rows = []
    for _, row in df.iterrows():
        rid = str(row.get("id", "")).strip()
        raw_alloc = row.get("groupAllocations", "")
        if not raw_alloc or str(raw_alloc).strip() == "":
            continue  # skip incomplete allocations
        alloc = _parse_allocations(raw_alloc)
        if not alloc:
            continue
        counts = {g: int(alloc.get(g, 0)) for g in GROUPS}
        ent = _entropy(list(counts.values()))
        out_row = {"id": rid, **counts, "entropy": ent}
        rows.append(out_row)

    if not rows:
        print(f"No valid rows in {path}")
        return None

    out_df = pd.DataFrame(rows, columns=["id", *GROUPS, "entropy"])

    avg_values = {col: out_df[col].astype(float).mean() for col in GROUPS + ["entropy"]}
    avg_row = {"id": "average", **avg_values}
    out_df = pd.concat([out_df, pd.DataFrame([avg_row])], ignore_index=True)

    out_dir = os.path.dirname(path)
    out_path = os.path.join(out_dir, "group_allocation.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(rows)} participants + average)")
    return out_path


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Process group allocation results into CSVs.")
    parser.add_argument("folders", nargs="+", help="Root folder(s) to scan recursively for player.csv")
    args = parser.parse_args(argv)

    files = _find_player_csvs(args.folders)
    if not files:
        print("No player.csv found.")
        return

    for f in files:
        process_player_csv(f)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])