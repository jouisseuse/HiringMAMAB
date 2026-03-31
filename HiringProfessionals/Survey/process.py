# file: plot_university_survey.py
"""
Generate bar charts for:
1) Mean familiarity per university (bars with SD), optionally top N universities.
2) Mean centered evaluation per university (bars with 95% CI) ordered by familiarity.
3) NEW: Trimmed evaluation (drop one lowest & one highest per university) summary + plot,
   ordered identically to familiarity.

Usage:
  python plot_university_survey.py --csv "University-Survey-data.csv" --top 20 --output-dir "plots"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# 0. Familiarity mapping (edit as needed)
# -----------------------------
FAM_MAP: Dict[str, float] = {
    "I have never heard of this university": 0.0,
    "I have heard of it, but I am not familiar with it": 0.33,
    "I am somewhat familiar with it": 0.6,
    "I am very familiar with it": 1.0,
}


# -----------------------------
# Helpers
# -----------------------------
def parse_familiarity(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s in FAM_MAP:
        return float(FAM_MAP[s])
    try:
        return float(s)
    except Exception:
        return np.nan


def load_wide_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw CSV, ignoring first row, using second row as header. Insert participantId if missing."""
    df = pd.read_csv(csv_path, header=1)
    if "participantId" not in df.columns:
        df.insert(0, "participantId", np.arange(len(df)))
    return df


def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape wide survey into long with parsed familiarity and participant-centered evaluation."""
    pat = re.compile(r"^(.*)\s*-\s*(Familiarity|Evaluation)\s*$", re.IGNORECASE)

    records = []
    for col in df.columns:
        m = pat.match(str(col).strip())
        if not m:
            continue
        uni = m.group(1).strip()
        qtype = m.group(2).capitalize()
        records.append((col, uni, qtype))

    if not records:
        raise ValueError("No columns matched the pattern 'University - Familiarity/Evaluation'")

    map_df = pd.DataFrame(records, columns=["col", "university", "qtype"])

    long_parts = []
    for uni, sub in map_df.groupby("university"):
        fam_cols = sub.loc[sub["qtype"] == "Familiarity", "col"].tolist()
        eval_cols = sub.loc[sub["qtype"] == "Evaluation", "col"].tolist()
        if len(fam_cols) != 1 or len(eval_cols) != 1:
            # Why: mismatched pairs skew analysis
            print(f"Warning: '{uni}' has fam_cols={fam_cols}, eval_cols={eval_cols}. Skipping.")
            continue

        fam_col = fam_cols[0]
        eval_col = eval_cols[0]

        tmp = df[["participantId", fam_col, eval_col]].copy()
        tmp.columns = ["participantId", "familiarity_raw", "evaluation_raw"]
        tmp["university"] = uni
        long_parts.append(tmp)

    if not long_parts:
        raise ValueError("No valid university rows found after column pairing.")

    long_df = pd.concat(long_parts, ignore_index=True)

    long_df["familiarity_num"] = long_df["familiarity_raw"].apply(parse_familiarity)
    long_df["evaluation_raw"] = pd.to_numeric(long_df["evaluation_raw"], errors="coerce")

    # Participant-centered evaluation (remove rater bias)
    participant_mean = (
        long_df.groupby("participantId")["evaluation_raw"].mean().rename("participant_mean_rating").reset_index()
    )
    long_df = long_df.merge(participant_mean, on="participantId", how="left")
    long_df["evaluation_centered"] = long_df["evaluation_raw"] - long_df["participant_mean_rating"]

    return long_df


def compute_familiarity_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """Mean/SD/Variance/count familiarity per university."""
    summ = (
        long_df.groupby("university")
        .agg(
            mean_familiarity=("familiarity_num", "mean"),
            sd_familiarity=("familiarity_num", "std"),
            var_familiarity=("familiarity_num", "var"),
            n=("participantId", "nunique"),
        )
        .reset_index()
    )
    return summ


def compute_evaluation_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """Mean/SD/SE/95% CI for centered evaluation per university."""
    summ = (
        long_df.groupby("university")
        .agg(
            mean_centered=("evaluation_centered", "mean"),
            sd_centered=("evaluation_centered", "std"),
            n_ratings=("evaluation_centered", lambda s: s.notna().sum()),
        )
        .reset_index()
    )
    # Guard against division by zero
    summ["se_centered"] = summ.apply(
        lambda r: (r["sd_centered"] / np.sqrt(r["n_ratings"])) if (r["n_ratings"] and pd.notna(r["sd_centered"])) else np.nan,
        axis=1,
    )
    summ["ci95"] = 1.96 * summ["se_centered"]
    return summ


def compute_evaluation_summary_trimmed(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    NEW: For each university, drop exactly one lowest and one highest centered evaluation,
    then compute mean/SD/VAR/SE/95% CI on the remaining values.
    """
    rows = []
    for uni, sub in long_df.groupby("university"):
        s = sub["evaluation_centered"].dropna()
        n = int(s.shape[0])
        if n < 3:
            rows.append(
                {
                    "university": uni,
                    "mean_centered_trim": np.nan,
                    "sd_centered_trim": np.nan,
                    "var_centered_trim": np.nan,
                    "n_ratings_trim": 0,
                    "se_centered_trim": np.nan,
                    "ci95_trim": np.nan,
                    "n_original": n,
                }
            )
            continue

        # Drop one min and one max (handles duplicates by removing a single occurrence)
        idx_min = s.idxmin()
        s_wo_min = s.drop(idx_min)
        idx_max = s_wo_min.idxmax()
        s_trim = s_wo_min.drop(idx_max)

        n_trim = int(s_trim.shape[0])
        mean_trim = s_trim.mean()
        sd_trim = s_trim.std()
        var_trim = s_trim.var()
        se_trim = (sd_trim / np.sqrt(n_trim)) if (n_trim and pd.notna(sd_trim)) else np.nan
        ci95_trim = 1.96 * se_trim if pd.notna(se_trim) else np.nan

        rows.append(
            {
                "university": uni,
                "mean_centered_trim": mean_trim,
                "sd_centered_trim": sd_trim,
                "var_centered_trim": var_trim,
                "n_ratings_trim": n_trim,
                "se_centered_trim": se_trim,
                "ci95_trim": ci95_trim,
                "n_original": n,
            }
        )

    return pd.DataFrame(rows)


def get_fam_order(fam_summary: pd.DataFrame, top: int | None) -> Tuple[List[str], pd.DataFrame]:
    """Order universities by mean familiarity DESC and optionally keep top N. Return (order, selected df)."""
    data = fam_summary.sort_values("mean_familiarity", ascending=False)
    if top is not None and top > 0:
        data = data.head(top)
    order = data["university"].tolist()
    return order, data


# -----------------------------
# Plotting (matplotlib only; one figure each; no custom colors)
# -----------------------------
def plot_familiarity_bars(data: pd.DataFrame, outfile: Path) -> None:
    x = np.arange(len(data))
    y = data["mean_familiarity"].to_numpy()
    yerr = data["sd_familiarity"].fillna(0).to_numpy()

    plt.figure(figsize=(12, 7))
    plt.bar(x, y, yerr=yerr, capsize=3)
    plt.xticks(x, data["university"], rotation=45, ha="right")
    plt.ylabel("Mean familiarity (0–1)")
    # plt.title("Mean Familiarity by University (error bars = SD)")
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300, format="pdf")
    plt.close()


def plot_evaluation_ci_in_fam_order(eval_summary: pd.DataFrame, fam_order: List[str], outfile: Path) -> None:
    """Plot evaluation bars in the exact same order (and subset) as familiarity."""
    eval_idx = eval_summary.set_index("university")
    eval_ordered = eval_idx.loc[fam_order].reset_index()

    x = np.arange(len(eval_ordered))
    y = eval_ordered["mean_centered"].to_numpy()
    yerr = eval_ordered["ci95"].fillna(0).to_numpy()

    plt.figure(figsize=(12, 7))
    plt.bar(x, y, yerr=yerr, capsize=3)
    plt.xticks(x, eval_ordered["university"], rotation=45, ha="right")
    plt.ylabel("Perceived Quality")
    # plt.title("Mean Centered Evaluation by University (ordered by Familiarity; error bars = 95% CI)")
    plt.axhline(0.0)
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300, format="pdf")
    plt.close()


def plot_evaluation_trimmed_ci_in_fam_order(eval_trim: pd.DataFrame, fam_order: List[str], outfile: Path) -> None:
    """NEW: Plot trimmed evaluation bars (drop one min & one max) in familiarity order."""
    idx = eval_trim.set_index("university")
    ordered = idx.loc[fam_order].reset_index()

    x = np.arange(len(ordered))
    y = ordered["mean_centered_trim"].to_numpy()
    yerr = ordered["ci95_trim"].fillna(0).to_numpy()

    plt.figure(figsize=(12, 7))
    plt.bar(x, y, yerr=yerr, capsize=3)
    plt.xticks(x, ordered["university"], rotation=45, ha="right")
    plt.ylabel("Mean centered evaluation (trimmed)")
    # plt.title("Trimmed Mean Centered Evaluation (drop one min & one max per university; 95% CI)")
    plt.axhline(0.0)
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300, format="pdf")
    plt.close()


# -----------------------------
# Orchestration
# -----------------------------
def run(csv_path: str | Path, top: int | None, output_dir: str | Path) -> Tuple[Path, Path, Path, Path, Path, Path]:
    df = load_wide_csv(csv_path)
    long_df = wide_to_long(df)

    fam_summary = compute_familiarity_summary(long_df)
    eval_summary = compute_evaluation_summary(long_df)
    eval_trimmed = compute_evaluation_summary_trimmed(long_df)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine order/subset from familiarity
    fam_order, fam_top = get_fam_order(fam_summary, top)

    # Save summaries and order for reproducibility/audit
    fam_csv = output_dir / "familiarity_summary.csv"
    eval_csv = output_dir / "evaluation_summary_centered.csv"
    eval_trim_csv = output_dir / "evaluation_summary_centered__trimmed.csv"
    order_csv = output_dir / "xaxis_university_order_by_familiarity.csv"
    fam_top.to_csv(fam_csv, index=False)
    eval_summary.to_csv(eval_csv, index=False)
    eval_trimmed.to_csv(eval_trim_csv, index=False)
    pd.DataFrame({"university": fam_order}).to_csv(order_csv, index=False)

    # Plots
    fam_png = output_dir / "familiarity_mean_sd_bar.pdf"
    eval_png = output_dir / "evaluation_centered_mean_ci_bar__fam_order.pdf"
    eval_trim_png = output_dir / "evaluation_centered_mean_ci_bar__fam_order__trimmed.pdf"
    plot_familiarity_bars(fam_top, fam_png)
    plot_evaluation_ci_in_fam_order(eval_summary, fam_order, eval_png)
    plot_evaluation_trimmed_ci_in_fam_order(eval_trimmed, fam_order, eval_trim_png)

    return fam_csv, eval_csv, eval_trim_csv, order_csv, fam_png, eval_png, eval_trim_png


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="University survey plots")
    p.add_argument("--csv", required=True, help="Path to survey CSV (header in second row)")
    p.add_argument("--top", type=int, default=20, help="Show top N universities by familiarity; use 0 for all")
    p.add_argument("--output-dir", default="plots", help="Where to save CSVs and PNGs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    top_n = None if args.top == 0 else args.top
    paths = run(args.csv, top_n, args.output_dir)
    print("Saved:")
    for p in paths:
        print(f" - {p}")


if __name__ == "__main__":
    main()
