# file: analyze_demographics_and_balance.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# =========================
# Configuration
# =========================

DEMOGRAPHIC_COLS = {
    "age": "survey__age",
    "gender": "survey__gender",
    "race": "survey__race",
    "education": "survey__education",
    "political": "survey__politicalOrientation",
}

PROFESSIONAL_COLS = [
    "survey__familiarityUniversities",
    "survey__employmentStatus",
    "survey__employmentStatusOther",
    "survey__hiringInvolved",
    "survey__hiringExperienceYears",
]

OUTCOME_COL = "entropy"

# Change this if your treatment variable has a different name
TREATMENT_COL = "experimentFolder"

# NEW: binary covariate for file source
CONDITION_COL = "isSocial"  # social=1, asocial=0


# =========================
# Helpers
# =========================

def load_all(csv_files: List[Path]) -> pd.DataFrame:
    """Legacy loader (no isSocial column)."""
    dfs = [pd.read_csv(p) for p in csv_files]
    return pd.concat(dfs, ignore_index=True)


def _read_csvs(paths: Iterable[Path]) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        df["_source_file"] = str(p)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_social_asocial(social_files: List[Path], asocial_files: List[Path]) -> pd.DataFrame:
    """
    Load two sets of inputs and add a binary covariate:
      - isSocial = 1 for social files
      - isSocial = 0 for asocial files
    """
    df_social = _read_csvs(social_files)
    df_asocial = _read_csvs(asocial_files)

    if df_social.empty and df_asocial.empty:
        raise ValueError("No input files provided.")

    if not df_social.empty:
        df_social[CONDITION_COL] = 1
    if not df_asocial.empty:
        df_asocial[CONDITION_COL] = 0

    df = pd.concat([df_social, df_asocial], ignore_index=True)
    df[CONDITION_COL] = pd.to_numeric(df[CONDITION_COL], errors="coerce").fillna(0).astype(int)
    return df


def pct(series: pd.Series) -> pd.Series:
    return (series.value_counts(normalize=True) * 100).round(2)


def clean_str(x):
    if pd.isna(x):
        return np.nan
    return str(x).strip()


def hiring_year_bin(x):
    if pd.isna(x):
        return np.nan
    try:
        x = float(x)
    except Exception:
        return np.nan

    if x < 1:
        return "<1 year"
    elif x <= 3:
        return "1–3 years"
    elif x <= 6:
        return "4–6 years"
    else:
        return "7+ years"


def hiring_involved_flag(series: pd.Series) -> pd.Series:
    s = series.map(lambda v: "" if pd.isna(v) else str(v).strip()).str.lower()
    return s.isin(["yes", "true", "1", "y", "是"])


# =========================
# Descriptive statistics
# =========================

def descriptive_stats(df: pd.DataFrame, out_dir: Path) -> None:
    rows = []

    age = pd.to_numeric(df[DEMOGRAPHIC_COLS["age"]], errors="coerce")
    rows.append({
        "variable": "age",
        "mean": round(age.mean(), 2),
        "min": int(age.min()) if not np.isnan(age.min()) else np.nan,
        "max": int(age.max()) if not np.isnan(age.max()) else np.nan,
    })

    for k, col in DEMOGRAPHIC_COLS.items():
        if k == "age":
            continue
        dist = pct(df[col].map(clean_str))
        for cat, val in dist.items():
            rows.append({
                "variable": k,
                "category": cat,
                "percent": val,
            })

    demo_df = pd.DataFrame(rows)
    demo_df.to_csv(out_dir / "demographic_descriptive_stats.csv", index=False)


# =========================
# Balance test (OLS)
# =========================

def balance_test(df: pd.DataFrame, out_dir: Path) -> None:
    df = df.copy()

    df["age"] = pd.to_numeric(df[DEMOGRAPHIC_COLS["age"]], errors="coerce")

    for k, col in DEMOGRAPHIC_COLS.items():
        if k != "age":
            df[k] = df[col].astype("category")

    if CONDITION_COL not in df.columns:
        df[CONDITION_COL] = 0
    df[CONDITION_COL] = pd.to_numeric(df[CONDITION_COL], errors="coerce").fillna(0).astype(int)

    formula = (
        f"{OUTCOME_COL} ~ "
        f"{CONDITION_COL} + "
        "age + C(gender) + C(race) + "
        "C(education) + C(political) + "
        f"C({TREATMENT_COL})"
    )

    model = smf.ols(formula=formula, data=df).fit()

    summary_df = model.summary2().tables[1]
    summary_df.to_csv(out_dir / "balance_test_regression.csv")

    with open(out_dir / "balance_test_summary.txt", "w") as f:
        f.write(f"R^2 = {model.rsquared:.3f}\n")
        f.write(f"Adj. R^2 = {model.rsquared_adj:.3f}\n\n")
        f.write(str(model.summary()))


# =========================
# Professional analysis
# =========================

def professional_stats(df: pd.DataFrame, out_dir: Path) -> None:
    import ast

    rows = []
    n_total = len(df)

    def _clean_str(x):
        if pd.isna(x):
            return ""
        return str(x).strip()

    def parse_list_cell(x) -> list[str]:
        s = _clean_str(x)
        if not s:
            return []
        if isinstance(x, list):
            return [str(i).strip() for i in x if str(i).strip()]
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return [str(i).strip() for i in v if str(i).strip()]
            return []
        except Exception:
            return []

    def is_none_of_above(items: list[str]) -> bool:
        lowered = {i.lower() for i in items}
        return "none of the above" in lowered

    def normalize_hiring_years_text(x) -> str | None:
        s = _clean_str(x)
        if not s:
            return None

        s_norm = (
            s.replace("–", "-")
             .replace("—", "-")
             .replace("to", "-")
             .lower()
             .strip()
        )

        if "less" in s_norm or "<1" in s_norm or "under 1" in s_norm or "0" == s_norm:
            return "Less than 1 year"
        if "1-3" in s_norm or "1 - 3" in s_norm or "1–3" in s or "1- 3" in s_norm:
            return "1–3 years"
        if "4-6" in s_norm or "4 - 6" in s_norm or "4–6" in s:
            return "4–6 years"
        if "7+" in s_norm or ("7" in s_norm and "+" in s_norm):
            return "7+ years"
        if s_norm.isdigit() and int(s_norm) >= 7:
            return "7+ years"

        return None

    fam_lists = df["survey__familiarityUniversities"].apply(parse_list_cell)

    has_fam_mask = fam_lists.apply(lambda items: bool(items) and not is_none_of_above(items))
    n_fam = int(has_fam_mask.sum())

    rows.append({
        "section": "familiar_university_overall",
        "metric": "Has familiar university",
        "count": n_fam,
        "percent": round(n_fam / n_total * 100, 2) if n_total else 0.0,
    })

    fam_exploded = (
        pd.DataFrame({"university": fam_lists[has_fam_mask]})
        .explode("university")["university"]
        .map(_clean_str)
    )
    fam_exploded = fam_exploded[(fam_exploded != "") & (fam_exploded.str.lower() != "none of the above")]

    uni_counts = fam_exploded.value_counts(dropna=True)
    uni_df = uni_counts.reset_index()
    uni_df.columns = ["university", "count"]
    uni_df["percent_of_familiar"] = (uni_df["count"] / n_fam * 100).round(2) if n_fam else 0.0
    uni_df.to_csv(out_dir / "professional_familiar_universities_distribution.csv", index=False)

    fam_df = df.loc[has_fam_mask].copy()
    fam_hiring = hiring_involved_flag(fam_df["survey__hiringInvolved"])
    rows.append({
        "section": "familiar_university_hiring",
        "metric": "Hiring involved among familiar university",
        "count": int(fam_hiring.sum()),
        "percent": round(float(fam_hiring.mean() * 100), 2) if len(fam_hiring) else 0.0,
    })

    emp = df["survey__employmentStatus"].map(_clean_str).replace({"": np.nan})
    emp_dist = (emp.value_counts(normalize=True, dropna=True) * 100).round(2)
    emp_df = emp_dist.reset_index()
    emp_df.columns = ["employmentStatus", "percent"]
    emp_df.to_csv(out_dir / "professional_employment_status_distribution.csv", index=False)

    hiring = hiring_involved_flag(df["survey__hiringInvolved"])
    rows.append({
        "section": "hiring_overall",
        "metric": "Hiring involved",
        "count": int(hiring.sum()),
        "percent": round(float(hiring.mean() * 100), 2) if len(hiring) else 0.0,
    })

    years_bin = df["survey__hiringExperienceYears"].apply(normalize_hiring_years_text)
    order = ["Less than 1 year", "1–3 years", "4–6 years", "7+ years"]

    year_counts = years_bin.value_counts(dropna=True)
    year_df = pd.DataFrame({"hiringExperienceYearsBin": order})
    year_df["count"] = year_df["hiringExperienceYearsBin"].map(year_counts).fillna(0).astype(int)

    denom = int(year_counts.sum())
    year_df["percent"] = (year_df["count"] / denom * 100).round(2) if denom else 0.0
    year_df.to_csv(out_dir / "professional_hiring_experience_distribution.csv", index=False)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "professional_summary_stats.csv", index=False)


def professional_balance_test(df: pd.DataFrame, out_dir: Path) -> None:
    df = df.copy()

    if CONDITION_COL not in df.columns:
        df[CONDITION_COL] = 0
    df[CONDITION_COL] = pd.to_numeric(df[CONDITION_COL], errors="coerce").fillna(0).astype(int)

    def has_familiar(x):
        if pd.isna(x):
            return False
        s = str(x).lower()
        return "none of the above" not in s and s.strip() != "[]"

    df["hasFamiliarUniversity"] = df["survey__familiarityUniversities"].apply(has_familiar)

    df["hiringInvolvedFlag"] = hiring_involved_flag(df["survey__hiringInvolved"])

    def normalize_years(x):
        if pd.isna(x):
            return "Missing"
        s = str(x).strip().replace("–", "-").lower()
        if "less" in s or "<1" in s:
            return "Less than 1 year"
        if "1-3" in s:
            return "1–3 years"
        if "4-6" in s:
            return "4–6 years"
        if "7+" in s or ("7" in s and "+" in s):
            return "7+ years"
        return "Other"

    df["hiringYearsCat"] = df["survey__hiringExperienceYears"].apply(normalize_years)

    df["employmentStatusCat"] = (
        df["survey__employmentStatus"]
        .astype(str)
        .str.strip()
        .replace("", "Missing")
        .astype("category")
    )

    formula = (
        "entropy ~ "
        f"{CONDITION_COL} + "
        "hasFamiliarUniversity + "
        "hiringInvolvedFlag + "
        "C(hiringYearsCat) + "
        "C(employmentStatusCat) + "
        f"C({TREATMENT_COL})"
    )

    model = smf.ols(formula=formula, data=df).fit()

    coef_df = model.summary2().tables[1]
    coef_df.to_csv(out_dir / "professional_balance_test_regression.csv")

    with open(out_dir / "professional_balance_test_summary.txt", "w") as f:
        f.write(f"R^2 = {model.rsquared:.3f}\n")
        f.write(f"Adj. R^2 = {model.rsquared_adj:.3f}\n\n")
        f.write(str(model.summary()))

    print("✔ Professional balance test completed")


# =========================
# Hiring-years effect on entropy (OLS)
# =========================

def hiring_involved_effect_regression(df: pd.DataFrame, out_dir: Path) -> None:
    """
    OLS regression to assess how hiring experience (hiringYearsCat),
    relative to Missing, is associated with entropy.

    Reference category:
      - hiringYearsCat == "Missing"

    Includes:
      - isSocial as a binary covariate (social=1, asocial=0)
    Excludes:
      - No C(experimentFolder)
    """
    df = df.copy()

    if CONDITION_COL not in df.columns:
        df[CONDITION_COL] = 0
    df[CONDITION_COL] = pd.to_numeric(df[CONDITION_COL], errors="coerce").fillna(0).astype(int)

    # Outcome
    df["entropy"] = pd.to_numeric(df["entropy"], errors="coerce")

    # Demographics
    df["age"] = pd.to_numeric(df[DEMOGRAPHIC_COLS["age"]], errors="coerce")
    df["gender"] = df[DEMOGRAPHIC_COLS["gender"]].map(clean_str).astype("category")
    df["race"] = df[DEMOGRAPHIC_COLS["race"]].map(clean_str).astype("category")
    df["education"] = df[DEMOGRAPHIC_COLS["education"]].map(clean_str).astype("category")
    df["political"] = df[DEMOGRAPHIC_COLS["political"]].map(clean_str).astype("category")

    # Professional controls
    def has_familiar(x) -> bool:
        if pd.isna(x):
            return False
        s = str(x).lower().strip()
        return s != "[]" and "none of the above" not in s

    df["hasFamiliarUniversity"] = (
        df["survey__familiarityUniversities"]
        .apply(has_familiar)
        .astype(int)
    )

    df["employmentStatusCat"] = (
        df["survey__employmentStatus"]
        .astype(str)
        .str.strip()
        .replace("", "Missing")
        .astype("category")
    )

    def normalize_years(x) -> str:
        if pd.isna(x):
            return "Missing"
        s = str(x).strip().replace("–", "-").lower()
        if "less" in s or "<1" in s:
            return "Less than 1 year"
        if "1-3" in s or "1 - 3" in s:
            return "1–3 years"
        if "4-6" in s or "4 - 6" in s:
            return "4–6 years"
        if "7+" in s or ("7" in s and "+" in s):
            return "7+ years"
        return "Other"

    df["hiringYearsCat"] = (
        df["survey__hiringExperienceYears"]
        .apply(normalize_years)
        .astype("category")
    )

    model = smf.ols(
        "entropy ~ "
        f"{CONDITION_COL} + "
        "age + C(gender) + C(race) + C(education) + C(political) + "
        "hasFamiliarUniversity + "
        "C(hiringYearsCat, Treatment(reference='Missing')) + "
        "C(employmentStatusCat)",
        data=df,
    ).fit()

    model.summary2().tables[1].to_csv(out_dir / "hiring_years_effect_regression.csv")

    with open(out_dir / "hiring_years_effect_summary.txt", "w") as f:
        f.write("Hiring experience (relative to Missing) → Entropy (OLS)\n")
        f.write("========================================================\n\n")
        f.write(f"R^2 = {model.rsquared:.3f}\n")
        f.write(f"Adj. R^2 = {model.rsquared_adj:.3f}\n\n")
        f.write(str(model.summary()))

    print("✔ Hiring-years effect regression completed (no experimentFolder)")


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--social",
        nargs="+",
        type=Path,
        required=True,
        help="social matched_entropy_with_survey.csv files",
    )
    parser.add_argument(
        "--asocial",
        nargs="+",
        type=Path,
        required=True,
        help="asocial matched_entropy_with_survey.csv files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_social_asocial(args.social, args.asocial)

    # descriptive_stats(df, args.out_dir)
    # balance_test(df, args.out_dir)
    professional_stats(df, args.out_dir)
    professional_balance_test(df, args.out_dir)

    # NEW
    hiring_involved_effect_regression(df, args.out_dir)

    print("✔ Demographic stats saved")
    print("✔ Balance test completed")
    print("✔ Professional analysis completed")


if __name__ == "__main__":
    main()