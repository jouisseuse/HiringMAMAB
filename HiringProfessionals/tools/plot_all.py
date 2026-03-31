# path: analysis_human_arm_pulls_by_probabilities.py
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

PROBABILITIES: Dict[str, float] = {
    "Trinity College Dublin": 0.9,
    "The University of Western Australia": 0.3,
    "University of Glasgow": 0.7,
    "Heidelberg University": 0.1,
    "University of Adelaide": 0.5,
    "University of Leeds": 0.9,
    "University of Southampton": 0.3,
    "University of Sheffield": 0.5,
    "University of Nottingham": 0.7,
    "Karlsruhe Institute of Technology": 0.1,
}

color_schemes = {
    "Com": ("#3594cc", "#8cc5e3"),  # Med Blue & Light Blue
    "Non": ("#ea801c", "#f0b077"),  # Med Orange & Light Orange
}

COLOR_COMM = color_schemes["Com"][0]  # Social (normal)
COLOR_NCOM = color_schemes["Non"][0]  # Asocial (normal)


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_player_round_csv(path: Path, expected_agents: int = 10, drop_last_row: bool = True) -> List[List[str]]:
    rows: List[List[str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i < 2:
                continue
            if not row:
                continue
            fields = [x.strip() for x in row[1:] if x is not None]
            if len(fields) < expected_agents:
                raise ValueError(f"{path}: row {i+1} has {len(fields)} agent fields, expected >= {expected_agents}")
            rows.append(fields[:expected_agents])

    if drop_last_row and rows:
        rows = rows[:-1]

    if not rows:
        raise ValueError(f"{path}: no usable rounds after skipping headers and dropping last row.")
    return rows


def discover_csvs(base_folder: Path) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for condition in ("social", "asocial"):
        cond_dir = base_folder / condition
        if not cond_dir.is_dir():
            continue
        hiring_dirs = sorted({*cond_dir.glob("Hiring-game-*"), *cond_dir.glob("hiring-game-*")})
        for hiring_dir in hiring_dirs:
            p = hiring_dir / "playerRound_processed.csv"
            if p.is_file():
                out.append((condition, p))
    return out


def count_choices(choices_by_round: List[List[str]]) -> Counter[str]:
    c: Counter[str] = Counter()
    for rnd in choices_by_round:
        for v in rnd:
            k = v.strip()
            if k:
                c[k] += 1
    return c


def order_schools(
    counts: Counter[str],
    probabilities: Dict[str, float],
    unknown_order: str = "count_desc",
) -> List[str]:
    known = sorted(probabilities.items(), key=lambda kv: (-kv[1], kv[0]))
    known_names = [name for name, _p in known]
    known_set = set(known_names)

    unknown = [k for k in counts.keys() if k not in known_set]
    if unknown_order == "name_asc":
        unknown_sorted = sorted(unknown)
    else:
        unknown_sorted = sorted(unknown, key=lambda x: counts.get(x, 0), reverse=True)

    return known_names + unknown_sorted


def top_k_by_probability(probabilities: Dict[str, float], k: int) -> List[str]:
    k = max(1, int(k))
    known = sorted(probabilities.items(), key=lambda kv: (-kv[1], kv[0]))
    return [name for name, _p in known[: min(k, len(known))]]


def write_counts_csv(out_path: Path, ordered: List[str], counts: Counter[str], probabilities: Dict[str, float]) -> None:
    _ensure_dir(out_path)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["School", "Probability", "Count"])
        for s in ordered:
            w.writerow([s, probabilities.get(s, ""), int(counts.get(s, 0))])


def plot_counts_pdf(out_path: Path, title: str, ordered: List[str], counts: Counter[str], bar_color: str) -> None:
    _ensure_dir(out_path)
    values = np.asarray([counts.get(s, 0) for s in ordered], dtype=float)
    x = np.arange(len(ordered), dtype=float)

    fig_w = max(10.0, 0.35 * len(ordered))
    plt.figure(figsize=(fig_w, 4.0))
    ax = plt.gca()

    ax.bar(x, values, edgecolor="black", width=0.7, linewidth=0.6, color=bar_color)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered, rotation=45, ha="right")
    ax.set_ylabel("# Pulls", fontsize=12)
    ax.set_title(title, fontsize=12)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="both", direction="out", length=3, width=0.8)

    ymax = float(values.max(initial=0.0))
    pad = 0.01 * max(1.0, ymax)
    for xi, v in zip(x, values):
        ax.text(float(xi), float(v) + pad, f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True, format="pdf")
    plt.close()


def _pct(value: int, total: float) -> float:
    return 0.0 if total <= 0.0 else 100.0 * float(value) / total


def _entropy_bits_from_counts(counts: List[int]) -> float:
    total = float(sum(counts))
    if total <= 0.0:
        return 0.0
    ps = [c / total for c in counts if c > 0]
    return float(-sum(p * math.log(p, 2) for p in ps))


def _entropy_bits_from_percentages(pcts: List[float]) -> float:
    total = float(sum(pcts))
    if total <= 0.0:
        return 0.0
    ps = [p / total for p in pcts if p > 0.0]
    return float(-sum(p * math.log(p, 2) for p in ps))


def _rank_aggregate_topk(counts: Counter[str], top_schools: List[str]) -> List[int]:
    """Per-CSV: take counts for top_schools, sort desc, return [rank1, rank2, ...]."""
    vals = [int(counts.get(s, 0)) for s in top_schools]
    vals.sort(reverse=True)
    return vals


def write_combined_csv_rank_topk(
    out_path: Path,
    k: int,
    asocial_rank_counts: List[int],
    social_rank_counts: List[int],
    asocial_total_pulls: int,
    social_total_pulls: int,
) -> None:
    _ensure_dir(out_path)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["RankLabel", "AsocialCount", "AsocialPct", "SocialCount", "SocialPct"])
        for i in range(k):
            label = "High" if i == 0 else ("Low" if i == 1 and k == 2 else f"Rank{i+1}")
            ac = int(asocial_rank_counts[i]) if i < len(asocial_rank_counts) else 0
            sc = int(social_rank_counts[i]) if i < len(social_rank_counts) else 0
            w.writerow(
                [
                    label,
                    ac,
                    f"{_pct(ac, float(asocial_total_pulls)):.4f}",
                    sc,
                    f"{_pct(sc, float(social_total_pulls)):.4f}",
                ]
            )


def plot_combined_rank_topk_split_conditions_percent(
    out_path: Path,
    title: str,
    k: int,
    asocial_rank_counts: List[int],
    social_rank_counts: List[int],
    asocial_total_pulls: int,
    social_total_pulls: int,
) -> None:
    """Existing pooled plot: percentages from pooled totals."""
    _ensure_dir(out_path)

    asocial_pct = np.asarray([_pct(int(asocial_rank_counts[i]), float(asocial_total_pulls)) for i in range(k)], dtype=float)
    social_pct = np.asarray([_pct(int(social_rank_counts[i]), float(social_total_pulls)) for i in range(k)], dtype=float)

    arm_percentages = np.concatenate([asocial_pct, social_pct], axis=0)

    non_med, non_light = color_schemes["Non"]
    com_med, com_light = color_schemes["Com"]

    asocial_colors = [non_med if i == 0 else non_light for i in range(k)]
    social_colors = [com_med if i == 0 else com_light for i in range(k)]
    colors = asocial_colors + social_colors

    left_pos = np.arange(k, dtype=float)
    right_pos = np.arange(k, dtype=float) + k + 1.0
    x_positions = np.concatenate([left_pos, right_pos], axis=0)

    ent_a = _entropy_bits_from_counts([int(asocial_rank_counts[i]) for i in range(k)])
    ent_s = _entropy_bits_from_counts([int(social_rank_counts[i]) for i in range(k)])

    plt.figure(figsize=(4, 4))
    plt.bar(x_positions, arm_percentages, color=colors, width=0.7, edgecolor="none")

    y_limit = 100.0
    plt.ylim(0, y_limit)
    plt.ylabel("Selection Percentage (%)", fontsize=12, fontname="Arial")
    plt.xticks(x_positions, [], rotation=45, fontsize=10, fontname="Arial")

    for i, percentage in enumerate(arm_percentages):
        plt.text(
            float(x_positions[i]),
            float(percentage) + 1.0,
            f"{percentage:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontname="Arial",
        )

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    asocial_center = float(left_pos.mean()) if k > 0 else 0.0
    social_center = float(right_pos.mean()) if k > 0 else 0.0
    ax.text(
        asocial_center,
        -0.05,
        "Asocial",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=11,
        fontname="Arial",
    )
    ax.text(
        social_center,
        -0.05,
        "Social",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=11,
        fontname="Arial",
    )

    plt.text(
        0.0 * float(x_positions.max(initial=0.0)),
        0.9 * y_limit,
        f"Entropy: {ent_a:.2f}",
        fontsize=10,
        fontname="Arial",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    plt.text(
        0.75 * float(x_positions.max(initial=0.0)),
        0.9 * y_limit,
        f"Entropy: {ent_s:.2f}",
        fontsize=10,
        fontname="Arial",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.32)
    plt.savefig(out_path, dpi=450, bbox_inches="tight", transparent=True, format="pdf")
    plt.close()


_T_CRIT_975_DF_1_TO_30 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _t_crit_975(df: int) -> float:
    """Two-sided 95% CI => t_{0.975, df}. Falls back to ~1.96 for large df."""
    if df <= 0:
        return 0.0
    if df in _T_CRIT_975_DF_1_TO_30:
        return _T_CRIT_975_DF_1_TO_30[df]
    return 1.96


def _mean_ci95(values: List[float]) -> Tuple[float, float, float]:
    """
    Returns (mean, ci_low, ci_high) using t-interval.
    If n < 2, CI collapses to mean.
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = float(sum(values)) / float(n)
    if n < 2:
        return mean, mean, mean

    var = float(sum((v - mean) ** 2 for v in values)) / float(n - 1)
    se = math.sqrt(var) / math.sqrt(float(n))
    t = _t_crit_975(n - 1)
    half = t * se
    return mean, mean - half, mean + half


def write_combined_csv_rank_topk_mean_ci(
    out_path: Path,
    k: int,
    asocial_rank_pcts: List[List[float]],
    social_rank_pcts: List[List[float]],
) -> None:
    _ensure_dir(out_path)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "RankLabel",
                "AsocialN",
                "AsocialMeanPct",
                "AsocialCI95_Low",
                "AsocialCI95_High",
                "SocialN",
                "SocialMeanPct",
                "SocialCI95_Low",
                "SocialCI95_High",
            ]
        )

        for i in range(k):
            label = "High" if i == 0 else ("Low" if i == 1 and k == 2 else f"Rank{i+1}")
            a_vals = asocial_rank_pcts[i] if i < len(asocial_rank_pcts) else []
            s_vals = social_rank_pcts[i] if i < len(social_rank_pcts) else []
            a_mean, a_lo, a_hi = _mean_ci95(a_vals)
            s_mean, s_lo, s_hi = _mean_ci95(s_vals)
            w.writerow(
                [
                    label,
                    len(a_vals),
                    f"{a_mean:.6f}",
                    f"{a_lo:.6f}",
                    f"{a_hi:.6f}",
                    len(s_vals),
                    f"{s_mean:.6f}",
                    f"{s_lo:.6f}",
                    f"{s_hi:.6f}",
                ]
            )


def plot_combined_rank_topk_split_conditions_mean_ci95(
    out_path: Path,
    title: str,
    k: int,
    asocial_rank_pcts: List[List[float]],
    social_rank_pcts: List[List[float]],
) -> None:
    """New plot: per-CSV mean percentages with 95% CI error bars."""
    _ensure_dir(out_path)

    a_means: List[float] = []
    a_err: List[float] = []
    s_means: List[float] = []
    s_err: List[float] = []

    for i in range(k):
        a_mean, a_lo, a_hi = _mean_ci95(asocial_rank_pcts[i])
        s_mean, s_lo, s_hi = _mean_ci95(social_rank_pcts[i])
        a_means.append(a_mean)
        s_means.append(s_mean)
        a_err.append(max(0.0, a_hi - a_mean))
        s_err.append(max(0.0, s_hi - s_mean))

    arm_means = np.asarray(a_means + s_means, dtype=float)
    arm_err = np.asarray(a_err + s_err, dtype=float)

    non_med, non_light = color_schemes["Non"]
    com_med, com_light = color_schemes["Com"]
    asocial_colors = [non_med if i == 0 else non_light for i in range(k)]
    social_colors = [com_med if i == 0 else com_light for i in range(k)]
    colors = asocial_colors + social_colors

    left_pos = np.arange(k, dtype=float)
    right_pos = np.arange(k, dtype=float) + k + 1.0
    x_positions = np.concatenate([left_pos, right_pos], axis=0)

    ent_a = _entropy_bits_from_percentages(a_means)
    ent_s = _entropy_bits_from_percentages(s_means)

    plt.figure(figsize=(4, 4))
    plt.bar(x_positions, arm_means, color=colors, width=0.7, edgecolor="none")

    # Error bars for 95% CI
    plt.errorbar(
        x_positions,
        arm_means,
        yerr=arm_err,
        fmt="none",
        ecolor="black",
        elinewidth=1.0,
        capsize=3,
        capthick=1.0,
    )

    y_limit = 100.0
    plt.ylim(0, y_limit)
    plt.ylabel("Selection Percentage (%)", fontsize=12, fontname="Arial")
    plt.xticks(x_positions, [], rotation=45, fontsize=10, fontname="Arial")

    for i, mean_pct in enumerate(arm_means):
        plt.text(
            float(x_positions[i]),
            float(mean_pct) + 6.0,
            f"{mean_pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontname="Arial",
        )

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    asocial_center = float(left_pos.mean()) if k > 0 else 0.0
    social_center = float(right_pos.mean()) if k > 0 else 0.0
    ax.text(
        asocial_center,
        -0.05,
        "Asocial",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=11,
        fontname="Arial",
    )
    ax.text(
        social_center,
        -0.05,
        "Social",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=11,
        fontname="Arial",
    )

    plt.text(
        0.0 * float(x_positions.max(initial=0.0)),
        0.9 * y_limit,
        f"Entropy: {ent_a:.2f}",
        fontsize=10,
        fontname="Arial",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    plt.text(
        0.75 * float(x_positions.max(initial=0.0)),
        0.9 * y_limit,
        f"Entropy: {ent_s:.2f}",
        fontsize=10,
        fontname="Arial",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    # Title intentionally omitted in original style; kept consistent.
    _ = title

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.32)
    plt.savefig(out_path, dpi=450, bbox_inches="tight", transparent=True, format="pdf")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-CSV arm pull plots + combined rank-aggregated top-K plot.")
    ap.add_argument("base_folder", type=str, help="Folder containing social/ and asocial/ dirs.")
    ap.add_argument(
        "--unknown-order",
        type=str,
        default="count_desc",
        choices=["count_desc", "name_asc"],
        help="How to append schools not present in PROBABILITIES (per-file only).",
    )
    ap.add_argument(
        "--combined-top-k",
        type=int,
        default=2,
        help="Top-K schools by PROBABILITIES used for rank aggregation in combined plot/csv.",
    )
    args = ap.parse_args()

    base = Path(args.base_folder)
    if not base.is_dir():
        raise SystemExit(f"base_folder not found or not a directory: {base}")

    targets = discover_csvs(base)
    if not targets:
        raise SystemExit("No playerRound_processed.csv found under social/ or asocial/")

    top_schools = top_k_by_probability(PROBABILITIES, args.combined_top_k)
    k = len(top_schools)

    # Pooled (existing)
    asocial_rank_totals = [0] * k
    social_rank_totals = [0] * k
    asocial_total_pulls = 0
    social_total_pulls = 0

    # Per-session percentages for mean+CI
    asocial_rank_pcts: List[List[float]] = [[] for _ in range(k)]
    social_rank_pcts: List[List[float]] = [[] for _ in range(k)]

    skipped = 0
    for condition, csv_path in targets:
        try:
            choices = read_player_round_csv(csv_path, expected_agents=10, drop_last_row=True)
            counts = count_choices(choices)

            hiring_dir = csv_path.parent
            out_pdf = hiring_dir / f"arm_pulls_by_probabilities_{condition}.pdf"
            out_csv = hiring_dir / f"arm_pulls_by_probabilities_{condition}.csv"
            ordered = order_schools(counts, PROBABILITIES, unknown_order=args.unknown_order)

            title = f"{condition.upper()} | {hiring_dir.name} | Arm pulls ordered by PROBABILITIES"
            bar_color = COLOR_COMM if condition == "social" else COLOR_NCOM
            plot_counts_pdf(out_pdf, title, ordered, counts, bar_color=bar_color)
            write_counts_csv(out_csv, ordered, counts, PROBABILITIES)

            ranked = _rank_aggregate_topk(counts, top_schools)
            pulls_this_csv = int(sum(counts.values()))
            if pulls_this_csv <= 0:
                raise ValueError("total pulls is 0")

            if condition == "asocial":
                asocial_total_pulls += pulls_this_csv
                for i in range(k):
                    asocial_rank_totals[i] += int(ranked[i])
                    asocial_rank_pcts[i].append(_pct(int(ranked[i]), float(pulls_this_csv)))
            else:
                social_total_pulls += pulls_this_csv
                for i in range(k):
                    social_rank_totals[i] += int(ranked[i])
                    social_rank_pcts[i].append(_pct(int(ranked[i]), float(pulls_this_csv)))

            print(f"[OK] {csv_path}")
            print(f"     -> {out_pdf}")
            print(f"     -> {out_csv}")
        except Exception as e:
            skipped += 1
            print(f"[WARN] Skipping {csv_path}: {e}")

    # Existing pooled combined outputs (unchanged)
    combined_pdf = base / "arm_pulls_by_probabilities_combined.pdf"
    combined_csv = base / "arm_pulls_by_probabilities_combined.csv"
    combined_title = f"COMBINED | Top {k} by PROBABILITIES | Rank-aggregated (pooled)"
    plot_combined_rank_topk_split_conditions_percent(
        combined_pdf,
        combined_title,
        k,
        asocial_rank_totals,
        social_rank_totals,
        asocial_total_pulls,
        social_total_pulls,
    )
    write_combined_csv_rank_topk(
        combined_csv,
        k,
        asocial_rank_totals,
        social_rank_totals,
        asocial_total_pulls,
        social_total_pulls,
    )

    # New mean+CI combined outputs
    combined_ci_pdf = base / "arm_pulls_by_probabilities_combined_mean_ci.pdf"
    combined_ci_csv = base / "arm_pulls_by_probabilities_combined_mean_ci.csv"
    combined_ci_title = f"COMBINED | Top {k} by PROBABILITIES | Rank-aggregated (mean ± 95% CI)"

    plot_combined_rank_topk_split_conditions_mean_ci95(
        combined_ci_pdf,
        combined_ci_title,
        k,
        asocial_rank_pcts,
        social_rank_pcts,
    )
    write_combined_csv_rank_topk_mean_ci(
        combined_ci_csv,
        k,
        asocial_rank_pcts,
        social_rank_pcts,
    )

    print(f"[OK] Combined (pooled)")
    print(f"     -> {combined_pdf}")
    print(f"     -> {combined_csv}")
    print(f"[OK] Combined (mean ± 95% CI)")
    print(f"     -> {combined_ci_pdf}")
    print(f"     -> {combined_ci_csv}")
    print(f"[DONE] skipped={skipped}")


if __name__ == "__main__":
    main()