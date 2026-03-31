# path: analysis_entropy_human_subset.py
"""
Human experiment analysis (social vs asocial), subset-only entropy + cumulative optimal arm rate + cumulative reward.

Adds:
  - Final-round % change (Social vs Asocial) for all metrics
  - Trend CSV includes mean diff and diff pct by round
  - OLS b and CI also exported as percentage relative to Asocial final-round mean
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
import statsmodels.api as sm


# -----------------------------
# Stats helpers
# -----------------------------

def shannon_entropy_base2(counts: Sequence[float]) -> float:
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    probs = [c / total for c in counts if c > 0.0]
    return float(-sum(p * math.log2(p) for p in probs))


def _mean_ci95_1d(x: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    m = float(np.mean(x)) if x.size else 0.0
    if x.size > 1:
        sem = float(np.std(x, ddof=1) / math.sqrt(x.size))
        margin = sem * float(t.ppf(0.975, x.size - 1))
        return m, m - margin, m + margin
    return m, m, m


def mean_ci95_curves(curves: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not curves:
        raise ValueError("No curves to aggregate.")
    L = min(len(c) for c in curves)
    A = np.vstack([c[:L] for c in curves])
    mean = A.mean(axis=0)
    if A.shape[0] > 1:
        sem = A.std(axis=0, ddof=1) / np.sqrt(A.shape[0])
        margin = sem * t.ppf(0.975, A.shape[0] - 1)
        lo, hi = mean - margin, mean + margin
    else:
        lo, hi = mean.copy(), mean.copy()
    return mean, lo, hi


def final_round_ols(y_social: np.ndarray, y_asocial: np.ndarray) -> Tuple[float, float, float, float]:
    """
    OLS: y ~ 1 + SocialDummy, where Asocial=0, Social=1.
    Returns: (b, ci_low, ci_high, p_value).
    """
    y_social = np.asarray(y_social, dtype=float)
    y_asocial = np.asarray(y_asocial, dtype=float)

    y = np.concatenate([y_asocial, y_social], axis=0)
    social_dummy = np.concatenate(
        [np.zeros_like(y_asocial, dtype=float), np.ones_like(y_social, dtype=float)],
        axis=0,
    )

    X = sm.add_constant(social_dummy)
    model = sm.OLS(y, X).fit()

    b = float(model.params[1])

    ci_raw = model.conf_int()
    ci_arr = np.asarray(ci_raw, dtype=float)
    ci_low, ci_high = float(ci_arr[1, 0]), float(ci_arr[1, 1])

    p_raw = model.pvalues
    p_arr = np.asarray(p_raw, dtype=float)
    p = float(p_arr[1])

    return b, ci_low, ci_high, p


def _pct_change(new_value: float, base_value: float) -> float:
    """
    Percent change relative to base_value:
      (new - base) / abs(base) * 100
    If base==0: return 0 if both 0 else NaN.
    """
    if base_value == 0.0:
        return 0.0 if new_value == 0.0 else float("nan")
    return (new_value - base_value) / abs(base_value) * 100.0


def _pct_of_base(delta: float, base_value: float) -> float:
    """delta / abs(base) * 100; safe for base==0."""
    if base_value == 0.0:
        return 0.0 if delta == 0.0 else float("nan")
    return delta / abs(base_value) * 100.0


# -----------------------------
# IO helpers
# -----------------------------

def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _slugify_subset(subset: Sequence[str], max_len: int = 80) -> str:
    raw = "_".join(s.strip() for s in subset if s.strip())
    raw = re.sub(r"\s+", "_", raw)
    raw = re.sub(r"[^A-Za-z0-9_\-]+", "", raw)
    return (raw or "subset")[:max_len]


def _read_subset_from_file(path: Path) -> List[str]:
    items: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                items.append(s)
    return items


# -----------------------------
# Data model + parsing
# -----------------------------

@dataclass(frozen=True)
class RepeatData:
    condition: str  # "social" | "asocial"
    repeat_id: str
    csv_path: Path
    log_path: Optional[Path]
    choices_by_round: List[List[str]]  # [round][agent]
    rewards_by_round: Optional[List[List[float]]]  # [round][agent]


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


def _load_json_robust(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def read_player_round_log_rewards(path: Path, expected_agents: int = 10) -> List[List[float]]:
    data = _load_json_robust(path)
    results = data.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError(f"{path}: missing or empty 'results' list.")

    rewards_by_round: List[List[float]] = []
    for idx, entry in enumerate(results, start=1):
        rewards = entry.get("rewards")
        if not isinstance(rewards, list) or len(rewards) < expected_agents:
            raise ValueError(f"{path}: round {idx} rewards invalid; expected list length >= {expected_agents}.")
        rewards_by_round.append([float(x) for x in rewards[:expected_agents]])
    return rewards_by_round


def _iter_hiring_dirs(cond_dir: Path) -> List[Path]:
    out = set()
    for pat in ("Hiring-game-*", "hiring-game-*"):
        for p in cond_dir.glob(pat):
            if p.is_dir():
                out.add(p)
    return sorted(out)


def discover_repeats(base_folder: Path) -> List[RepeatData]:
    out: List[RepeatData] = []
    for condition in ("social", "asocial"):
        cond_dir = base_folder / condition
        if not cond_dir.is_dir():
            continue

        for hiring_dir in _iter_hiring_dirs(cond_dir):
            csv_path = hiring_dir / "playerRound_processed.csv"
            if not csv_path.is_file():
                continue

            log_path = hiring_dir / "playerRound.log"
            log_path_opt: Optional[Path] = log_path if log_path.is_file() else None

            choices = read_player_round_csv(csv_path, expected_agents=10, drop_last_row=True)

            rewards: Optional[List[List[float]]] = None
            if log_path_opt is not None:
                try:
                    rewards = read_player_round_log_rewards(log_path_opt, expected_agents=10)
                    print(csv_path,len(rewards))
                except Exception as e:
                    print(f"[WARN] failed to read rewards from {log_path_opt}: {e}")
                    rewards = None

            out.append(
                RepeatData(
                    condition=condition,
                    repeat_id=hiring_dir.name,
                    csv_path=csv_path,
                    log_path=log_path_opt,
                    choices_by_round=choices,
                    rewards_by_round=rewards,
                )
            )
    return out


# -----------------------------
# Metrics
# -----------------------------

def subset_cumulative_entropy(choices_by_round: List[List[str]], subset: Sequence[str]) -> np.ndarray:
    subset_set = set(s.strip() for s in subset if s.strip())
    counts: Dict[str, float] = {k: 0.0 for k in subset_set}

    ent: List[float] = []
    for rnd in choices_by_round:
        for c in rnd:
            key = c.strip()
            if key in counts:
                counts[key] += 1.0
        ent.append(shannon_entropy_base2(list(counts.values())))
    return np.asarray(ent, dtype=float)


def optimal_rate_cumulative(choices_by_round: List[List[str]], subset: Sequence[str]) -> np.ndarray:
    subset_set = set(s.strip() for s in subset if s.strip())
    rates: List[float] = []
    cum_hit = 0.0
    cum_total = 0.0

    for rnd in choices_by_round:
        n = len(rnd)
        if n <= 0:
            rates.append((cum_hit / cum_total) * 100.0 if cum_total > 0 else 0.0)
            continue

        hit = sum(1 for c in rnd if c.strip() in subset_set)
        cum_hit += float(hit)
        cum_total += float(n)
        rates.append((cum_hit / cum_total) * 100.0 if cum_total > 0 else 0.0)

    return np.asarray(rates, dtype=float)


def cumulative_reward_mean(rewards_by_round: List[List[float]]) -> np.ndarray:
    curve: List[float] = []
    cum_reward = 0.0
    cum_total = 0.0

    for rnd_rewards in rewards_by_round:
        n = len(rnd_rewards)
        if n <= 0:
            curve.append(cum_reward / cum_total if cum_total > 0 else 0.0)
            continue
        cum_reward += float(np.sum(rnd_rewards))
        cum_total += float(n)
        curve.append(cum_reward / cum_total if cum_total > 0 else 0.0)

    return np.asarray(curve, dtype=float)


# -----------------------------
# Plot + CSV writers
# -----------------------------

def write_trend_csv(
    out_path: Path,
    mean_social: np.ndarray, lo_social: np.ndarray, hi_social: np.ndarray,
    mean_asocial: np.ndarray, lo_asocial: np.ndarray, hi_asocial: np.ndarray,
    kind: str,
) -> None:
    _ensure_dir(out_path)
    L = min(len(mean_social), len(mean_asocial))
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Round",
            f"Mean_{kind}_Social", f"Low95_{kind}_Social", f"High95_{kind}_Social",
            f"Mean_{kind}_Asocial", f"Low95_{kind}_Asocial", f"High95_{kind}_Asocial",
            f"MeanDiff_{kind}_SocialMinusAsocial",
            f"MeanDiffPct_{kind}_vsAsocial",
        ])
        for r in range(L):
            ms = float(mean_social[r])
            ma = float(mean_asocial[r])
            diff = ms - ma
            diff_pct = _pct_change(ms, ma)
            w.writerow([
                r + 1,
                ms, float(lo_social[r]), float(hi_social[r]),
                ma, float(lo_asocial[r]), float(hi_asocial[r]),
                diff,
                diff_pct,
            ])


def write_last_round_summary_csv(
    out_path: Path,
    social_last: np.ndarray,
    asocial_last: np.ndarray,
    kind: str,
    b: float,
    ci_low: float,
    ci_high: float,
    p: float,
) -> None:
    _ensure_dir(out_path)
    ms, ls, hs = _mean_ci95_1d(social_last)
    ma, la, ha = _mean_ci95_1d(asocial_last)

    diff = ms - ma
    diff_pct = _pct_change(ms, ma)

    # OLS % (relative to Asocial mean)
    b_pct = _pct_of_base(b, ma)
    ci_low_pct = _pct_of_base(ci_low, ma)
    ci_high_pct = _pct_of_base(ci_high, ma)

    # Mean-based % diff CI (simple transform; relative to Asocial mean point-estimate)
    diff_pct_lo = _pct_change(ls, ma)
    diff_pct_hi = _pct_change(hs, ma)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Group", "Mean", "Low95", "High95", "N"])
        w.writerow([kind, "Social", ms, ls, hs, int(social_last.size)])
        w.writerow([kind, "Asocial", ma, la, ha, int(asocial_last.size)])
        w.writerow([])
        w.writerow(["MeanDiff(Social-Asocial)", diff])
        w.writerow(["MeanDiffPct(vs Asocial)", diff_pct])
        w.writerow(["MeanDiffPct_Low95(vs Asocial)", diff_pct_lo])
        w.writerow(["MeanDiffPct_High95(vs Asocial)", diff_pct_hi])
        w.writerow([])
        w.writerow(["OLS: y ~ 1 + SocialDummy (Asocial=0 reference)"])
        w.writerow(["b(Social-Asocial)", "CI_low", "CI_high", "p", "bPct(vs AsocialMean)", "CI_low_pct", "CI_high_pct"])
        w.writerow([b, ci_low, ci_high, p, b_pct, ci_low_pct, ci_high_pct])


def plot_trend(
    out_path: Path,
    mean_social: np.ndarray, lo_social: np.ndarray, hi_social: np.ndarray,
    mean_asocial: np.ndarray, lo_asocial: np.ndarray, hi_asocial: np.ndarray,
    ylabel: str,
    title: str,
    ylim: Tuple[float, float] | None,
) -> None:
    _ensure_dir(out_path)
    L = min(len(mean_social), len(mean_asocial))
    x = np.arange(1, L + 1)

    plt.figure(figsize=(4,3))
    ax = plt.gca()

    ax.plot(x, mean_social[:L], label="Social", linewidth=1.8)
    ax.fill_between(x, lo_social[:L], hi_social[:L], alpha=0.2)

    ax.plot(x, mean_asocial[:L], label="Asocial", linewidth=1.8)
    ax.fill_between(x, lo_asocial[:L], hi_asocial[:L], alpha=0.2)

    ax.set_xlabel("Rounds", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    # ax.set_title(title, fontsize=11)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="both", direction="out", length=3, width=0.8)
    ax.legend(frameon=False, fontsize=9, loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True)
    plt.close()


# def plot_last_round_points(
#     out_path: Path,
#     social_last: np.ndarray,
#     asocial_last: np.ndarray,
#     ylabel: str,
#     title: str,
#     ylim: Tuple[float, float] | None,
# ) -> None:
#     _ensure_dir(out_path)
#     rng = np.random.default_rng(7)

#     x_social = 0 + rng.normal(0, 0.05, size=social_last.size)
#     x_asocial = 1 + rng.normal(0, 0.05, size=asocial_last.size)

#     ms, ls, hs = _mean_ci95_1d(social_last)
#     ma, la, ha = _mean_ci95_1d(asocial_last)

#     plt.figure(figsize=(4,3))
#     ax = plt.gca()

#     ax.scatter(x_social, social_last, s=12, alpha=0.2)
#     ax.scatter(x_asocial, asocial_last, s=12, alpha=0.2)

#     ax.errorbar([0], [ms], yerr=[[ms - ls], [hs - ms]], fmt="o", capsize=4)
#     ax.errorbar([1], [ma], yerr=[[ma - la], [ha - ma]], fmt="o", capsize=4)

#     ax.set_xticks([0, 1])
#     ax.set_xticklabels(["Social", "Asocial"])
#     ax.set_ylabel(ylabel, fontsize=11)
#     # ax.set_title(title, fontsize=11)
#     if ylim is not None:
#         ax.set_ylim(*ylim)

#     ax.spines["right"].set_visible(False)
#     ax.spines["top"].set_visible(False)
#     ax.tick_params(axis="both", direction="out", length=3, width=0.8)

#     plt.tight_layout()
#     plt.savefig(out_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True)
#     plt.close()

def plot_last_round_points(
    out_path: Path,
    social_last: np.ndarray,
    asocial_last: np.ndarray,
    ylabel: str,
    title: str,
    ylim: Tuple[float, float] | None,
    *,
    dot_size: float = 25,
    mean_dot_size: float = 6,
    jitter: float = 0.08,
    dpi: int = 450,
    color_social: str | None = None,
    color_asocial: str | None = None,
    seed: int = 7,
) -> None:
    """
    Scatter (jittered) + mean ± 95% CI for last-round values (two groups).
    Saves a transparent PDF.

    Parameters
    ----------
    out_path:
        Output PDF path.
    social_last, asocial_last:
        1D arrays of values.
    ylabel, title:
        Axis label and optional title (title currently not drawn).
    ylim:
        If not None, sets y-limits (low, high). Otherwise auto.
    dot_size, mean_dot_size:
        Sizes for individual dots and mean marker (matplotlib points^2).
    jitter:
        Std dev for x jitter.
    dpi:
        Output DPI.
    color_social, color_asocial:
        Optional colors for the two groups. If None, matplotlib default is used.
    seed:
        RNG seed for deterministic jitter.
    """
    _ensure_dir(out_path)

    # optional, only if your project already has it
    try:
        _apply_science_style()  # type: ignore[name-defined]
    except Exception:
        pass

    social = np.asarray(social_last, dtype=float).ravel()
    asocial = np.asarray(asocial_last, dtype=float).ravel()

    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    def _scatter_with_ci(xi: float, vals: np.ndarray, color: str | None) -> None:
        if vals.size == 0:
            return

        xs = xi + rng.normal(0.0, jitter, size=vals.size)
        scatter_kwargs = dict(s=dot_size, alpha=0.50, edgecolors="none")
        if color is not None:
            scatter_kwargs["color"] = color
        ax.scatter(xs, vals, **scatter_kwargs)

        m, lo, hi = _mean_ci95_1d(vals)
        ax.errorbar(
            xi,
            m,
            yerr=[[m - lo], [hi - m]],
            fmt="o",
            color="black",
            ecolor="black",
            markersize=mean_dot_size,
            linewidth=1.2,
            capsize=0,
            zorder=3,
        )

    _scatter_with_ci(0.0, social, color_social)
    _scatter_with_ci(1.0, asocial, color_asocial)

    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Social", "Asocial"])
    ax.set_ylabel(ylabel)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        if social.size or asocial.size:
            y_min = np.inf
            y_max = -np.inf
            if social.size:
                y_min = min(y_min, float(np.min(social)))
                y_max = max(y_max, float(np.max(social)))
            if asocial.size:
                y_min = min(y_min, float(np.min(asocial)))
                y_max = max(y_max, float(np.max(asocial)))
            yr = max(1e-9, y_max - y_min)
            pad = 0.05 * yr
            ax.set_ylim(max(0.0, y_min - pad), y_max + pad)
        else:
            ax.set_ylim(0.0, 1.0)

    # Title omitted for publication style (uncomment to enable)
    # ax.set_title(title, fontsize=11)

    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.tick_params(axis="x", which="major", direction="out", length=7, width=1.0)
    ax.tick_params(axis="y", which="major", direction="out", length=7, width=1.0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_linewidth(1.1)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", transparent=True, format="pdf")
    plt.close(fig)
    print(f"Saved last-round: {out_path}")
    
# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Human CSV subset-only entropy + cumulative optimal-rate + cumulative reward (social vs asocial).")
    ap.add_argument("base_folder", type=str, help="Base folder containing social/ and asocial/ dirs.")
    ap.add_argument("--subset", type=str, default="", help="Comma-separated school names for subset.")
    ap.add_argument("--subset-file", type=str, default="", help="Text file (UTF-8), one school name per line.")
    args = ap.parse_args()

    base = Path(args.base_folder)
    if not base.is_dir():
        raise SystemExit(f"base_folder not found or not a directory: {base}")

    if args.subset_file:
        subset = _read_subset_from_file(Path(args.subset_file))
    else:
        subset = [s.strip() for s in args.subset.split(",") if s.strip()]

    if not subset:
        raise SystemExit("Provide subset via --subset 'A,B' or --subset-file subset.txt")

    repeats = discover_repeats(base)
    if not repeats:
        raise SystemExit("No repeats found. Check directory structure and filenames.")

    ent_social: List[np.ndarray] = []
    ent_asocial: List[np.ndarray] = []
    opt_social: List[np.ndarray] = []
    opt_asocial: List[np.ndarray] = []
    rew_social: List[np.ndarray] = []
    rew_asocial: List[np.ndarray] = []

    for rep in repeats:
        ent = subset_cumulative_entropy(rep.choices_by_round, subset=subset)
        opt = optimal_rate_cumulative(rep.choices_by_round, subset=subset)

        if rep.condition == "social":
            ent_social.append(ent)
            opt_social.append(opt)
        elif rep.condition == "asocial":
            ent_asocial.append(ent)
            opt_asocial.append(opt)

        if rep.rewards_by_round is not None:
            rew = cumulative_reward_mean(rep.rewards_by_round)
            if rep.condition == "social":
                rew_social.append(rew)
            elif rep.condition == "asocial":
                rew_asocial.append(rew)

    if not ent_social or not ent_asocial:
        raise SystemExit(f"Need both conditions. Found social={len(ent_social)}, asocial={len(ent_asocial)}")

    slug = _slugify_subset(subset)
    K = len(set(s.strip() for s in subset if s.strip()))

    # ----- Entropy -----
    m_es, l_es, u_es = mean_ci95_curves(ent_social)
    m_ea, l_ea, u_ea = mean_ci95_curves(ent_asocial)
    L_ent = min(len(m_es), len(m_ea))
    max_entropy = max(0.1, math.log2(max(1, K)))

    ent_trend_pdf = base / f"entropy_trend_subset_{slug}.pdf"
    ent_trend_csv = base / f"entropy_trend_subset_{slug}.csv"
    ent_last_pdf = base / f"entropy_last_round_subset_{slug}.pdf"
    ent_last_csv = base / f"entropy_last_round_subset_{slug}.csv"

    plot_trend(
        ent_trend_pdf,
        m_es[:L_ent], l_es[:L_ent], u_es[:L_ent],
        m_ea[:L_ent], l_ea[:L_ent], u_ea[:L_ent],
        ylabel="Entropy (Optimal)",
        title=f"Subset-only cumulative entropy (K={K})",
        ylim=(0.0, max_entropy),
    )
    write_trend_csv(
        ent_trend_csv,
        m_es[:L_ent], l_es[:L_ent], u_es[:L_ent],
        m_ea[:L_ent], l_ea[:L_ent], u_ea[:L_ent],
        kind="Entropy",
    )

    social_last_ent = np.asarray([c[L_ent - 1] for c in ent_social if len(c) >= L_ent], dtype=float)
    asocial_last_ent = np.asarray([c[L_ent - 1] for c in ent_asocial if len(c) >= L_ent], dtype=float)
    b_e, ci_lo_e, ci_hi_e, p_e = final_round_ols(social_last_ent, asocial_last_ent)

    plot_last_round_points(
        ent_last_pdf,
        social_last=social_last_ent,
        asocial_last=asocial_last_ent,
        ylabel="Entropy",
        title=f"Final round entropy (mean ± 95% CI), K={K}",
        ylim=(0.0, max_entropy),
    )
    write_last_round_summary_csv(
        ent_last_csv,
        social_last_ent,
        asocial_last_ent,
        kind="Entropy_FinalRound",
        b=b_e,
        ci_low=ci_lo_e,
        ci_high=ci_hi_e,
        p=p_e,
    )

    # ----- Cumulative optimal rate -----
    m_os, l_os, u_os = mean_ci95_curves(opt_social)
    m_oa, l_oa, u_oa = mean_ci95_curves(opt_asocial)
    L_opt = min(len(m_os), len(m_oa))

    opt_trend_pdf = base / f"optimal_rate_cum_trend_subset_{slug}.pdf"
    opt_trend_csv = base / f"optimal_rate_cum_trend_subset_{slug}.csv"
    opt_last_pdf = base / f"optimal_rate_cum_last_round_subset_{slug}.pdf"
    opt_last_csv = base / f"optimal_rate_cum_last_round_subset_{slug}.csv"

    plot_trend(
        opt_trend_pdf,
        m_os[:L_opt], l_os[:L_opt], u_os[:L_opt],
        m_oa[:L_opt], l_oa[:L_opt], u_oa[:L_opt],
        ylabel="Cumulative optimal arm rate (%)",
        title=f"Cumulative optimal arm rate (subset cumulative hit rate), K={K}",
        ylim=(0.0, 100.0),
    )
    write_trend_csv(
        opt_trend_csv,
        m_os[:L_opt], l_os[:L_opt], u_os[:L_opt],
        m_oa[:L_opt], l_oa[:L_opt], u_oa[:L_opt],
        kind="OptimalRateCumPct",
    )

    social_last_opt = np.asarray([c[L_opt - 1] for c in opt_social if len(c) >= L_opt], dtype=float)
    asocial_last_opt = np.asarray([c[L_opt - 1] for c in opt_asocial if len(c) >= L_opt], dtype=float)
    b_o, ci_lo_o, ci_hi_o, p_o = final_round_ols(social_last_opt, asocial_last_opt)

    plot_last_round_points(
        opt_last_pdf,
        social_last=social_last_opt,
        asocial_last=asocial_last_opt,
        ylabel="Cumulative optimal arm rate (%)",
        title=f"Final round cumulative optimal rate (mean ± 95% CI), K={K}",
        ylim=(0.0, 100.0),
    )
    write_last_round_summary_csv(
        opt_last_csv,
        social_last_opt,
        asocial_last_opt,
        kind="OptimalRateCum_FinalRoundPct",
        b=b_o,
        ci_low=ci_lo_o,
        ci_high=ci_hi_o,
        p=p_o,
    )

    # ----- Cumulative reward -----
    have_reward = bool(rew_social) and bool(rew_asocial)
    b_r = ci_lo_r = ci_hi_r = p_r = float("nan")

    if not have_reward:
        print(f"\n[WARN] reward curves missing or incomplete: social_logs={len(rew_social)} | asocial_logs={len(rew_asocial)}")
    else:
        m_rs, l_rs, u_rs = mean_ci95_curves(rew_social)
        m_ra, l_ra, u_ra = mean_ci95_curves(rew_asocial)
        L_rew = min(len(m_rs), len(m_ra))

        rew_trend_pdf = base / f"reward_cum_mean_trend_{slug}.pdf"
        rew_trend_csv = base / f"reward_cum_mean_trend_{slug}.csv"
        rew_last_pdf = base / f"reward_cum_mean_last_round_{slug}.pdf"
        rew_last_csv = base / f"reward_cum_mean_last_round_{slug}.csv"

        plot_trend(
            rew_trend_pdf,
            m_rs[:L_rew], l_rs[:L_rew], u_rs[:L_rew],
            m_ra[:L_rew], l_ra[:L_rew], u_ra[:L_rew],
            ylabel="Cumulative reward (mean)",
            title="Cumulative reward (running mean per choice)",
            ylim=(0.0, 1.0),
        )
        write_trend_csv(
            rew_trend_csv,
            m_rs[:L_rew], l_rs[:L_rew], u_rs[:L_rew],
            m_ra[:L_rew], l_ra[:L_rew], u_ra[:L_rew],
            kind="RewardCumMean",
        )

        social_last_rew = np.asarray([c[L_rew - 1] for c in rew_social if len(c) >= L_rew], dtype=float)
        asocial_last_rew = np.asarray([c[L_rew - 1] for c in rew_asocial if len(c) >= L_rew], dtype=float)
        b_r, ci_lo_r, ci_hi_r, p_r = final_round_ols(social_last_rew, asocial_last_rew)

        plot_last_round_points(
            rew_last_pdf,
            social_last=social_last_rew,
            asocial_last=asocial_last_rew,
            ylabel="Cumulative reward (mean per choice)",
            title="Final round cumulative reward (mean ± 95% CI)",
            ylim=(0.0, 1.0),
        )
        write_last_round_summary_csv(
            rew_last_csv,
            social_last_rew,
            asocial_last_rew,
            kind="RewardCumMean_FinalRound",
            b=b_r,
            ci_low=ci_lo_r,
            ci_high=ci_hi_r,
            p=p_r,
        )

        print(f"[OUT] {rew_trend_pdf}")
        print(f"[OUT] {rew_trend_csv}")
        print(f"[OUT] {rew_last_pdf}")
        print(f"[OUT] {rew_last_csv}")

    # ----- Print requested stats (+ % change + OLS % CI) -----
    ms_e, _, _ = _mean_ci95_1d(social_last_ent)
    ma_e, _, _ = _mean_ci95_1d(asocial_last_ent)
    diff_e = ms_e - ma_e
    diffpct_e = _pct_change(ms_e, ma_e)
    b_pct_e = _pct_of_base(b_e, ma_e)
    ci_lo_pct_e = _pct_of_base(ci_lo_e, ma_e)
    ci_hi_pct_e = _pct_of_base(ci_hi_e, ma_e)

    ms_o, _, _ = _mean_ci95_1d(social_last_opt)
    ma_o, _, _ = _mean_ci95_1d(asocial_last_opt)
    diff_o = ms_o - ma_o
    diffpct_o = _pct_change(ms_o, ma_o)
    b_pct_o = _pct_of_base(b_o, ma_o)
    ci_lo_pct_o = _pct_of_base(ci_lo_o, ma_o)
    ci_hi_pct_o = _pct_of_base(ci_hi_o, ma_o)

    print("\n=== Final-round OLS (Asocial reference): y ~ 1 + SocialDummy ===")
    print(
        f"[Entropy]        b={b_e:.4f}, 95% CI=[{ci_lo_e:.4f}, {ci_hi_e:.4f}], p={p_e:.6g} | "
        f"b%={b_pct_e:.3f}% CI%=[{ci_lo_pct_e:.3f}%, {ci_hi_pct_e:.3f}%] | "
        f"Δ={diff_e:.4f} | %Δ={diffpct_e:.3f}%"
    )
    print(
        f"[OptRateCum]     b={b_o:.4f}, 95% CI=[{ci_lo_o:.4f}, {ci_hi_o:.4f}], p={p_o:.6g} | "
        f"b%={b_pct_o:.3f}% CI%=[{ci_lo_pct_o:.3f}%, {ci_hi_pct_o:.3f}%] | "
        f"Δ={diff_o:.4f} | %Δ={diffpct_o:.3f}%"
    )

    if have_reward:
        ms_r, _, _ = _mean_ci95_1d(social_last_rew)
        ma_r, _, _ = _mean_ci95_1d(asocial_last_rew)
        diff_r = ms_r - ma_r
        diffpct_r = _pct_change(ms_r, ma_r)
        b_pct_r = _pct_of_base(b_r, ma_r)
        ci_lo_pct_r = _pct_of_base(ci_lo_r, ma_r)
        ci_hi_pct_r = _pct_of_base(ci_hi_r, ma_r)

        print(
            f"[RewardCumMean]  b={b_r:.4f}, 95% CI=[{ci_lo_r:.4f}, {ci_hi_r:.4f}], p={p_r:.6g} | "
            f"b%={b_pct_r:.3f}% CI%=[{ci_lo_pct_r:.3f}%, {ci_hi_pct_r:.3f}%] | "
            f"Δ={diff_r:.4f} | %Δ={diffpct_r:.3f}%"
        )
    else:
        print("[RewardCumMean]  skipped (missing playerRound.log for one/both conditions)")

    print(f"\n[OK] repeats: social={len(ent_social)} | asocial={len(ent_asocial)}")
    print(f"[INFO] per-csv: last data row ignored")

    print(f"[OUT] {ent_trend_pdf}")
    print(f"[OUT] {ent_trend_csv}")
    print(f"[OUT] {ent_last_pdf}")
    print(f"[OUT] {ent_last_csv}")

    print(f"[OUT] {opt_trend_pdf}")
    print(f"[OUT] {opt_trend_csv}")
    print(f"[OUT] {opt_last_pdf}")
    print(f"[OUT] {opt_last_csv}")


if __name__ == "__main__":
    main()