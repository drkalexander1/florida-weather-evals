"""Score predictions: CRPS, pinball loss, interval width, confidence calibration."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.schema import (
    FAWN_SYNC_PATH,
    QUANTILE_LEVELS,
    ROOT,
    PredictionRecord,
    Scenario,
    load_fawn_sync,
    load_scenarios,
    scoring_reference_years,
    scoring_targets,
)

RESULTS_DIR = ROOT / "results" / "latest"


def pinball_loss(y: float, q: float, tau: float) -> float:
    err = y - q
    return float(tau * err if err >= 0 else (tau - 1) * err)


def _extrapolate_margin(q10: float, q50: float, q90: float) -> float:
    # Scale-aware: dry-season totals (~6 in) need a much smaller tail than annual (~55 in).
    return max((q90 - q10) * 0.35, q50 * 0.1, 0.5)


def forecast_cdf(x: float, q10: float, q50: float, q90: float) -> float:
    """Piecewise-linear CDF through (q10, 0.1), (q50, 0.5), (q90, 0.9) with linear tails."""
    margin = _extrapolate_margin(q10, q50, q90)
    left = q10 - margin
    right = q90 + margin

    if x <= left:
        return 0.0
    if x >= right:
        return 1.0
    if x <= q10:
        return 0.1 * (x - left) / (q10 - left)
    if x <= q50:
        return 0.1 + 0.4 * (x - q10) / (q50 - q10)
    if x <= q90:
        return 0.5 + 0.4 * (x - q50) / (q90 - q50)
    return 0.9 + 0.1 * (x - q90) / (right - q90)


def crps_observation(y: float, q10: float, q50: float, q90: float, *, n_grid: int = 400) -> float:
    margin = _extrapolate_margin(q10, q50, q90)
    lo = min(y, q10) - margin
    hi = max(y, q90) + margin
    xs = np.linspace(lo, hi, n_grid)
    fs = np.array([forecast_cdf(x, q10, q50, q90) for x in xs])
    hs = (xs >= y).astype(float)
    return float(np.trapezoid((fs - hs) ** 2, xs))


def mean_pinball(y: float, q10: float, q50: float, q90: float) -> float:
    losses = [
        pinball_loss(y, q10, 0.1),
        pinball_loss(y, q50, 0.5),
        pinball_loss(y, q90, 0.9),
    ]
    return float(np.mean(losses))


def crps_reference(q10: float, q50: float, q90: float, reference_years: list[float]) -> float:
    if not reference_years:
        return float("nan")
    return float(np.mean([crps_observation(y, q10, q50, q90) for y in reference_years]))


def expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_score >= bins[i]) & (
            y_score < bins[i + 1] if i < n_bins - 1 else y_score <= bins[i + 1]
        )
        if not mask.any():
            continue
        acc = y_true[mask].mean()
        conf = y_score[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)


def load_predictions(path: Path) -> list[PredictionRecord]:
    records: list[PredictionRecord] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(PredictionRecord.model_validate_json(line))
    return records


def build_frame(
    scenarios: list[Scenario],
    predictions: list[PredictionRecord],
    *,
    sync_path: Path | None = None,
) -> pd.DataFrame:
    scenario_map = {s.id: s for s in scenarios}
    sync = load_fawn_sync(sync_path)
    rows = []
    for rec in predictions:
        sc = scenario_map.get(rec.scenario_id)
        if not sc:
            continue
        pred = rec.prediction
        ref_years = scoring_reference_years(sc, sync)
        tgt_p10, tgt_p50, tgt_p90 = scoring_targets(sc, sync)
        p10, p50, p90 = float(pred.p10), float(pred.p50), float(pred.p90)
        median_errors = [abs(p50 - y) for y in ref_years]
        crps = crps_reference(p10, p50, p90, ref_years)
        scale = max(tgt_p50, 1.0)
        rows.append(
            {
                "scenario_id": rec.scenario_id,
                "model": rec.model,
                "stratum": sc.stratum,
                "season": sc.season,
                "prompt_variant": sc.prompt_variant,
                "location_description": sc.location_description,
                "fawn_station_id": sc.fawn_station_id,
                "target_p10": tgt_p10,
                "target_p50": tgt_p50,
                "target_p90": tgt_p90,
                "reference_year_count": len(ref_years),
                "p10": p10,
                "p50": p50,
                "p90": p90,
                "interval_width": p90 - p10,
                "relative_interval_width": (p90 - p10) / scale,
                "target_interval_width": tgt_p90 - tgt_p10,
                "confidence": float(pred.confidence),
                "crps": crps,
                # CRPS scales with the variable's magnitude; normalize by the target
                # median so dry-season and annual scenarios are comparable.
                "crps_relative": crps / scale,
                "mean_pinball_loss": float(
                    np.mean([mean_pinball(y, p10, p50, p90) for y in ref_years])
                ),
                "median_abs_error": float(np.mean(median_errors)),
                "median_abs_error_relative": float(np.mean(median_errors)) / scale,
                "latency_ms": rec.latency_ms,
            }
        )
    return pd.DataFrame(rows)


def compute_metrics(df: pd.DataFrame) -> dict:
    conf = df["confidence"].to_numpy(dtype=float)
    # Relative threshold so a 1-inch dry-season miss and a 9-inch annual miss
    # count the same; floor avoids hypersensitivity on tiny medians.
    well_calibrated = (df["median_abs_error_relative"] < 0.15).astype(float).to_numpy()

    return {
        "n": int(len(df)),
        "crps": float(df["crps"].mean()),
        "crps_relative": float(df["crps_relative"].mean()),
        "mean_pinball_loss": float(df["mean_pinball_loss"].mean()),
        "mean_interval_width": float(df["interval_width"].mean()),
        "mean_relative_interval_width": float(df["relative_interval_width"].mean()),
        "mean_median_abs_error": float(df["median_abs_error"].mean()),
        "ece_confidence": expected_calibration_error(well_calibrated, conf),
    }


# Paired power analysis constants: alpha = 0.05 two-sided, 80% power.
_Z_ALPHA = 1.96
_Z_BETA = 0.84


def pairwise_power_analysis(df: pd.DataFrame) -> dict:
    """Paired comparison of relative CRPS for every model pair in the run.

    Because all models answer the same scenarios, per-scenario differencing
    cancels shared scenario difficulty. Reports the observed gap, the paired
    t statistic at the current n, the scenario count needed for 80% power at
    the observed gap, and the minimal detectable gap at the current n.
    """
    pairs = []
    for a, b in combinations(sorted(df["model"].unique()), 2):
        da = df[df["model"] == a].set_index("scenario_id")["crps_relative"]
        db = df[df["model"] == b].set_index("scenario_id")["crps_relative"]
        diffs = (da - db).dropna()
        n = len(diffs)
        if n < 2:
            continue
        delta = float(diffs.mean())
        sd = float(diffs.std(ddof=1))
        se = sd / np.sqrt(n)
        pairs.append(
            {
                "model_a": a,
                "model_b": b,
                "n_scenarios": n,
                "delta_crps_relative": delta,
                "sd_of_paired_diffs": sd,
                "t_paired": delta / se if se > 0 else float("inf"),
                "scenarios_needed_80pct_power": (
                    float((_Z_ALPHA + _Z_BETA) ** 2 * (sd / delta) ** 2)
                    if delta != 0
                    else None
                ),
                "min_detectable_delta_at_n": float((_Z_ALPHA + _Z_BETA) * se),
            }
        )
    return {
        "method": (
            "Paired t on per-scenario crps_relative differences; "
            "alpha=0.05 two-sided, power=0.80. Treats scenarios as exchangeable, "
            "which overstates power given shared locations; with multiple pairs, "
            "apply a multiple-comparison correction before claiming significance."
        ),
        "pairs": pairs,
    }


def seasonal_consistency(df: pd.DataFrame) -> dict:
    """Self-consistency check: annual p50 vs wet + dry season p50s.

    Needs no ground truth. May falls in neither season window and quantiles
    are not additive, so the calibrated reference is the targets' own gap
    (~+30%), not zero. Models whose gap deviates far from the target gap, or
    varies wildly across cells, are internally incoherent across timescales.
    """
    sub = df[["scenario_id", "model", "season", "p50", "target_p50"]].copy()
    parts = sub["scenario_id"].str.rsplit("_", n=2)
    sub["location"] = parts.str[0]
    sub["variant"] = parts.str[2]
    required = {"annual", "wet_season", "dry_season"}

    per_model: dict = {}
    for model, g in sub.groupby("model"):
        gaps = []
        for (_loc, _var), cell in g.groupby(["location", "variant"]):
            vals = dict(zip(cell["season"], cell["p50"]))
            if required <= set(vals) and vals["annual"] > 0:
                gaps.append(
                    (vals["annual"] - vals["wet_season"] - vals["dry_season"]) / vals["annual"]
                )
        if gaps:
            per_model[model] = {
                "mean_gap": float(np.mean(gaps)),
                "min_gap": float(np.min(gaps)),
                "max_gap": float(np.max(gaps)),
                "gap_spread": float(np.max(gaps) - np.min(gaps)),
                "n_cells": len(gaps),
            }
    if not per_model:
        return {}

    target_gaps: dict = {}
    targets = sub.drop_duplicates(["location", "season"])
    for loc, g in targets.groupby("location"):
        vals = dict(zip(g["season"], g["target_p50"]))
        if required <= set(vals) and vals["annual"] > 0:
            target_gaps[loc] = float(
                (vals["annual"] - vals["wet_season"] - vals["dry_season"]) / vals["annual"]
            )

    return {
        "method": (
            "gap = (p50_annual - p50_wet - p50_dry) / p50_annual per location x "
            "prompt-variant cell. Compare each model's mean_gap to target_gaps "
            "(the calibrated reference; nonzero because May is in neither season "
            "and quantiles are not additive). Large gap_spread means the model's "
            "annual and seasonal answers are internally inconsistent."
        ),
        "per_model": per_model,
        "target_gaps": target_gaps,
    }


def plot_interval_width_by_stratum(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    strata = ["specific_station", "regional_inference", "underspecified"]
    seasons = [s for s in ["annual", "wet_season", "dry_season"] if s in set(df["season"])]
    models = sorted(df["model"].unique())
    bar_w = 0.8 / max(len(models), 1)
    x = np.arange(len(strata))

    fig, axes = plt.subplots(1, max(len(seasons), 1), figsize=(5 * max(len(seasons), 1), 4), squeeze=False)
    for ax, season in zip(axes[0], seasons):
        grouped = (
            df[df["season"] == season]
            .groupby(["model", "stratum"], as_index=False)["interval_width"]
            .mean()
        )
        for i, model in enumerate(models):
            sub = grouped[grouped["model"] == model].set_index("stratum")
            ys = [sub.loc[s, "interval_width"] if s in sub.index else np.nan for s in strata]
            ax.bar(x + i * bar_w, ys, width=bar_w, label=model)
        ax.set_xticks(x + bar_w * (len(models) - 1) / 2)
        ax.set_xticklabels(["specific", "regional", "underspec."], rotation=15)
        ax.set_title(season)
        ax.set_ylabel("Mean interval width (in)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Interval width by specificity stratum and season")
    fig.tight_layout()
    fig.savefig(out_dir / "interval_width_by_stratum.png", dpi=120)
    plt.close(fig)


def score_run(
    run_dir: Path,
    scenarios_path: Path | None = None,
    sync_path: Path | None = None,
) -> dict:
    scenarios = load_scenarios(scenarios_path)
    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.exists():
        raise FileNotFoundError(f"No predictions at {predictions_path}")

    predictions = load_predictions(predictions_path)
    df = build_frame(scenarios, predictions, sync_path=sync_path)
    if df.empty:
        raise ValueError("No matching scenarios for predictions")

    sync = load_fawn_sync(sync_path)
    summary: dict = {
        "target_source": "fawn_sync" if sync is not None else "curator_target_quantiles",
        "overall": compute_metrics(df),
        "by_model": {},
        "by_stratum": {},
        "by_season": {},
        "by_prompt_variant": {},
    }

    for model, sub in df.groupby("model"):
        summary["by_model"][model] = compute_metrics(sub)
    for stratum, sub in df.groupby("stratum"):
        summary["by_stratum"][stratum] = compute_metrics(sub)
    for season, sub in df.groupby("season"):
        summary["by_season"][season] = compute_metrics(sub)
    for variant, sub in df.groupby("prompt_variant"):
        summary["by_prompt_variant"][variant] = compute_metrics(sub)

    if df["model"].nunique() >= 2:
        summary["power_analysis"] = pairwise_power_analysis(df)
    consistency = seasonal_consistency(df)
    if consistency:
        summary["seasonal_consistency"] = consistency

    if sync is not None:
        summary["fawn_sync"] = {
            "generated_at": sync.generated_at,
            "stations_included": sync.stations_included,
            "scenario_count": len(sync.scenarios),
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    df.to_csv(run_dir / "by_scenario.csv", index=False)

    agg_spec = dict(
        crps=("crps", "mean"),
        crps_relative=("crps_relative", "mean"),
        mean_pinball_loss=("mean_pinball_loss", "mean"),
        interval_width=("interval_width", "mean"),
        relative_interval_width=("relative_interval_width", "mean"),
        n=("scenario_id", "count"),
    )

    df.groupby(["model", "stratum", "season"], as_index=False).agg(**agg_spec).to_csv(
        run_dir / "by_stratum.csv", index=False
    )
    df.groupby(["model", "season"], as_index=False).agg(**agg_spec).to_csv(
        run_dir / "by_season.csv", index=False
    )
    df.groupby(["model", "prompt_variant"], as_index=False).agg(**agg_spec).to_csv(
        run_dir / "by_prompt_variant.csv", index=False
    )

    plot_interval_width_by_stratum(df, run_dir)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score Florida weather eval predictions")
    parser.add_argument("--run", type=Path, default=RESULTS_DIR)
    parser.add_argument("--scenarios", type=Path, default=ROOT / "data" / "scenarios.yaml")
    parser.add_argument("--sync", type=Path, default=FAWN_SYNC_PATH)
    args = parser.parse_args(argv)

    summary = score_run(args.run, scenarios_path=args.scenarios, sync_path=args.sync)
    print(json.dumps(summary["overall"], indent=2))
    print(f"Wrote summary to {args.run / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
