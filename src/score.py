"""Score predictions: CRPS, pinball loss, interval width, confidence calibration."""

from __future__ import annotations

import argparse
import json
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
    return max(5.0, (q90 - q10) * 0.35)


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
        rows.append(
            {
                "scenario_id": rec.scenario_id,
                "model": rec.model,
                "stratum": sc.stratum,
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
                "target_interval_width": tgt_p90 - tgt_p10,
                "confidence": float(pred.confidence),
                "crps": crps_reference(p10, p50, p90, ref_years),
                "mean_pinball_loss": float(
                    np.mean([mean_pinball(y, p10, p50, p90) for y in ref_years])
                ),
                "median_abs_error": float(np.mean(median_errors)),
                "latency_ms": rec.latency_ms,
            }
        )
    return pd.DataFrame(rows)


def compute_metrics(df: pd.DataFrame) -> dict:
    conf = df["confidence"].to_numpy(dtype=float)
    well_calibrated = (df["median_abs_error"] < 5.0).astype(float).to_numpy()

    return {
        "n": int(len(df)),
        "crps": float(df["crps"].mean()),
        "mean_pinball_loss": float(df["mean_pinball_loss"].mean()),
        "mean_interval_width": float(df["interval_width"].mean()),
        "mean_median_abs_error": float(df["median_abs_error"].mean()),
        "ece_confidence": expected_calibration_error(well_calibrated, conf),
    }


def plot_interval_width_by_stratum(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    grouped = df.groupby(["model", "stratum"], as_index=False)["interval_width"].mean()
    models = sorted(grouped["model"].unique())
    strata = ["specific_station", "regional_inference", "underspecified"]
    x = np.arange(len(strata))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, model in enumerate(models):
        sub = grouped[grouped["model"] == model].set_index("stratum")
        ys = [sub.loc[s, "interval_width"] if s in sub.index else np.nan for s in strata]
        ax.bar(x + i * width, ys, width=width, label=model)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(strata, rotation=15)
    ax.set_ylabel("Mean interval width (p90 - p10, inches)")
    ax.set_title("Interval width by specificity stratum")
    ax.legend(fontsize=8)
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
        "by_prompt_variant": {},
    }

    for model, sub in df.groupby("model"):
        summary["by_model"][model] = compute_metrics(sub)
    for stratum, sub in df.groupby("stratum"):
        summary["by_stratum"][stratum] = compute_metrics(sub)
    for variant, sub in df.groupby("prompt_variant"):
        summary["by_prompt_variant"][variant] = compute_metrics(sub)

    if sync is not None:
        summary["fawn_sync"] = {
            "generated_at": sync.generated_at,
            "stations_included": sync.stations_included,
            "scenario_count": len(sync.scenarios),
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    df.to_csv(run_dir / "by_scenario.csv", index=False)

    by_stratum = (
        df.groupby(["model", "stratum"], as_index=False)
        .agg(
            crps=("crps", "mean"),
            mean_pinball_loss=("mean_pinball_loss", "mean"),
            interval_width=("interval_width", "mean"),
            n=("scenario_id", "count"),
        )
    )
    by_stratum.to_csv(run_dir / "by_stratum.csv", index=False)

    by_variant = (
        df.groupby(["model", "prompt_variant"], as_index=False)
        .agg(
            crps=("crps", "mean"),
            mean_pinball_loss=("mean_pinball_loss", "mean"),
            interval_width=("interval_width", "mean"),
            n=("scenario_id", "count"),
        )
    )
    by_variant.to_csv(run_dir / "by_prompt_variant.csv", index=False)

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
