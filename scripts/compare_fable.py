"""Compare anthropic-seasonal results with vs without claude-fable-5."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.score import (
    build_frame,
    compute_metrics,
    load_predictions,
    load_scenarios,
    pairwise_power_analysis,
    seasonal_consistency,
)

RUN = Path("results/anthropic-seasonal")


def cross_location_recycling(df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for model, sub in df.groupby("model"):
        dup_cells = 0
        for season in ("annual", "wet_season", "dry_season"):
            sid_season = season.replace("_season", "")
            for pv in ("natural", "statistical"):
                triples = []
                for pref in ("miami", "gulf_city", "random_fl"):
                    row = sub[sub.scenario_id == f"{pref}_{sid_season}_{pv}"]
                    if len(row) == 1:
                        r = row.iloc[0]
                        triples.append((r.p10, r.p50, r.p90))
                if len(triples) == 3 and len(set(triples)) < 3:
                    dup_cells += 1
        counts[model] = dup_cells
    return counts


def summarize(recs, label: str) -> pd.DataFrame:
    df = build_frame(load_scenarios(), recs)
    print(f"\n===== {label} ({len(recs)} preds, {df.model.nunique()} models) =====")
    print("crps_relative by model:")
    for model, sub in df.groupby("model"):
        ece = compute_metrics(sub)["ece_confidence"]
        print(f"  {model:22s} {sub.crps_relative.mean():.4f}  ece {ece:.3f}")

    print("by stratum interval width (mean in):")
    for stratum in ("specific_station", "regional_inference", "underspecified"):
        print(f"  {stratum:20s} {df[df.stratum == stratum].interval_width.mean():.1f}")

    print("by season crps_relative:")
    for season in ("annual", "wet_season", "dry_season"):
        print(f"  {season:20s} {df[df.season == season].crps_relative.mean():.4f}")

    print("power pairs (t):")
    for p in pairwise_power_analysis(df)["pairs"]:
        print(
            f"  {p['model_a']} vs {p['model_b']}: "
            f"t={p['t_paired']:+.2f} delta={p['delta_crps_relative']:+.4f}"
        )

    print("seasonal consistency mean_gap / spread:")
    sc = seasonal_consistency(df)
    for model, v in sorted(sc["per_model"].items()):
        print(f"  {model:22s} gap={v['mean_gap']:+.3f} spread={v['gap_spread']:.3f}")

    print("cross-location duplicate triples /6:")
    for model, n in sorted(cross_location_recycling(df).items()):
        print(f"  {model}: {n}/6")
    return df


def main() -> int:
    scenarios = load_scenarios()
    all_recs = load_predictions(RUN / "predictions.jsonl")
    no_fable = [r for r in all_recs if r.model != "claude-fable-5"]

    df_all = summarize(all_recs, "WITH Fable")
    df_no = summarize(no_fable, "WITHOUT Fable (tool-use trio only)")

    fo = build_frame(scenarios, load_predictions(Path("results/openai-seasonal/predictions.jsonl")))
    combined = pd.concat([df_no, fo])
    print("\n===== Full ranking WITHOUT Fable =====")
    for model, sub in combined.groupby("model"):
        print(f"  {model:22s} {sub.crps_relative.mean():.4f}")

    # Fable-only deltas vs opus/sonnet
    fable = df_all[df_all.model == "claude-fable-5"]
    print("\n===== Fable vs tool-use trio (mean crps_relative gap) =====")
    for other in ("claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"):
        o = df_all[df_all.model == other]
        merged = fable.merge(o, on="scenario_id", suffixes=("_f", "_o"))
        delta = (merged.crps_relative_f - merged.crps_relative_o).mean()
        print(f"  fable - {other}: {delta:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
