"""Print power-analysis and seasonal-consistency sections from a scored run.

These metrics are written automatically to summary.json by `python -m src.score`.
This script is a convenience for terminal review before posting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.score import score_run  # noqa: E402


def _load_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        score_run(run_dir)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _print_power(summary: dict) -> None:
    block = summary.get("power_analysis")
    if not block:
        print("No power_analysis block (need >= 2 models in the run).")
        return
    print(block["method"])
    print()
    for pair in block["pairs"]:
        print(
            f"{pair['model_a']} vs {pair['model_b']}  "
            f"(n={pair['n_scenarios']}, delta={pair['delta_crps_relative']:+.4f}, "
            f"t={pair['t_paired']:+.2f}, "
            f"need ~{pair['scenarios_needed_80pct_power']:.0f} scenarios for 80% power, "
            f"MDE@n={pair['min_detectable_delta_at_n']:.4f})"
        )


def _print_consistency(summary: dict) -> None:
    block = summary.get("seasonal_consistency")
    if not block:
        print("No seasonal_consistency block.")
        return
    print(block["method"])
    print()
    print("Target gaps (FAWN quantiles):")
    for loc, gap in sorted(block["target_gaps"].items()):
        print(f"  {loc}: {gap:+.1%}")
    print()
    for model, stats in sorted(block["per_model"].items()):
        print(
            f"  {model}: mean_gap={stats['mean_gap']:+.1%}, "
            f"spread={stats['gap_spread']:.1%} "
            f"({stats['min_gap']:+.1%} to {stats['max_gap']:+.1%}, "
            f"{stats['n_cells']} cells)"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect power and consistency metrics for a run")
    parser.add_argument(
        "runs",
        nargs="*",
        default=["results/demo"],
        help="Run directories (default: results/demo)",
    )
    args = parser.parse_args(argv)

    for run in args.runs:
        run_dir = Path(run)
        if not (run_dir / "predictions.jsonl").exists():
            print(f"Skip {run_dir}: no predictions.jsonl", file=sys.stderr)
            continue
        print(f"=== {run_dir} ===")
        summary = _load_summary(run_dir)
        print("\n--- power_analysis ---")
        _print_power(summary)
        print("\n--- seasonal_consistency ---")
        _print_consistency(summary)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
