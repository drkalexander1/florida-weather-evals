"""Pretty-print key sections from a run's summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, default=Path("results/demo"), nargs="?")
    parser.add_argument(
        "--section",
        choices=["overall", "by_model", "power", "consistency", "all"],
        default="all",
    )
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))

    if args.section in ("overall", "all"):
        print("=== overall ===")
        print(json.dumps(summary["overall"], indent=2))
    if args.section in ("by_model", "all"):
        print("\n=== by_model (crps_relative) ===")
        for model, m in sorted(summary.get("by_model", {}).items()):
            print(f"  {model:24s} {m['crps_relative']:.4f}")
    if args.section in ("power", "all") and "power_analysis" in summary:
        print("\n=== power_analysis ===")
        for p in summary["power_analysis"]["pairs"]:
            print(
                f"  {p['model_a']:20s} vs {p['model_b']:20s}  "
                f"t={p['t_paired']:+.2f}  delta={p['delta_crps_relative']:+.4f}"
            )
    if args.section in ("consistency", "all") and "seasonal_consistency" in summary:
        print("\n=== seasonal_consistency ===")
        sc = summary["seasonal_consistency"]
        print("  targets:", {k: f"{v:+.1%}" for k, v in sc["target_gaps"].items()})
        for model, s in sorted(sc["per_model"].items()):
            print(f"  {model:24s} mean={s['mean_gap']:+.1%}  spread={s['gap_spread']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
