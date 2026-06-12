"""Export the scored per-prediction frame for one or more runs as JSON (ad-hoc analysis helper)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.score import build_frame, load_predictions, load_scenarios

COLUMNS = [
    "scenario_id",
    "model",
    "stratum",
    "season",
    "prompt_variant",
    "p10",
    "p50",
    "p90",
    "confidence",
    "target_p10",
    "target_p50",
    "target_p90",
    "crps",
    "crps_relative",
    "interval_width",
    "relative_interval_width",
    "median_abs_error_relative",
]


def main() -> int:
    out = {}
    for run in sys.argv[1:]:
        run_dir = Path("results") / run
        frame = build_frame(load_scenarios(), load_predictions(run_dir / "predictions.jsonl"))
        out[run] = frame[COLUMNS].to_dict(orient="records")
    dest = Path("results") / "combined_frame.json"
    dest.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
