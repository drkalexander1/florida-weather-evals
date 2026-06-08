"""Generate synthetic predictions for results/demo (no API keys)."""

from __future__ import annotations

import json
from pathlib import Path

from src.schema import Prediction, PredictionRecord, Scenario, load_scenarios

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "demo"

# Models that fail calibration: same width everywhere
BAD_WIDTH = 20.0


def synthetic_quantiles(sc: Scenario, model: str) -> tuple[float, float, float]:
    p50 = float(sc.target_p50)
    if "mini" in model:
        # Overconfident: narrow intervals regardless of stratum
        half = BAD_WIDTH / 2
        return p50 - half, p50, p50 + half
    # Better calibrated: match target spread with slight bias
    spread = float(sc.target_p90 - sc.target_p10)
    return p50 - spread / 2, p50, p50 + spread / 2


def main() -> None:
    scenarios = load_scenarios()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "predictions.jsonl"
    models = ["gpt-4o-mini-demo", "gpt-4o-demo"]

    with path.open("w", encoding="utf-8") as f:
        for model in models:
            for i, sc in enumerate(scenarios):
                p10, p50, p90 = synthetic_quantiles(sc, model)
                pred = Prediction(
                    p10=p10,
                    p50=p50,
                    p90=p90,
                    confidence=0.85 if "mini" in model else 0.75,
                    reasoning="Synthetic demo prediction for pipeline test.",
                )
                rec = PredictionRecord(
                    scenario_id=sc.id,
                    model=model,
                    provider="DemoProvider",
                    prediction=pred,
                    latency_ms=100.0 + i,
                )
                f.write(rec.model_dump_json() + "\n")

    manifest = {
        "created_at": "demo",
        "models": models,
        "scenario_count": len(scenarios),
        "predictions_file": "predictions.jsonl",
        "note": "Synthetic data for scoring smoke test only",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(scenarios) * len(models)} lines to {path}")


if __name__ == "__main__":
    main()
