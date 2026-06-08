"""Pydantic models for weather scenarios and model predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, confloat, field_validator, model_validator

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SCENARIOS_PATH = DATA_DIR / "scenarios.yaml"
FAWN_SYNC_PATH = DATA_DIR / "fawn_sync.json"
PROMPT_PATHS = {
    "natural": ROOT / "prompts" / "natural_v1.txt",
    "statistical": ROOT / "prompts" / "statistical_v1.txt",
}

SpecificityStratum = Literal["specific_station", "regional_inference", "underspecified"]
PromptVariant = Literal["natural", "statistical"]
QUANTILE_LEVELS = (0.1, 0.5, 0.9)


class Scenario(BaseModel):
    id: str
    stratum: SpecificityStratum
    prompt_variant: PromptVariant
    location_description: str
    fawn_station_id: str | None = None
    region: str = "Florida, USA"
    measurement: str = "annual rainfall (inches)"
    target_p10: confloat(gt=0)
    target_p50: confloat(gt=0)
    target_p90: confloat(gt=0)
    notes: str = ""

    @model_validator(mode="after")
    def ordered_quantiles(self) -> Scenario:
        if not (self.target_p10 <= self.target_p50 <= self.target_p90):
            raise ValueError(
                f"target quantiles must satisfy p10 <= p50 <= p90 for {self.id}"
            )
        return self


REASONING_MAX_LENGTH = 400


class Prediction(BaseModel):
    p10: confloat(gt=0)
    p50: confloat(gt=0)
    p90: confloat(gt=0)
    confidence: confloat(ge=0, le=1)
    reasoning: str = Field(max_length=REASONING_MAX_LENGTH)

    @model_validator(mode="after")
    def ordered_quantiles(self) -> Prediction:
        if not (self.p10 <= self.p50 <= self.p90):
            raise ValueError("predicted quantiles must satisfy p10 <= p50 <= p90")
        return self


def _strict_json_schema(schema: dict) -> dict:
    out = dict(schema)
    if out.get("type") == "object":
        out["additionalProperties"] = False
    if "properties" in out:
        out["properties"] = {k: _strict_json_schema(v) for k, v in out["properties"].items()}
    if "items" in out:
        out["items"] = _strict_json_schema(out["items"])
    for key in ("anyOf", "oneOf", "allOf"):
        if key in out:
            out[key] = [_strict_json_schema(s) for s in out[key]]
    if "$defs" in out:
        out["$defs"] = {k: _strict_json_schema(v) for k, v in out["$defs"].items()}
    return out


def prediction_json_schema() -> dict:
    return _strict_json_schema(Prediction.model_json_schema())


def parse_prediction(data: dict) -> Prediction:
    normalized = dict(data)
    reasoning = normalized.get("reasoning")
    if isinstance(reasoning, str) and len(reasoning) > REASONING_MAX_LENGTH:
        normalized["reasoning"] = reasoning[:REASONING_MAX_LENGTH]
    return Prediction.model_validate(normalized)


class PredictionRecord(BaseModel):
    scenario_id: str
    model: str
    provider: str
    prediction: Prediction
    latency_ms: float | None = None
    raw_response: str | None = None


class FawnScenarioSync(BaseModel):
    scenario_id: str
    fawn_station_id: str | None = None
    reference_years: list[float] = Field(
        description="Clean annual rainfall totals (inches) used as CRPS observations"
    )
    p10: confloat(gt=0)
    p50: confloat(gt=0)
    p90: confloat(gt=0)
    mean: confloat(gt=0)
    std: confloat(ge=0)
    target_p10_curator: confloat(gt=0)
    target_p50_curator: confloat(gt=0)
    target_p90_curator: confloat(gt=0)

    @field_validator("reference_years")
    @classmethod
    def non_empty_years(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("reference_years must not be empty")
        return v


class FawnSync(BaseModel):
    schema_version: int
    generated_at: str
    method: str
    method_note: str
    region: str
    years: list[int]
    stations_included: list[str]
    scenarios: dict[str, FawnScenarioSync]


def load_scenarios(path: Path | None = None) -> list[Scenario]:
    path = path or SCENARIOS_PATH
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Expected list in {path}")
    return [Scenario.model_validate(item) for item in raw]


def load_prompt_template(variant: PromptVariant, path: Path | None = None) -> str:
    path = path or PROMPT_PATHS[variant]
    return path.read_text(encoding="utf-8")


def load_fawn_sync(path: Path | None = None) -> FawnSync | None:
    path = path or FAWN_SYNC_PATH
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    return FawnSync.model_validate(raw)


def scoring_reference_years(scenario: Scenario, sync: FawnSync | None) -> list[float]:
    if sync is not None:
        row = sync.scenarios.get(scenario.id)
        if row is not None:
            return [float(v) for v in row.reference_years]
    return [float(scenario.target_p50)]


def scoring_targets(scenario: Scenario, sync: FawnSync | None) -> tuple[float, float, float]:
    if sync is not None:
        row = sync.scenarios.get(scenario.id)
        if row is not None:
            return float(row.p10), float(row.p50), float(row.p90)
    return float(scenario.target_p10), float(scenario.target_p50), float(scenario.target_p90)
