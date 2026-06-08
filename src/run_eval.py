"""Run LLM evaluation over curated weather scenarios."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import yaml
from dotenv import load_dotenv
from openai import AuthenticationError as OpenAIAuthenticationError

from src.providers.anthropic_provider import AnthropicProvider
from src.providers.openai_provider import OpenAIProvider
from src.schema import (
    PROMPT_PATHS,
    ROOT,
    PredictionRecord,
    Scenario,
    load_prompt_template,
    load_scenarios,
)

DEFAULT_MODELS = ["gpt-4o-mini", "gpt-4o"]
RESULTS_DIR = ROOT / "results" / "latest"
ENV_PATH = ROOT / ".env"


def _require_api_keys(models: list[str]) -> None:
    missing: list[str] = []
    needs_openai = any(not m.startswith("claude") for m in models)
    needs_anthropic = any(m.startswith("claude") for m in models)
    if needs_openai and not os.environ.get("OPENAI_API_KEY", "").strip():
        missing.append("OPENAI_API_KEY")
    if needs_anthropic and not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        missing.append("ANTHROPIC_API_KEY")
    if not missing:
        return

    env_hint = (
        f"Create {ENV_PATH} from .env.example and set: {', '.join(missing)}"
        if not ENV_PATH.exists()
        else f"Set in {ENV_PATH}: {', '.join(missing)}"
    )
    print(
        "API key missing. " + env_hint + "\n"
        "For a no-API smoke test: python scripts/generate_demo_predictions.py",
        file=sys.stderr,
    )
    raise SystemExit(1)


def build_prompt(scenario: Scenario) -> str:
    template = load_prompt_template(scenario.prompt_variant)
    return template.format(
        location_description=scenario.location_description,
        region=scenario.region,
        measurement=scenario.measurement,
    )


def get_provider(model: str):
    if model.startswith("claude"):
        return AnthropicProvider(model)
    return OpenAIProvider(model)


def report_provider_failure(
    exc: Exception,
    *,
    model: str,
    scenario_id: str,
    predictions_path: Path,
) -> None:
    if isinstance(exc, (OpenAIAuthenticationError, anthropic.AuthenticationError)):
        print(
            f"Authentication failed for {model}. "
            "Check OPENAI_API_KEY / ANTHROPIC_API_KEY in .env.",
            file=sys.stderr,
        )
        return
    saved = sum(1 for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip())
    if saved:
        print(
            f"Stopped on [{model}] {scenario_id} ({saved} predictions saved). "
            f"Re-run the same command to resume from {predictions_path}.",
            file=sys.stderr,
        )
    else:
        print(f"Failed on [{model}] {scenario_id}: {exc}", file=sys.stderr)


def load_completed(path: Path) -> dict[tuple[str, str], PredictionRecord]:
    completed: dict[tuple[str, str], PredictionRecord] = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = PredictionRecord.model_validate_json(line)
                completed[(record.scenario_id, record.model)] = record
    return completed


def run(
    scenarios: list[Scenario],
    models: list[str],
    output_dir: Path,
    limit: int | None = None,
    *,
    fresh: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    subset = scenarios[:limit] if limit else scenarios

    predictions_path = output_dir / "predictions.jsonl"
    completed = {} if fresh else load_completed(predictions_path)
    expected = len(subset) * len(models)
    pending = expected - len(completed)

    if pending <= 0:
        print(f"Already complete: {len(completed)}/{expected} predictions in {predictions_path}")
        return predictions_path

    if completed:
        print(f"Resuming: {len(completed)}/{expected} done, {pending} remaining")

    manifest: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "scenario_count": len(subset),
        "predictions_file": str(predictions_path.name),
        "prompt_versions": {k: v.stem for k, v in PROMPT_PATHS.items()},
    }
    if completed:
        manifest["resumed_from"] = len(completed)

    with predictions_path.open("a" if completed else "w", encoding="utf-8") as out:
        for model_name in models:
            provider = get_provider(model_name)
            for scenario in subset:
                key = (scenario.id, model_name)
                if key in completed:
                    continue
                prompt = build_prompt(scenario)
                try:
                    pred, latency_ms = provider.complete_structured(prompt)
                except Exception as exc:
                    report_provider_failure(
                        exc,
                        model=model_name,
                        scenario_id=scenario.id,
                        predictions_path=predictions_path,
                    )
                    raise SystemExit(1) from None
                record = PredictionRecord(
                    scenario_id=scenario.id,
                    model=model_name,
                    provider=provider.__class__.__name__,
                    prediction=pred,
                    latency_ms=latency_ms,
                )
                out.write(record.model_dump_json() + "\n")
                out.flush()
                print(
                    f"[{model_name}] {scenario.id} -> "
                    f"p50={pred.p50:.1f} (width={pred.p90 - pred.p10:.1f})"
                )
                time.sleep(0.2)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    scenarios_copy = output_dir / "scenarios_snapshot.yaml"
    with scenarios_copy.open("w", encoding="utf-8") as f:
        yaml.dump(
            [s.model_dump() for s in subset],
            f,
            default_flow_style=False,
            allow_unicode=True,
        )

    return predictions_path


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ENV_PATH)
    parser = argparse.ArgumentParser(description="Run Florida weather rainfall eval against LLMs")
    parser.add_argument("--scenarios", type=Path, default=ROOT / "data" / "scenarios.yaml")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model ids",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing predictions.jsonl and start over",
    )
    args = parser.parse_args(argv)

    scenarios = load_scenarios(args.scenarios)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    _require_api_keys(models)
    path = run(scenarios, models, args.output, limit=args.limit, fresh=args.fresh)
    print(f"Wrote predictions to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
