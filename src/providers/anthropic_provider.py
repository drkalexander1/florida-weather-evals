"""Anthropic structured JSON predictions via structured outputs.

Uses output_config.format (json_schema), the same grammar-constrained
mechanism as the OpenAI provider's response_format, so all models across
both providers get an identical elicitation protocol: same prompt, no
tools, constrained JSON output. Works with always-on adaptive thinking
models (claude-fable-5), unlike forced tool_choice.
"""

from __future__ import annotations

import json
import os
import time

import anthropic

from src.schema import Prediction, parse_prediction, prediction_json_schema

_TEMPERATURE_OK_PREFIXES = ("claude-haiku-", "claude-sonnet-", "claude-3-")

# Always-on adaptive thinking; the thinking parameter must not be sent.
_ALWAYS_THINKING_PREFIXES = ("claude-fable-", "claude-mythos-")
# Models where opt-in thinking uses {"type": "adaptive"}; older models
# (e.g. claude-haiku-4-5) only support manual {"type": "enabled", budget_tokens}.
_ADAPTIVE_THINKING_PREFIXES = (
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-4-6",
)
_MANUAL_THINKING_BUDGET = 4096

# Anthropic structured outputs reject these JSON Schema keywords with a 400.
# Pydantic still enforces the constraints in parse_prediction on the way back.
_UNSUPPORTED_SCHEMA_KEYS = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
)


def _anthropic_supports_temperature(model: str) -> bool:
    return model.startswith(_TEMPERATURE_OK_PREFIXES)


def _strip_unsupported(schema: dict) -> dict:
    out = {k: v for k, v in schema.items() if k not in _UNSUPPORTED_SCHEMA_KEYS}
    if "properties" in out:
        out["properties"] = {k: _strip_unsupported(v) for k, v in out["properties"].items()}
    if "items" in out:
        out["items"] = _strip_unsupported(out["items"])
    for key in ("anyOf", "oneOf", "allOf"):
        if key in out:
            out[key] = [_strip_unsupported(s) for s in out[key]]
    if "$defs" in out:
        out["$defs"] = {k: _strip_unsupported(v) for k, v in out["$defs"].items()}
    return out


class AnthropicProvider:
    def __init__(self, model: str, *, thinking: bool = False) -> None:
        self.model = model
        self.name = model
        # Fable/Mythos think unconditionally; the flag only affects other models.
        self.thinking = thinking and not model.startswith(_ALWAYS_THINKING_PREFIXES)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete_structured(self, prompt: str) -> tuple[Prediction, float | None]:
        schema = _strip_unsupported(prediction_json_schema())
        start = time.perf_counter()
        # Generous cap: adaptive-thinking models (e.g. claude-fable-5) spend
        # output tokens on reasoning before the final JSON.
        request: dict = {
            "model": self.model,
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": schema,
                }
            },
        }
        if self.thinking:
            if self.model.startswith(_ADAPTIVE_THINKING_PREFIXES):
                request["thinking"] = {"type": "adaptive"}
            else:
                request["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": _MANUAL_THINKING_BUDGET,
                }
        elif _anthropic_supports_temperature(self.model):
            # Thinking is incompatible with temperature, so temperature=0 is
            # only set on the no-thinking path.
            request["temperature"] = 0
        response = self._client.messages.create(**request)
        latency_ms = (time.perf_counter() - start) * 1000

        text = "".join(block.text for block in response.content if block.type == "text")
        if not text.strip():
            raise ValueError(
                f"Empty structured response (blocks: {[b.type for b in response.content]})"
            )
        return parse_prediction(json.loads(text)), latency_ms
