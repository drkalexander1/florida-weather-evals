"""Anthropic structured JSON predictions via tool use."""

from __future__ import annotations

import os
import time

import anthropic

from src.schema import Prediction, parse_prediction, prediction_json_schema

TOOL_NAME = "submit_weather_rainfall_prediction"
_TEMPERATURE_OK_PREFIXES = ("claude-haiku-", "claude-sonnet-", "claude-3-")


def _anthropic_supports_temperature(model: str) -> bool:
    return model.startswith(_TEMPERATURE_OK_PREFIXES)


class AnthropicProvider:
    def __init__(self, model: str) -> None:
        self.model = model
        self.name = model
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete_structured(self, prompt: str) -> tuple[Prediction, float | None]:
        schema = prediction_json_schema()
        start = time.perf_counter()
        request: dict = {
            "model": self.model,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [
                {
                    "name": TOOL_NAME,
                    "description": "Submit the annual rainfall quantile prediction for this scenario.",
                    "input_schema": schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": TOOL_NAME},
        }
        if _anthropic_supports_temperature(self.model):
            request["temperature"] = 0
        response = self._client.messages.create(**request)
        latency_ms = (time.perf_counter() - start) * 1000
        for block in response.content:
            if block.type == "tool_use" and block.name == TOOL_NAME:
                if not isinstance(block.input, dict):
                    raise ValueError(f"Unexpected tool input type: {type(block.input)}")
                return parse_prediction(block.input), latency_ms
        raise ValueError("No tool_use block in Anthropic response")
