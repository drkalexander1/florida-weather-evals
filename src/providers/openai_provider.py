"""OpenAI structured JSON predictions."""

from __future__ import annotations

import json
import os
import time

from openai import OpenAI

from src.schema import Prediction, parse_prediction, prediction_json_schema


def _openai_supports_temperature(model: str) -> bool:
    return not model.startswith(("gpt-5", "o3", "o4"))


class OpenAIProvider:
    def __init__(self, model: str) -> None:
        self.model = model
        self.name = model
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=api_key)

    def complete_structured(self, prompt: str) -> tuple[Prediction, float | None]:
        schema = prediction_json_schema()
        start = time.perf_counter()
        request: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "weather_rainfall_prediction",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        if _openai_supports_temperature(self.model):
            request["temperature"] = 0
        response = self._client.chat.completions.create(**request)
        latency_ms = (time.perf_counter() - start) * 1000
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from OpenAI")
        data = json.loads(content)
        return parse_prediction(data), latency_ms
