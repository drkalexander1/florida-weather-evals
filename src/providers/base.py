"""Provider protocol for structured weather predictions."""

from __future__ import annotations

from typing import Protocol

from src.schema import Prediction


class Provider(Protocol):
    name: str

    def complete_structured(self, prompt: str) -> tuple[Prediction, float | None]: ...
