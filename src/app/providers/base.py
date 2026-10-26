from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class BaseProvider(ABC):
    """Abstract interface for LLM providers."""

    def __init__(self, provider_name: str, model_name: str):
        self.provider_name = provider_name
        self.model_name = model_name

    @abstractmethod
    def complete(self, prompt: str, max_tokens: int) -> dict[str, Any]:
        """Call the model. Returns dict with keys: text, input_tokens, output_tokens."""
        pass

    @abstractmethod
    def list_models(self) -> list[dict[str, Any]]:
        """List available models. Each dict has: name, cost_per_input_token, cost_per_output_token."""
        pass

    def _calculate_cost(self, input_tokens: int, output_tokens: int, input_cost: float, output_cost: float) -> float:
        """Calculate cost for a completion."""
        return input_tokens * input_cost + output_tokens * output_cost
