from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class BaseProvider(ABC):
    """Abstract interface for LLM providers."""

    def __init__(self, provider_name: str, model_name: str):
        self.provider_name = provider_name
        self.model_name = model_name

    @abstractmethod
    def complete(self, prompt: str, max_tokens: int, model_name: str | None = None) -> dict[str, Any]:
        """Call the model. Returns dict with keys: text, input_tokens, output_tokens."""
        pass

    @abstractmethod
    def list_models(self) -> list[dict[str, Any]]:
        """List available models. Each dict has: name, cost_per_input_token, cost_per_output_token."""
        pass

    def _extract_token_counts(self, response: Any, prompt: str, text: str, input_field: str = "input_tokens", output_field: str = "output_tokens") -> tuple[int, int]:
        """Extract token counts from response, with word-count fallback."""
        usage = response.usage_metadata if hasattr(response, "usage_metadata") else None
        if isinstance(usage, dict):
            input_tokens = usage.get(input_field, len(prompt.split()))
            output_tokens = usage.get(output_field, len(text.split()))
        else:
            input_tokens = len(prompt.split())
            output_tokens = len(text.split())
        return input_tokens, output_tokens

    def _calculate_cost(self, input_tokens: int, output_tokens: int, input_cost: float, output_cost: float) -> float:
        """Calculate cost for a completion."""
        return input_tokens * input_cost + output_tokens * output_cost
