"""Anthropic provider: Claude models."""
from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from .base import BaseProvider


class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: str, model_name: str = "claude-sonnet-4-5-20250929", models: list[dict] | None = None):
        super().__init__("anthropic", model_name)
        self.api_key = api_key
        self.models = models or []

    def complete(self, prompt: str, max_tokens: int, model_name: str | None = None) -> dict[str, Any]:
        client = ChatAnthropic(api_key=self.api_key, model_name=self.model_name, max_tokens=max_tokens)  # type: ignore
        response = client.invoke([HumanMessage(content=prompt)])
        text = response.content if hasattr(response, "content") else str(response)
        input_tokens, output_tokens = self._extract_token_counts(response, prompt, text)

        return {
            "text": text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    def list_models(self) -> list[dict[str, Any]]:
        if self.models:
            return self.models
        return [
            {
                "name": self.model_name,
                "cost_per_input_token": 0.003 / 1000,
                "cost_per_output_token": 0.015 / 1000,
            }
        ]
