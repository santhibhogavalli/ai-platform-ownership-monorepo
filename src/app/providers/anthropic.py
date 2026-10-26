"""Anthropic provider: Claude models."""
from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from .base import BaseProvider


class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: str, model_name: str = "claude-sonnet-4-5-20250929"):
        super().__init__("anthropic", model_name)
        self.api_key = api_key
        self.input_cost = 0.003 / 1000
        self.output_cost = 0.015 / 1000

    def complete(self, prompt: str, max_tokens: int) -> dict[str, Any]:
        client = ChatAnthropic(api_key=self.api_key, model_name=self.model_name, max_tokens=max_tokens)  # type: ignore
        response = client.invoke([HumanMessage(content=prompt)])
        text = response.content if hasattr(response, "content") else str(response)
        usage = response.usage_metadata if hasattr(response, "usage_metadata") else None

        input_tokens = usage.get("input_tokens", 0) if isinstance(usage, dict) else 0
        output_tokens = usage.get("output_tokens", 0) if isinstance(usage, dict) else 0

        return {
            "text": text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "name": self.model_name,
                "cost_per_input_token": self.input_cost,
                "cost_per_output_token": self.output_cost,
            }
        ]
