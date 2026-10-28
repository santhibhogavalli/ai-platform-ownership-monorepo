"""Google provider: Gemini models."""
from __future__ import annotations

from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from .base import BaseProvider


class GoogleProvider(BaseProvider):
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash", models: list[dict] | None = None):
        super().__init__("google", model_name)
        self.api_key = api_key
        self.models = models or []

    def complete(self, prompt: str, max_tokens: int, model_name: str | None = None) -> dict[str, Any]:
        model = model_name or self.model_name
        client = ChatGoogleGenerativeAI(api_key=self.api_key, model=model, max_tokens=max_tokens)  # type: ignore
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
                "cost_per_input_token": 0.075 / 1000000,
                "cost_per_output_token": 0.3 / 1000000,
            }
        ]
