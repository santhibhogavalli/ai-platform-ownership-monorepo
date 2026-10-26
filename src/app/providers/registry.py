"""Provider registry: load, select, and fallback across providers."""
from __future__ import annotations

from typing import Any, Optional

from ..config import ProviderConfig, load_providers_from_env
from .base import BaseProvider
from .factory import ProviderFactory


class ProviderRegistry:
    """Singleton registry for managing all LLM providers."""

    _instance: ProviderRegistry | None = None
    _initialized: bool = False

    def __new__(cls, configs: list[ProviderConfig] | None = None) -> ProviderRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, configs: list[ProviderConfig] | None = None):
        if self._initialized:
            return
        if configs is None:
            configs = load_providers_from_env()
        self.configs = configs
        self.providers: dict[str, BaseProvider] = {}
        self.factory = ProviderFactory()
        self._load_providers()
        ProviderRegistry._initialized = True

    def _load_providers(self):
        """Instantiate all configured providers using factory."""
        for config in self.configs:
            try:
                provider = self.factory.create_provider(config)
                if provider:
                    self.providers[config.provider_type] = provider
            except Exception:
                pass

    def list_all_models(self) -> list[dict[str, Any]]:
        """Return all models across all providers, sorted by cost (cheapest first)."""
        models = []
        for provider_name, provider in self.providers.items():
            for model_dict in provider.list_models():
                model_dict["provider"] = provider_name
                models.append(model_dict)
        models.sort(key=lambda m: m.get("cost_per_input_token", 0))
        return models

    def select_model(self, max_cost: Optional[float] = None) -> tuple[BaseProvider, str, str]:
        """
        Select cheapest available model.

        Args:
            max_cost: Max input token cost. If set, filter to models ≤ this cost.

        Returns:
            (provider, model_name, reasoning)
        """
        all_models = self.list_all_models()
        if not all_models:
            raise ValueError("No providers available")

        if max_cost is not None:
            filtered = [m for m in all_models if m["cost_per_input_token"] <= max_cost]
            if not filtered:
                raise ValueError(f"No models available under ${max_cost}/1k input tokens")
            selected = filtered[0]
        else:
            selected = all_models[0]

        provider_name = selected["provider"]
        model_name = selected["name"]
        cost = selected["cost_per_input_token"]
        reasoning = f"selected for lowest cost (${cost*1000:.3f}/1k input tokens)"

        return self.providers[provider_name], model_name, reasoning

    def complete_with_fallback(self, prompt: str, max_tokens: int, max_cost: Optional[float] = None) -> dict[str, Any]:
        """
        Complete a prompt, falling back to next-cheapest provider on failure.

        Returns standardized response with text, tokens, cost, and selection reasoning.
        """
        all_models = self.list_all_models()
        if not all_models:
            raise ValueError("No providers available")

        if max_cost is not None:
            all_models = [m for m in all_models if m["cost_per_input_token"] <= max_cost]
            if not all_models:
                raise ValueError(f"No models available under ${max_cost}/1k input tokens")

        last_error = None
        for model_dict in all_models:
            provider_name = model_dict["provider"]
            model_name = model_dict["name"]
            provider = self.providers[provider_name]

            try:
                completion = provider.complete(prompt, max_tokens)
                input_tokens = completion["input_tokens"]
                output_tokens = completion["output_tokens"]
                input_cost = model_dict["cost_per_input_token"]
                output_cost = model_dict["cost_per_output_token"]
                total_cost = input_tokens * input_cost + output_tokens * output_cost

                reasoning = f"selected for lowest cost (${input_cost*1000:.3f}/1k input tokens)"

                return {
                    "text": completion["text"],
                    "model": {
                        "name": model_name,
                        "provider": provider_name,
                        "reasoning": reasoning,
                    },
                    "tokens": {
                        "input": input_tokens,
                        "output": output_tokens,
                        "total": input_tokens + output_tokens,
                    },
                    "cost": {
                        "input_cost_per_token": input_cost,
                        "output_cost_per_token": output_cost,
                        "total_cost": total_cost,
                    },
                }
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(f"All providers failed. Last error: {last_error}")
