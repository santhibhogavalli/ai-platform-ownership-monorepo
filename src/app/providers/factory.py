"""Factory for creating provider instances."""
from __future__ import annotations

from typing import Optional

from ..config import ProviderConfig
from .anthropic import AnthropicProvider
from .base import BaseProvider
from .google import GoogleProvider
from .ollama import OllamaProvider


class ProviderFactory:
    """Factory for creating LLM provider instances from config."""

    _provider_classes = {
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "ollama": OllamaProvider,
    }

    def create_provider(self, config: ProviderConfig) -> Optional[BaseProvider]:
        """
        Create a provider instance from config.

        Args:
            config: Provider configuration

        Returns:
            Provider instance or None if config is invalid or provider type is unsupported
        """
        if config.provider_type not in self._provider_classes:
            return None

        provider_class = self._provider_classes[config.provider_type]

        if config.provider_type == "anthropic":
            if not config.api_key:
                return None
            model_name = config.models[0].name if config.models else "claude-3-5-sonnet"
            return AnthropicProvider(api_key=config.api_key, model_name=model_name)

        elif config.provider_type == "google":
            if not config.api_key:
                return None
            model_name = config.models[0].name if config.models else "gemini-1.5-flash"
            return GoogleProvider(api_key=config.api_key, model_name=model_name)

        elif config.provider_type == "ollama":
            if not config.base_url:
                return None
            return OllamaProvider(name="ollama", base_url=config.base_url)

        return None

    @classmethod
    def register_provider(cls, provider_type: str, provider_class: type[BaseProvider]):
        """Register a new provider type."""
        cls._provider_classes[provider_type] = provider_class
