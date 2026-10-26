"""Provider configuration: load from env vars and optional YAML."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from dotenv import dotenv_values

# Find .env file relative to project root (2 levels up from config.py)
config_dir = Path(__file__).parent  # src/app
env_file = config_dir.parent.parent / ".env"  # ../../.env
env_vars = dotenv_values(env_file)

@dataclass
class ModelConfig:
    name: str
    cost_per_input_token: float
    cost_per_output_token: float


@dataclass
class ProviderConfig:
    name: str
    provider_type: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    models: list[ModelConfig] | None = None

    def __post_init__(self):
        if self.models is None:
            self.models = []


def load_providers_from_env() -> list[ProviderConfig]:
    """Load provider configs from .env file ONLY (not system env)."""
    providers: dict[str, ProviderConfig] = {}

    anthropic_key = env_vars.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        providers["anthropic"] = ProviderConfig(
            name="anthropic",
            provider_type="anthropic",
            api_key=anthropic_key,
            # hardcoding models because we will have always same model to hit api 
            models=[
                ModelConfig(
                    name="claude-sonnet-4-5-20250929",
                    cost_per_input_token=0.003 / 1000,
                    cost_per_output_token=0.015 / 1000,
                )
            ],
        )

    # google_key = env_vars.get("GOOGLE_API_KEY")
    # if google_key:
    #     providers["google"] = ProviderConfig(
    #         name="google",
    #         provider_type="google",
    #         api_key=google_key,
    #         # hardcoding models because we will have always same model to hit api 
    #         models=[
    #             ModelConfig(
    #                 name="gemini-1.5-flash",
    #                 cost_per_input_token=0.075 / 1000000,
    #                 cost_per_output_token=0.3 / 1000000,
    #             )
    #         ],
    #     )

    ollama_primary = env_vars.get("OLLAMA_PRIMARY_URL")
    if ollama_primary:
        providers["ollama"] = ProviderConfig(
            name="ollama",
            provider_type="ollama",
            base_url=ollama_primary,
            # Models are empty because Ollama is intentionally different because it discovers models dynamically at runtime, not from static config
            models=[],
        )

    return list(providers.values())
