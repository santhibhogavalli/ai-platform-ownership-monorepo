"""Ollama adapter: discover installed models and call them."""
from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any

from langchain_ollama import ChatOllama

from .base import BaseProvider

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class OllamaProvider(BaseProvider):
    def __init__(self, name: str, base_url: str, timeout_s: float = 5.0, cache_ttl_s: float = 300.0):
        if not base_url:
            raise ValueError(f"{name}: base_url is required")
        super().__init__("ollama", "local-model")
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.cache_ttl_s = cache_ttl_s
        self.available_models = {}
        self.models_cache_time = 0
        self._load_available_models()

    def _load_available_models(self):
        """Refresh available models cache. Expires after cache_ttl_s."""
        try:
            with _OPENER.open(f"{self.base_url}/api/tags", timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode())
            self.available_models = {m["name"]: m for m in data.get("models", [])}
            self.models_cache_time = time.time()
        except OSError:
            pass

    def _should_refresh_cache(self) -> bool:
        """Check if model cache has expired."""
        return time.time() - self.models_cache_time > self.cache_ttl_s

    def complete(self, prompt: str, max_tokens: int, model_name: str | None = None) -> dict[str, Any]:
        """Call Ollama. Uses actual token counts from Ollama API when available."""
        from langchain_core.messages import HumanMessage

        if not self.available_models:
            self._load_available_models()
        if not self.available_models:
            raise RuntimeError("No models available on Ollama server")

        if model_name and model_name not in self.available_models:
            raise RuntimeError(f"Model '{model_name}' not available on {self.name}. Available: {list(self.available_models.keys())}")
        model = model_name or next(iter(self.available_models.keys()))
        client = ChatOllama(base_url=self.base_url, model=model, num_predict=max_tokens, timeout=self.timeout_s)  # type: ignore
        response = client.invoke([HumanMessage(content=prompt)])
        text = response.content if hasattr(response, "content") else str(response)

        input_tokens, output_tokens = self._extract_token_counts(response, prompt, text, "prompt_eval_count", "eval_count")

        return {
            "text": text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    def list_models(self) -> list[dict[str, Any]]:
        """List installed models with zero cost (local inference). Refreshes cache if expired."""
        if self._should_refresh_cache():
            self._load_available_models()

        return [
            {
                "name": m["name"],
                "cost_per_input_token": 0.0,
                "cost_per_output_token": 0.0,
                "digest": m.get("digest", ""),
                "size_bytes": m.get("size", 0),
            }
            for m in self.available_models.values()
        ]


def load_ollama_hosts() -> list[OllamaProvider]:
    """Read config when called, not at import time."""
    hosts = []
    for prefix in ("PRIMARY", "BACKUP"):
        url = os.environ.get(f"OLLAMA_{prefix}_URL")
        if url:
            hosts.append(OllamaProvider(prefix.lower(), url))
    return hosts