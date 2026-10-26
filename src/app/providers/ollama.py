"""Ollama adapter: discover installed models and call them."""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from langchain_ollama import ChatOllama

from .base import BaseProvider

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class OllamaProvider(BaseProvider):
    def __init__(self, name: str, base_url: str, timeout_s: float = 5.0):
        if not base_url:
            raise ValueError(f"{name}: base_url is required")
        super().__init__("ollama", "local-model")
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.client = ChatOllama(base_url=self.base_url, model="llama2")  # type: ignore
        self.available_models = {}
        self._load_available_models()

    def _load_available_models(self):
        """Cache available models at init time."""
        try:
            with _OPENER.open(f"{self.base_url}/api/tags", timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode())
            for m in data.get("models", []):
                self.available_models[m["name"]] = m
        except OSError:
            pass

    def complete(self, prompt: str, max_tokens: int) -> dict[str, Any]:
        """Call Ollama. Note: Ollama doesn't provide token counts; estimate from prompt/response."""
        from langchain_core.messages import HumanMessage

        if not self.available_models:
            raise RuntimeError("No models available on Ollama server")

        first_model = next(iter(self.available_models.keys()))
        client = ChatOllama(base_url=self.base_url, model=first_model, num_predict=max_tokens)  # type: ignore
        response = client.invoke([HumanMessage(content=prompt)])
        text = response.content if hasattr(response, "content") else str(response)
        input_tokens = len(prompt.split())
        output_tokens = len(text.split())
        return {
            "text": text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    def list_models(self) -> list[dict[str, Any]]:
        """List installed models with zero cost (local inference)."""
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