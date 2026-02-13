from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Iterable

import requests


DEFAULT_EMBED_TIMEOUT = 15


@dataclass
class EmbeddingConfig:
    provider: str
    model: str | None = None
    base_url: str | None = None
    timeout: int = DEFAULT_EMBED_TIMEOUT
    api_key: str | None = None


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def resolve_lmstudio_model(base_url: str, timeout: int) -> str | None:
    url = base_url.rstrip("/") + "/models"
    headers = {"User-Agent": "mini-nexen/0.1 (+https://example.com)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    models = data.get("data") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return None
    for item in models:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "")
        lowered = model_id.lower()
        if "embed" in lowered or "embedding" in lowered:
            return model_id
    return None


class EmbeddingClient:
    def __init__(self, config: EmbeddingConfig):
        self.config = config

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        provider = (self.config.provider or "").lower()
        if provider == "gemini":
            return self._embed_gemini(texts)
        return self._embed_lmstudio(texts)

    def _embed_lmstudio(self, texts: list[str]) -> list[list[float]]:
        base_url = (
            self.config.base_url
            or os.getenv("MINI_NEXEN_EMBED_BASE_URL")
            or os.getenv("LMSTUDIO_BASE_URL")
        )
        if not base_url:
            return []
        model = self.config.model
        if not model:
            try:
                model = resolve_lmstudio_model(base_url, timeout=self.config.timeout)
            except Exception:
                return []
        if not model:
            return []
        url = base_url.rstrip("/") + "/embeddings"
        headers = {"Content-Type": "application/json"}
        payload = {"model": model, "input": texts}
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.config.timeout)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("data", []):
            embedding = item.get("embedding")
            if isinstance(embedding, list):
                results.append([float(x) for x in embedding])
        return results

    def _embed_gemini(self, texts: list[str]) -> list[list[float]]:
        try:
            from google import genai  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("google-genai is required for Gemini embeddings.") from exc

        model = self.config.model or "gemini-embedding-001"
        api_key = self.config.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        client = genai.Client(api_key=api_key) if api_key else genai.Client()
        output: list[list[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = client.models.embed_content(model=model, contents=batch)
            embeddings = getattr(result, "embeddings", None) or []
            for embedding in embeddings:
                values = None
                if isinstance(embedding, dict):
                    values = embedding.get("values") or embedding.get("embedding")
                else:
                    values = getattr(embedding, "values", None)
                if values is None and isinstance(embedding, list):
                    values = embedding
                if values:
                    output.append([float(x) for x in values])
        return output


def batch_embed(
    client: EmbeddingClient,
    texts: list[str],
    batch_size: int = 32,
) -> list[list[float]]:
    output: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        output.extend(client.embed_texts(batch))
    return output
