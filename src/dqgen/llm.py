"""LLM backend abstraction with retries and on-disk response caching.

Two backends sit behind one :class:`Backend` protocol:

* :class:`OpenAICompatBackend` — any OpenAI-compatible ``/chat/completions``
  server (Ollama, vLLM, llama.cpp, OpenAI, ...).
* :class:`AnthropicBackend` — the official Anthropic Messages API.

:class:`CachingClient` wraps a backend and persists every raw response keyed by
``sha256(model, prompt, params)`` so reruns are free and fully reproducible.
Token counts and latency are captured on every call (cache hits report the
cached latency and ``cached=True``).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from dqgen.config import ModelSpec
from dqgen.utils.hashing import cache_key

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """A single completion plus its bookkeeping."""

    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_s: float
    cached: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LLMResponse":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


class Backend(Protocol):
    """Minimal completion interface implemented by every backend."""

    model_id: str

    def complete(self, prompt: str, *, temperature: float, max_tokens: int) -> LLMResponse: ...


# Retry only transient connection failures. A read timeout means the request
# was received but generation is slow — retrying just re-runs it, so we let it
# propagate and rely on the (configurable) per-request timeout instead.
_RETRY = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type(
        (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError, ConnectionError)
    ),
)


class OpenAICompatBackend:
    """OpenAI-compatible ``/chat/completions`` backend (Ollama, vLLM, OpenAI...)."""

    def __init__(
        self, model: str, base_url: str, api_key: str | None = None, timeout: float = 120.0
    ):
        self.model_id = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    @_RETRY
    def complete(self, prompt: str, *, temperature: float, max_tokens: int) -> LLMResponse:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        start = time.perf_counter()
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=body,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        latency = time.perf_counter() - start
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=self.model_id,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_s=latency,
        )


class AnthropicBackend:
    """Anthropic Messages API backend (direct API, ``ANTHROPIC_API_KEY``)."""

    def __init__(self, model: str, api_key: str | None = None):
        from anthropic import Anthropic

        self.model_id = model
        self._client = Anthropic(api_key=api_key) if api_key else Anthropic()

    @_RETRY
    def complete(self, prompt: str, *, temperature: float, max_tokens: int) -> LLMResponse:
        start = time.perf_counter()
        msg = self._client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.perf_counter() - start
        text = "".join(block.text for block in msg.content if getattr(block, "type", None) == "text")
        return LLMResponse(
            text=text,
            model=self.model_id,
            prompt_tokens=msg.usage.input_tokens,
            completion_tokens=msg.usage.output_tokens,
            latency_s=latency,
        )


def build_backend(spec: ModelSpec) -> Backend:
    """Instantiate the backend described by ``spec``."""
    api_key = os.environ.get(spec.api_key_env) if spec.api_key_env else None
    if spec.backend == "openai":
        if not spec.base_url:
            raise ValueError(f"model {spec.name!r}: openai backend requires base_url")
        return OpenAICompatBackend(
            spec.model, spec.base_url, api_key=api_key, timeout=spec.timeout_s
        )
    if spec.backend == "anthropic":
        return AnthropicBackend(spec.model, api_key=api_key)
    raise ValueError(f"unknown backend: {spec.backend!r}")


class CachingClient:
    """Wrap a backend with a content-addressed on-disk response cache.

    The cache key combines the model label, the prompt, and the sampling
    parameters, so any change to any of them produces a fresh call.
    """

    def __init__(self, spec: ModelSpec, cache_dir: Path, backend: Backend | None = None):
        self.spec = spec
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._backend = backend if backend is not None else build_backend(spec)

    def _key(self, prompt: str, cache_salt: str | None) -> str:
        params = self.spec.params()
        if cache_salt is not None:
            params = {**params, "cache_salt": cache_salt}
        return cache_key(self.spec.name, prompt, params)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def generate(self, prompt: str, cache_salt: str | None = None) -> LLMResponse:
        """Generate a completion, using the cache.

        ``cache_salt`` (e.g. ``"rep=2"``) folds into the cache key so repeated
        generations under identical params are stored — and re-issued — as
        distinct calls, capturing any nondeterminism across repetitions.
        """
        key = self._key(prompt, cache_salt)
        path = self._path(key)
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                cached = json.load(fh)
            logger.debug("cache hit for %s (%s)", self.spec.name, key[:12])
            resp = LLMResponse.from_dict(cached["response"])
            resp.cached = True
            return resp

        resp = self._backend.complete(
            prompt, temperature=self.spec.temperature, max_tokens=self.spec.max_tokens
        )
        record = {
            "key": key,
            "model_name": self.spec.name,
            "params": self.spec.params(),
            "prompt": prompt,
            "response": resp.to_dict(),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)
        return resp
