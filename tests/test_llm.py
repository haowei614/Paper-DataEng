"""Tests for the LLM caching client and cache-key stability.

No network is used: a fake backend records call counts and returns fixed
responses.
"""

from __future__ import annotations

from dqgen.config import ModelSpec
from dqgen.llm import CachingClient, LLMResponse
from dqgen.utils.hashing import cache_key


class FakeBackend:
    model_id = "fake"

    def __init__(self):
        self.calls = 0

    def complete(self, prompt, *, temperature, max_tokens):
        self.calls += 1
        return LLMResponse(
            text=f"response-{self.calls}",
            model=self.model_id,
            prompt_tokens=10,
            completion_tokens=5,
            latency_s=0.01,
        )


def _spec() -> ModelSpec:
    return ModelSpec(name="fake", backend="openai", model="fake", base_url="http://x/v1")


def test_cache_miss_then_hit(tmp_path):
    backend = FakeBackend()
    client = CachingClient(_spec(), cache_dir=tmp_path, backend=backend)

    first = client.generate("hello")
    assert backend.calls == 1
    assert first.cached is False
    assert first.text == "response-1"

    second = client.generate("hello")
    assert backend.calls == 1  # served from cache, no new backend call
    assert second.cached is True
    assert second.text == "response-1"


def test_cache_file_written(tmp_path):
    client = CachingClient(_spec(), cache_dir=tmp_path, backend=FakeBackend())
    client.generate("hello")
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1


def test_different_prompt_is_new_call(tmp_path):
    backend = FakeBackend()
    client = CachingClient(_spec(), cache_dir=tmp_path, backend=backend)
    client.generate("a")
    client.generate("b")
    assert backend.calls == 2


def test_params_change_invalidates_cache(tmp_path):
    backend = FakeBackend()
    c1 = CachingClient(_spec(), cache_dir=tmp_path, backend=backend)
    c1.generate("hello")
    hot_spec = _spec()
    hot_spec.temperature = 0.7  # different params -> different key
    c2 = CachingClient(hot_spec, cache_dir=tmp_path, backend=backend)
    c2.generate("hello")
    assert backend.calls == 2


def test_cache_key_is_stable_and_order_independent():
    k1 = cache_key("m", "p", {"a": 1, "b": 2})
    k2 = cache_key("m", "p", {"b": 2, "a": 1})
    assert k1 == k2
    assert k1 != cache_key("m", "p2", {"a": 1, "b": 2})


def test_llm_response_round_trip():
    r = LLMResponse("t", "m", 1, 2, 0.5)
    assert LLMResponse.from_dict(r.to_dict()) == r
