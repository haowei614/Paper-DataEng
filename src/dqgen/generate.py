"""Turn an LLM response into a parsed constraint suite.

Pipeline: :class:`~dqgen.context.PromptContext` -> rendered prompt (jinja2) ->
:class:`~dqgen.llm.CachingClient` -> :func:`~dqgen.parse.parse_response` -> a
:class:`GenerationResult` carrying the valid GX expectations, parse-status
counts, and token/latency bookkeeping for RQ1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import great_expectations as gx
from jinja2 import Environment, FileSystemLoader

from dqgen.context import PromptContext
from dqgen.expectations import render_allowlist
from dqgen.llm import CachingClient, LLMResponse
from dqgen.parse import ParseResult, parse_response

_PROMPT_DIR = Path(__file__).parent / "prompts"
_ENV = Environment(
    loader=FileSystemLoader(str(_PROMPT_DIR)),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)
PROMPT_TEMPLATE = "constraints.jinja2"


def render_prompt(ctx: PromptContext, template: str = PROMPT_TEMPLATE) -> str:
    """Render the constraint-generation prompt for a context."""
    return _ENV.get_template(template).render(ctx=ctx, allowlist=render_allowlist())


@dataclass
class GenerationResult:
    """Everything produced for one (model, dataset, table, condition, rep)."""

    model_name: str
    dataset: str
    table: str
    condition: str
    repetition: int
    prompt: str
    response: LLMResponse
    parse_result: ParseResult
    meta: dict = field(default_factory=dict)

    @property
    def expectations(self) -> list:
        """The valid GX expectation objects."""
        return [r.expectation for r in self.parse_result.valid_rules]

    def to_suite(self, name: str | None = None) -> gx.ExpectationSuite:
        """Assemble a GX ExpectationSuite from the valid expectations."""
        suite_name = name or f"{self.model_name}_{self.dataset}_{self.table}_{self.condition}_r{self.repetition}"
        return gx.ExpectationSuite(name=suite_name, expectations=list(self.expectations))

    def summary(self) -> dict:
        """Flat identifiers + counts + cost, for the results table."""
        counts = self.parse_result.counts()
        total = len(self.parse_result.rules)
        return {
            "model": self.model_name,
            "dataset": self.dataset,
            "table": self.table,
            "condition": self.condition,
            "repetition": self.repetition,
            "n_rules_total": total,
            "n_valid": counts["valid"],
            "n_json_error": counts["json_error"],
            "n_unknown_expectation": counts["unknown_expectation"],
            "n_invalid_kwargs": counts["invalid_kwargs"],
            "json_ok": self.parse_result.json_ok,
            "prompt_tokens": self.response.prompt_tokens,
            "completion_tokens": self.response.completion_tokens,
            "latency_s": self.response.latency_s,
            "cached": self.response.cached,
        }


def generate_suite(
    client: CachingClient,
    ctx: PromptContext,
    repetition: int = 0,
    *,
    dataset: str | None = None,
    table: str | None = None,
) -> GenerationResult:
    """Run one generation: render prompt, call the model, parse the response."""
    prompt = render_prompt(ctx)
    response = client.generate(prompt, cache_salt=f"rep={repetition}")
    parsed = parse_response(response.text)
    return GenerationResult(
        model_name=client.spec.name,
        dataset=dataset or ctx.dataset,
        table=table or ctx.table,
        condition=ctx.condition,
        repetition=repetition,
        prompt=prompt,
        response=response,
        parse_result=parsed,
    )
