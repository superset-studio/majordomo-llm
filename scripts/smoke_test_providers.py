"""Smoke-test majordomo-llm across providers, direct and through Steward.

Two passes:

  Pass 1 — provider capability matrix: one canonical (latest) model per provider,
           exercises text / JSON / structured / stream. Catches provider-class bugs
           in majordomo-llm and in Steward's per-provider translation.

  Pass 2 — per-model smoke: additional representative models per provider,
           exercises text + stream only. Catches "is this model name routable
           through Steward" bugs (the Opus 4.7 class of breakage).

Each cell is run twice: once direct, once through Steward. A diff between the two
isolates Steward bugs from provider/library bugs.

Environment:
  MAJORDOMO_GATEWAY_URL    Steward base URL (default: http://localhost:7680)
  MAJORDOMO_API_KEY        Steward API key (required for the steward leg)
  OPENAI_API_KEY           Required if openai is in scope
  ANTHROPIC_API_KEY        Required if anthropic is in scope
  GEMINI_API_KEY           Required if gemini is in scope
  DEEPSEEK_API_KEY         Required if deepseek is in scope
  CO_API_KEY               Required if cohere is in scope
  AWS_BEARER_TOKEN_BEDROCK Bedrock auto-skips if unset

Usage:
  uv run python scripts/smoke_test_providers.py
  uv run python scripts/smoke_test_providers.py --provider anthropic
  uv run python scripts/smoke_test_providers.py --capability stream
  uv run python scripts/smoke_test_providers.py --capability image --skip-steward
  uv run python scripts/smoke_test_providers.py --skip-direct
"""

from __future__ import annotations

import argparse
import asyncio
import io
import os
import re
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import cast

from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel

from majordomo_llm import ImageInput, get_llm_instance
from majordomo_llm.base import LLM
from majordomo_llm.exceptions import StructuredOutputUnsupported
from majordomo_llm.factory import LLM_CONFIG, get_supported_providers

PROVIDER_API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "cohere": "CO_API_KEY",
    "bedrock": "AWS_BEARER_TOKEN_BEDROCK",
    "bedrock_mantle": "AWS_BEARER_TOKEN_BEDROCK",
    "fireworks": "FIREWORKS_API_KEY",
    "together": "TOGETHER_API_KEY",
}

# Providers currently routable through Steward. Others run direct-only — their
# steward-leg rows are suppressed so they don't pollute the matrix with known
# "not yet supported" failures. Bedrock's Steward routing has code support but
# has not been live-tested; runs here will surface any gaps.
STEWARD_SUPPORTED_PROVIDERS: set[str] = {
    "openai",
    "anthropic",
    "gemini",
    "bedrock",
    "bedrock_mantle",
    "fireworks",
    "together",
}

# Per-provider additional models for Pass 2 (text + stream only).
# Pass 1's canonical model is auto-picked as the first entry in llm_config.yaml.
EXTRA_MODELS: dict[str, list[str]] = {
    "openai": ["gpt-5.4", "gpt-5.4-mini"],
    "anthropic": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
    "gemini": ["gemini-3.5-flash", "gemini-3.1-flash-lite"],
    # Reasoning profile aliases exercise the YAML ``model:`` override + the
    # reasoning_effort/thinking plumbing introduced in v0.10.0.
    "deepseek": ["deepseek-v4-flash", "deepseek-v4-pro-reasoning", "deepseek-v4-pro-hard"],
    "cohere": ["command-r-plus-08-2024"],
    # Bedrock catalog is diverse — cover the model families that exercise each
    # distinct code path (forced toolChoice vs toolChoice-omitted Llama 4) so
    # regressions on any path are caught. Anthropic Claude is no longer served
    # here — see bedrock_mantle below.
    "bedrock": [
        "us.meta.llama4-scout-17b-instruct-v1:0",
        "moonshotai.kimi-k2.5",
        "nvidia.nemotron-nano-12b-v2",
        "deepseek.v3.2",
    ],
    # Mantle hosts the three open Claude SKUs — Pass 1 hits Opus 4.8 (first in
    # YAML); Pass 2 sweeps the rest to confirm each model ID routes cleanly.
    "bedrock_mantle": [
        "anthropic.claude-opus-4-7",
        "anthropic.claude-haiku-4-5",
    ],
    "fireworks": [
        "accounts/fireworks/models/deepseek-v4-pro",
        "deepseek-v4-pro-reasoning",
        "deepseek-v4-pro-hard",
        "accounts/fireworks/models/kimi-k2p6",
    ],
    "together": [
        "deepseek-ai/DeepSeek-V4-Pro",
        "deepseek-v4-pro-reasoning",
        "deepseek-v4-pro-hard",
    ],
}

OK = "✓"
FAIL = "✗"
SKIP = "—"


class _Person(BaseModel):
    name: str
    age: int


@dataclass
class CellResult:
    status: str  # OK, FAIL, SKIP
    elapsed: float = 0.0
    error: str = ""  # Truncated, for inline matrix display.
    full_error: str = ""  # Full exception text, written to the sidecar log.
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0


async def _run_text(llm: LLM, extra_headers: dict[str, str] | None) -> CellResult:
    t = time.time()
    r = await llm.get_response(
        "Reply with just the word OK.", temperature=0.0, extra_headers=extra_headers,
    )
    ok = bool(r.content and r.content.strip())
    return CellResult(OK if ok else FAIL, time.time() - t,
                      "" if ok else "empty content")


async def _run_json(llm: LLM, extra_headers: dict[str, str] | None) -> CellResult:
    t = time.time()
    r = await llm.get_json_response(
        'Reply with the JSON object {"status": "ok"} and nothing else.',
        temperature=0.0,
        extra_headers=extra_headers,
    )
    ok = isinstance(r.content, dict) and "status" in r.content
    return CellResult(OK if ok else FAIL, time.time() - t,
                      "" if ok else f"unexpected payload: {r.content!r}")


async def _run_structured(llm: LLM, extra_headers: dict[str, str] | None) -> CellResult:
    t = time.time()
    try:
        r = await llm.get_structured_json_response(
            response_model=_Person,
            user_prompt="Extract the person: Alice is 30 years old.",
            temperature=0.0,
            extra_headers=extra_headers,
        )
    except StructuredOutputUnsupported:
        return CellResult(SKIP, 0.0, "structured output unsupported")
    ok = isinstance(r.content, _Person) and r.content.age == 30
    return CellResult(OK if ok else FAIL, time.time() - t,
                      "" if ok else f"unexpected model: {r.content!r}")


async def _run_stream(llm: LLM, extra_headers: dict[str, str] | None) -> CellResult:
    t = time.time()
    stream = await llm.get_response_stream(
        "Reply with just the word OK.", temperature=0.0, extra_headers=extra_headers,
    )
    chunks: list[str] = []
    async for chunk in stream:
        chunks.append(chunk)
    text = "".join(chunks)
    ok = bool(text.strip())
    return CellResult(OK if ok else FAIL, time.time() - t,
                      "" if ok else "empty stream")


def _image_fixture() -> ImageInput:
    """Create a deterministic single-color PNG without persisting image bytes."""
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (0, 102, 204)).save(buffer, format="PNG")
    return ImageInput(data=buffer.getvalue(), media_type="image/png")


async def _run_image(llm: LLM, extra_headers: dict[str, str] | None) -> CellResult:
    t = time.time()
    response = await llm.get_response(
        "What is the only color in this image? Reply with exactly BLUE.",
        temperature=0.0,
        images=(_image_fixture(),),
        extra_headers=extra_headers,
    )
    content = response.content if isinstance(response.content, str) else ""
    ok = re.search(r"\bblue\b", content, flags=re.IGNORECASE) is not None
    return CellResult(
        status=OK if ok else FAIL,
        elapsed=time.time() - t,
        error="" if ok else "response did not identify the image as blue",
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        total_cost=response.total_cost,
    )


CapabilityFn = Callable[[LLM, "dict[str, str] | None"], Awaitable[CellResult]]

CAPABILITIES: dict[str, CapabilityFn] = {
    "text": _run_text,
    "json": _run_json,
    "structured": _run_structured,
    "stream": _run_stream,
    "image": _run_image,
}


def _canonical_model(provider: str) -> str:
    """First model in llm_config.yaml for the provider (= latest by convention)."""
    return cast(str, next(iter(LLM_CONFIG[provider]["models"])))


def _steward_default_headers(gateway_key: str, run_id: str) -> dict[str, str]:
    return {
        "X-Majordomo-Key": gateway_key,
        "X-Majordomo-Feature": "smoke-test",
        "X-Majordomo-Project": "majordomo-llm",
        "X-Majordomo-Run-Id": run_id,
    }


def _steward_base_url(provider: str, gateway_url: str) -> str:
    """Per-provider base URL adjustments for routing through Steward.

    The OpenAI SDK assumes its base_url already includes the ``/v1`` path
    segment (its default is ``https://api.openai.com/v1``), so when we point it
    at a bare gateway URL the SDK constructs paths like ``/responses`` instead
    of ``/v1/responses`` and Steward rejects them. Append ``/v1`` for OpenAI
    to match what the SDK expects.

    BedrockMantle uses the bare gateway URL — Steward routes Mantle vs vanilla
    Anthropic on the ``x-majordomo-provider`` header (auto-injected by the
    BedrockMantle provider when ``base_url`` is set), not on the request path.
    """
    base = gateway_url.rstrip("/")
    if provider == "openai":
        return f"{base}/v1"
    return base


def _build_llm(
    provider: str,
    model: str,
    *,
    via_steward: bool,
    gateway_url: str,
    gateway_key: str | None,
    run_id: str,
) -> LLM:
    if via_steward:
        assert gateway_key is not None
        return get_llm_instance(
            provider,
            model,
            base_url=_steward_base_url(provider, gateway_url),
            default_headers=_steward_default_headers(gateway_key, run_id),
        )
    return get_llm_instance(provider, model)


@dataclass
class Row:
    provider: str
    model: str
    route: str  # "direct" or "steward"
    cells: dict[str, CellResult] = field(default_factory=dict)


async def _run_cell(
    provider: str,
    model: str,
    capability: str,
    pass_name: str,
    *,
    via_steward: bool,
    gateway_url: str,
    gateway_key: str | None,
    run_id: str,
) -> CellResult:
    try:
        llm = _build_llm(
            provider, model,
            via_steward=via_steward,
            gateway_url=gateway_url,
            gateway_key=gateway_key,
            run_id=run_id,
        )
        if capability == "image" and not llm.supports_image_input:
            return CellResult(SKIP, 0.0, "image input unsupported by model")
        # Per-call headers only matter on the steward leg; direct providers
        # would just ignore them but we keep the wire clean.
        extra_headers: dict[str, str] | None = None
        if via_steward:
            extra_headers = {
                "X-Majordomo-Capability": capability,
                "X-Majordomo-Pass": pass_name,
            }
        return await CAPABILITIES[capability](llm, extra_headers)
    except Exception as e:  # noqa: BLE001 — smoke test wants to keep going
        full = f"{type(e).__name__}: {e}"
        return CellResult(
            status=FAIL,
            elapsed=0.0,
            error=full[:200].replace("\n", " "),
            full_error=full,
        )


def _row_prefix(provider: str, model: str, route: str) -> str:
    return f"  [{route:7}] {provider:14} {model:48}"


def _print_cell_done(
    provider: str,
    model: str,
    route: str,
    capability: str,
    cell: CellResult,
) -> None:
    """Print one line per cell as it completes. Live progress without TTY
    tricks — each cell's line appears as soon as the call returns."""
    elapsed = f"({cell.elapsed:.1f}s)" if cell.elapsed else ""
    err = f" — {cell.error}" if cell.error else ""
    usage = ""
    if cell.status == OK and capability == "image":
        usage = (
            f" tokens={cell.input_tokens}+{cell.output_tokens}"
            f" cost=${cell.total_cost:.6f}"
        )
    print(
        f"{_row_prefix(provider, model, route)} "
        f"{capability:11} {cell.status} {elapsed}{usage}{err}",
        flush=True,
    )


async def _run_all(
    providers: list[str],
    capabilities: list[str],
    routes: list[tuple[str, bool]],
    gateway_url: str,
    gateway_key: str | None,
    run_id: str,
) -> int:
    all_rows: list[Row] = []

    def routes_for(provider: str) -> list[tuple[str, bool]]:
        return [
            (name, via) for name, via in routes
            if not via or provider in STEWARD_SUPPORTED_PROVIDERS
        ]

    pass1_name = "capability-matrix"
    print("=" * 90)
    print(f"Pass 1 ({pass1_name}) — canonical model per provider × all capabilities")
    print("=" * 90)
    for provider in providers:
        model = _canonical_model(provider)
        for route_name, via_steward in routes_for(provider):
            row = Row(provider=provider, model=model, route=route_name)
            for cap in capabilities:
                row.cells[cap] = await _run_cell(
                    provider, model, cap, pass1_name,
                    via_steward=via_steward,
                    gateway_url=gateway_url,
                    gateway_key=gateway_key,
                    run_id=run_id,
                )
                _print_cell_done(provider, model, route_name, cap, row.cells[cap])
            all_rows.append(row)

    pass2_caps = [c for c in ("text", "stream") if c in capabilities]
    pass2_name = "per-model-smoke"
    print()
    print("=" * 90)
    print(f"Pass 2 ({pass2_name}) — extra models × text + stream")
    print("=" * 90)
    if not pass2_caps:
        print("  (skipped — neither text nor stream in --capability filter)")
    else:
        any_pass2 = False
        for provider in providers:
            for model in EXTRA_MODELS.get(provider, []):
                any_pass2 = True
                for route_name, via_steward in routes_for(provider):
                    row = Row(provider=provider, model=model, route=route_name)
                    for cap in pass2_caps:
                        row.cells[cap] = await _run_cell(
                            provider, model, cap, pass2_name,
                            via_steward=via_steward,
                            gateway_url=gateway_url,
                            gateway_key=gateway_key,
                            run_id=run_id,
                        )
                        _print_cell_done(provider, model, route_name, cap, row.cells[cap])
                    all_rows.append(row)
        if not any_pass2:
            print("  (no extra models configured for selected providers)")

    print()
    print("=" * 90)
    print("Failures")
    print("=" * 90)
    fail_count = 0
    log_path = f"smoke-test-{run_id}.log"
    log_lines: list[str] = []
    for row in all_rows:
        for cap, cell in row.cells.items():
            if cell.status == FAIL:
                fail_count += 1
                print(f"  [{row.route:7}] {row.provider}/{row.model} {cap}: {cell.error}")
                log_lines.append(
                    f"[{row.route}] {row.provider}/{row.model} {cap}\n"
                    f"{cell.full_error}\n"
                    f"{'-' * 80}\n"
                )
    if fail_count == 0:
        print("  (none)")
    elif log_lines:
        with open(log_path, "w") as f:
            f.write(f"Smoke-test run {run_id}\n{'=' * 80}\n\n")
            f.writelines(log_lines)
        print()
        print(f"  Full error bodies written to: {log_path}")

    # Highlight steward-vs-direct divergences — these are the most actionable signal.
    print()
    print("=" * 90)
    print("Steward regressions (cells that passed direct but failed through Steward)")
    print("=" * 90)
    by_key: dict[tuple[str, str, str], dict[str, CellResult]] = {}
    for row in all_rows:
        for cap, cell in row.cells.items():
            by_key.setdefault((row.provider, row.model, cap), {})[row.route] = cell
    divergence_count = 0
    for (provider, model, cap), routes_map in sorted(by_key.items()):
        direct = routes_map.get("direct")
        steward = routes_map.get("steward")
        if direct and steward and direct.status == OK and steward.status == FAIL:
            divergence_count += 1
            print(f"  {provider}/{model} {cap}: {steward.error}")
    if divergence_count == 0:
        print("  (none)")

    return fail_count


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--provider", action="append", default=None,
        choices=get_supported_providers(),
        help="Restrict to one provider (repeatable).",
    )
    parser.add_argument(
        "--capability", action="append", default=None,
        choices=list(CAPABILITIES.keys()),
        help="Restrict to one capability (repeatable).",
    )
    parser.add_argument(
        "--skip-direct", action="store_true",
        help="Only run through-steward calls.",
    )
    parser.add_argument(
        "--skip-steward", action="store_true",
        help="Only run direct calls.",
    )
    args = parser.parse_args()

    if args.skip_direct and args.skip_steward:
        print("ERROR: cannot pass both --skip-direct and --skip-steward.", file=sys.stderr)
        return 2

    gateway_url = os.environ.get("MAJORDOMO_GATEWAY_URL", "http://localhost:7680")
    gateway_key = os.environ.get("MAJORDOMO_API_KEY")
    if not args.skip_steward and not gateway_key:
        print(
            "ERROR: MAJORDOMO_API_KEY not set (required for steward leg). "
            "Pass --skip-steward to run only direct calls.",
            file=sys.stderr,
        )
        return 2

    requested_providers = args.provider or get_supported_providers()
    capabilities = args.capability or list(CAPABILITIES.keys())

    runnable_providers: list[str] = []
    for provider in requested_providers:
        env_var = PROVIDER_API_KEY_ENV[provider]
        if os.environ.get(env_var):
            runnable_providers.append(provider)
            continue
        if provider == "bedrock":
            print(f"[skip] bedrock: {env_var} not set", file=sys.stderr)
            continue
        print(
            f"ERROR: {env_var} not set (required for provider {provider!r}). "
            f"Set the env var or pass --provider to exclude it.",
            file=sys.stderr,
        )
        return 2

    if not runnable_providers:
        print("ERROR: no runnable providers.", file=sys.stderr)
        return 2

    routes: list[tuple[str, bool]] = []
    if not args.skip_direct:
        routes.append(("direct", False))
    if not args.skip_steward:
        routes.append(("steward", True))

    run_id = str(uuid.uuid4())
    steward_only_direct = [
        p for p in runnable_providers if p not in STEWARD_SUPPORTED_PROVIDERS
    ]
    print(f"Run ID: {run_id}")
    print(f"Gateway: {gateway_url}")
    print(f"Providers: {', '.join(runnable_providers)}")
    print(f"Capabilities: {', '.join(capabilities)}")
    print(f"Routes: {', '.join(name for name, _ in routes)}")
    if steward_only_direct and not args.skip_steward:
        print(
            f"Direct-only (not yet routable via Steward): "
            f"{', '.join(steward_only_direct)}"
        )
    print()

    fail_count = asyncio.run(_run_all(
        runnable_providers, capabilities, routes, gateway_url, gateway_key, run_id,
    ))
    return 1 if fail_count else 0


if __name__ == "__main__":
    sys.exit(main())
