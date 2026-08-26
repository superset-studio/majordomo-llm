#!/usr/bin/env python3
"""
Verify the ``max_tokens`` ceiling of every capped model against the live API.

Usage:
    uv run python scripts/check_max_tokens.py
    uv run python scripts/check_max_tokens.py --provider anthropic

Only three providers send an output cap: ``anthropic`` and ``bedrock_mantle``
(the Messages API requires ``max_tokens``) and ``bedrock`` (Converse requires
``inferenceConfig.maxTokens``). Everywhere else the model's own default applies
and ``llm_config.yaml`` carries no ``max_tokens`` key.

The configured ceilings were seeded from a third-party catalog. This script
checks each one against the vendor itself by requesting a deliberately absurd
cap and reading the ceiling back out of the rejection. The request is refused at
validation, before any inference runs, so the sweep costs nothing.

Credentials are read from the environment or a local ``.env`` (loaded via
``python-dotenv``, as ``scripts/smoke_test_providers.py`` does): ``ANTHROPIC_API_KEY``
for anthropic, and ``AWS_BEARER_TOKEN_BEDROCK`` + ``AWS_REGION`` for the two Bedrock
providers. A provider whose key is absent is skipped with one line rather than
reporting a failure per model.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from majordomo_llm import get_llm_instance  # noqa: E402
from majordomo_llm.base import DEFAULT_STREAM_MAX_TOKENS  # noqa: E402
from majordomo_llm.exceptions import MajordomoError  # noqa: E402

load_dotenv()

CONFIG_PATH = Path(__file__).parent.parent / "src" / "majordomo_llm" / "llm_config.yaml"

#: Providers whose API requires an output cap, and therefore carry the key.
CAPPED_PROVIDERS = ("anthropic", "bedrock_mantle", "bedrock")

#: Credential each capped provider needs before it is worth probing at all.
PROVIDER_API_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "bedrock_mantle": "AWS_BEARER_TOKEN_BEDROCK",
    "bedrock": "AWS_BEARER_TOKEN_BEDROCK",
}

#: Far above any real ceiling, so every vendor rejects it and names its own.
ABSURD_MAX_TOKENS = 99_000_000

#: Vendors report the true ceiling inside the 400 body. The wording differs, but
#: every one of them puts the number next to "max_tokens"/"maxTokens".
CEILING_PATTERNS = (
    # Anthropic: "max_tokens: 99000000 > 128000, which is the maximum allowed
    # number of output tokens for claude-..."
    re.compile(r">\s*(\d+),?\s*which is the maximum"),
    re.compile(r"less than or equal to (\d+)"),
    re.compile(r"maximum(?: allowed)?(?: value)?(?: is)?[: ]+(\d+)"),
    re.compile(r"at most (\d+)"),
    re.compile(r"max(?:imum)?[_ ]tokens[^0-9]{0,40}(\d{3,})", re.IGNORECASE),
    re.compile(r"(\d{4,})\s*(?:is the|tokens? maximum)"),
)

DIVIDER = "═" * 72


@dataclass
class Result:
    provider: str
    model: str
    configured: int
    reported: int | None
    detail: str

    @property
    def status(self) -> str:
        if self.reported is None:
            return "UNKNOWN"
        if self.configured == 0:
            # Not pinned: the model inherits the library defaults. Report the
            # vendor ceiling for reference, and flag only if it is low enough
            # that the default would be rejected.
            return "LOW" if self.reported < DEFAULT_STREAM_MAX_TOKENS else "INHERITS"
        if self.reported == self.configured:
            return "OK"
        return "MISMATCH"


def runnable_providers(only: str | None) -> tuple[list[str], list[str]]:
    """Split the capped providers into those with credentials and those without."""
    wanted = [p for p in CAPPED_PROVIDERS if not only or p == only]
    have = [p for p in wanted if os.environ.get(PROVIDER_API_KEY_ENV[p])]
    missing = [p for p in wanted if p not in have]
    return have, missing


def load_capped_models(providers: list[str]) -> list[tuple[str, str, int]]:
    """Return (provider, model_key, configured_max_tokens) for every capped model."""
    config = yaml.safe_load(CONFIG_PATH.read_text())
    out: list[tuple[str, str, int]] = []
    for provider in providers:
        for model, attrs in config[provider]["models"].items():
            out.append((provider, model, attrs.get("max_tokens") or 0))
    return out


def extract_ceiling(message: str) -> int | None:
    """Pull the vendor's stated ceiling out of a rejection message."""
    for pattern in CEILING_PATTERNS:
        match = pattern.search(message)
        if match:
            value = int(match.group(1))
            # Guard against matching the absurd value we sent back to ourselves.
            if value != ABSURD_MAX_TOKENS:
                return value
    return None


async def _attempt(llm, streaming: bool) -> str | None:
    """Issue one over-large request; return the vendor's message, or None if accepted."""
    try:
        if streaming:
            await llm.get_response_stream("hi", max_tokens=ABSURD_MAX_TOKENS)
        else:
            await llm.get_response("hi", max_tokens=ABSURD_MAX_TOKENS)
    except Exception as e:  # noqa: BLE001 — every vendor raises its own type
        return str(getattr(e, "original_error", None) or e)
    return None


async def probe(provider: str, model: str, configured: int) -> Result:
    """Send an over-large cap and read the ceiling out of the rejection."""
    try:
        llm = get_llm_instance(provider, model)
    except MajordomoError as e:
        return Result(provider, model, configured, None, f"cannot instantiate: {e}")

    # Non-streaming first, since most vendors answer there. Anthropic does not:
    # its SDK, and now our own guard, refuse an over-large cap before the request
    # is sent. Either way, retry through streaming, which has no such limit and
    # lets the API itself name its ceiling.
    message = await _attempt(llm, streaming=False)
    if message and extract_ceiling(message) is None:
        message = await _attempt(llm, streaming=True) or message

    if message is None:
        return Result(
            provider, model, configured, None,
            "accepted an absurd cap — the model may clamp silently rather than reject",
        )

    ceiling = extract_ceiling(message)
    detail = message if ceiling is None else f"vendor reports {ceiling}"
    return Result(provider, model, configured, ceiling, detail[:180])


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=CAPPED_PROVIDERS, help="probe one provider only")
    args = parser.parse_args()

    have, missing = runnable_providers(args.provider)
    for provider in missing:
        print(f"[skip] {provider}: {PROVIDER_API_KEY_ENV[provider]} not set", file=sys.stderr)
    if not have:
        print(
            "\nNo capped provider has credentials. Set the keys above in the "
            "environment or a local .env, then re-run.",
            file=sys.stderr,
        )
        return 2

    models = load_capped_models(have)
    print(f"\n{DIVIDER}")
    print(f"Probing {len(models)} capped models with max_tokens={ABSURD_MAX_TOKENS:,}")
    print(f"Providers: {', '.join(have)}")
    print(DIVIDER)

    results = [await probe(*entry) for entry in models]

    for r in results:
        marker = {"OK": "✓", "INHERITS": "·", "LOW": "✗", "MISMATCH": "✗", "UNKNOWN": "?"}[
            r.status
        ]
        line = f"{marker} {r.provider}/{r.model}"
        if r.status in ("MISMATCH", "LOW"):
            line += f"  pinned={r.configured or 'none':} vendor={r.reported:,}"
        elif r.status == "OK":
            line += f"  pinned {r.configured:,}"
        elif r.status == "INHERITS":
            line += f"  vendor {r.reported:,} — inherits the defaults"
        print(line)
        if r.status == "UNKNOWN":
            print(f"    {r.detail}")

    mismatches = [r for r in results if r.status in ("MISMATCH", "LOW")]
    unknown = [r for r in results if r.status == "UNKNOWN"]

    print(f"\n{DIVIDER}")
    print(f"{len(results) - len(mismatches) - len(unknown)} correct, "
          f"{len(mismatches)} need attention, {len(unknown)} unresolved")
    if mismatches:
        print("\nA model must pin max_tokens only when its ceiling is below the "
              f"{DEFAULT_STREAM_MAX_TOKENS} streaming default; pinning a higher\n"
              "value makes it the per-request default and breaks non-streaming calls.")
    print()
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
