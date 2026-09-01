# Images

majordomo-llm treats image understanding and image generation as two different
capabilities.

## Image understanding

Image understanding changes the input while retaining the normal text, JSON,
structured-output, streaming, retry, cascade, and cost contracts.

```python
from pathlib import Path

from majordomo_llm import ImageInput, get_llm_instance

llm = get_llm_instance("gemini", "gemini-2.5-flash")
response = await llm.get_response(
    "What objects are visible?",
    images=(ImageInput(Path("photo.webp").read_bytes(), "image/webp"),),
)
```

Anthropic, OpenAI, and Gemini support image inputs. A model without that
capability raises `InputModalityUnsupported` before network I/O. In a cascade,
an incompatible member is skipped and the next compatible member is attempted.

Only in-memory bytes are accepted. The library does not fetch image URLs. Valid
media types are `image/jpeg`, `image/png`, `image/gif`, and `image/webp`.

Hooks continue to inspect and modify the text prompt only. Request logging stores
image MIME type, byte length, and SHA-256 but never raw bytes.

## Image generation and editing

Generated image bytes use a separate interface because output and pricing differ
from `LLMResponse`.

```python
from pathlib import Path

from majordomo_llm import get_image_instance

model = get_image_instance("gemini", "gemini-3.1-flash-image")
response = await model.generate(
    "A clean editorial illustration of a coastal observatory",
    aspect_ratio="16:9",
    image_size="2K",
)
Path("observatory.jpg").write_bytes(response.images[0].data)
```

`ImageResponse` includes generated images, modality-specific token usage, input
and output costs, latency, provider, and model. Provider-specific unsupported
options fail explicitly rather than being silently ignored.

OpenAI supports masks and multiple outputs. Gemini uses conversational reference
images, supports one JPEG output per request, and does not expose masks, background,
or quality controls through this interface.

## Generation failover

`ImageCascade` accepts image provider/model pairs in priority order and implements
the same `generate()` and `edit()` interface as `ImageModel`. A child completes its
own retry policy before the cascade advances. Provider failures, invalid provider
responses, and `ImageOptionUnsupported` capability mismatches trigger failover.
Caller validation errors and broad configuration errors propagate immediately.

The returned `ImageResponse.provider` and `.model` identify the child that actually
succeeded, not the cascade or its primary model.

## Generation logging

`LoggingImageModel` wraps any `ImageModel`, including `ImageCascade`, and writes
cost, latency, aggregate token usage, status, and errors through the existing
logging adapters without delaying the provider response. On cascade success, the
database row identifies the child provider and model that returned the image.

Optional request/response body storage contains modality-specific usage and image
metadata (MIME type, byte length, SHA-256, and revised prompt). Raw reference,
mask, and generated image bytes are deliberately never duplicated into logs.

## Generation hooks

Image generation uses a typed `ImageHookPipeline` rather than encoding binary
responses into the text-only `HookPipeline`. Before hooks receive an immutable
`ImageHookRequest`; after hooks receive the request plus `ImageResponse`. Outcomes
can pass, warn, block, replace the request or response, or request the next cascade
provider. Hook exceptions are recorded as pass verdicts so faulty policy code does
not break provider calls, matching the text-hook contract.

Three provider-neutral hooks are included:

- `ImagePromptRegexHook` blocks, warns, or redacts prompt matches before generation.
- `ImageRequestLimitsHook` enforces output count, reference count, total input bytes,
  and an optional size allowlist before generation incurs cost.
- `ImageIntegrityHook` uses Pillow to decode inputs and outputs, verifies that MIME
  declarations match decoded formats, and enforces a pixel ceiling. Invalid inputs
  block; invalid provider outputs request cascade failover.

`caller_metadata` is available in both hook phases. A cascade runs its before hooks
once, then runs after hooks against each successful provider response until one
passes or the cascade is exhausted. Ordinary hook blocks never trigger failover.
