"""Image understanding and generation examples.

Examples:
    uv run python examples/image_demo.py understand photo.jpg --provider anthropic
    uv run python examples/image_demo.py generate "A watercolor lighthouse" output.jpg
"""

import argparse
import asyncio
import mimetypes
from pathlib import Path

from majordomo_llm import ImageInput, get_image_instance, get_llm_instance

UNDERSTANDING_MODELS = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-5.6-terra",
    "gemini": "gemini-2.5-flash",
}


def load_image(path: Path) -> ImageInput:
    media_type, _ = mimetypes.guess_type(path.name)
    if media_type is None:
        raise ValueError(f"Could not determine image type from {path}")
    return ImageInput(data=path.read_bytes(), media_type=media_type)


async def understand(args: argparse.Namespace) -> None:
    llm = get_llm_instance(args.provider, UNDERSTANDING_MODELS[args.provider])
    response = await llm.get_response(args.prompt, images=(load_image(args.input),))
    print(response.content)
    print(f"Cost: ${response.total_cost:.6f}")


async def generate(args: argparse.Namespace) -> None:
    model_name = "gpt-image-2" if args.provider == "openai" else "gemini-3.1-flash-image"
    model = get_image_instance(args.provider, model_name)
    response = await model.generate(
        args.prompt,
        aspect_ratio=args.aspect_ratio,
        image_size=args.image_size,
        output_format=args.output_format,
    )
    args.output.write_bytes(response.images[0].data)
    print(f"Wrote {args.output} ({response.images[0].media_type})")
    print(f"Cost: ${response.total_cost:.6f}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    vision = commands.add_parser("understand")
    vision.add_argument("input", type=Path)
    vision.add_argument("--provider", choices=UNDERSTANDING_MODELS, default="anthropic")
    vision.add_argument("--prompt", default="Describe the important details in this image.")
    vision.set_defaults(handler=understand)

    image = commands.add_parser("generate")
    image.add_argument("prompt")
    image.add_argument("output", type=Path)
    image.add_argument("--provider", choices=("openai", "gemini"), default="openai")
    image.add_argument("--aspect-ratio", default="1:1")
    image.add_argument("--image-size", default="1K")
    image.add_argument("--output-format", choices=("png", "jpeg", "webp"), default="jpeg")
    image.set_defaults(handler=generate)
    return root


async def main() -> None:
    args = parser().parse_args()
    await args.handler(args)


if __name__ == "__main__":
    asyncio.run(main())
