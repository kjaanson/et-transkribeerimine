from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm

from .config import (
    DEFAULT_MODEL_ALIAS,
    MODEL_REGISTRY,
    PipelineConfig,
    resolve_model_id,
)
from .io_audio import discover_media_files, is_supported_media
from .outputs import write_outputs
from .pipeline import TranscriptionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class _ListModelsAction(argparse.Action):
    def __init__(self, option_strings, dest=argparse.SUPPRESS, **kwargs):
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=0,
            default=argparse.SUPPRESS,
            **kwargs,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        width = max(len(alias) for alias in MODEL_REGISTRY)
        print("Available model aliases (use with --model):")
        for alias, hf_id in MODEL_REGISTRY.items():
            marker = " (default)" if alias == DEFAULT_MODEL_ALIAS else ""
            print(f"  {alias:<{width}}  {hf_id}{marker}")
        parser.exit(0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe one or more media files (or directories of files) "
            "with an Estonian Whisper model."
        ),
    )
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="Media files and/or directories to transcribe.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ALIAS,
        help=(
            "Model alias from the registry "
            f"({', '.join(MODEL_REGISTRY)}). Default: %(default)s. "
            "Use --list-models for details."
        ),
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Full Hugging Face model id (e.g. 'org/repo'). Overrides --model.",
    )
    parser.add_argument(
        "--list-models",
        action=_ListModelsAction,
        help="List available model aliases and exit.",
    )
    parser.add_argument("--language", default="et")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    parser.add_argument("--chunk-length", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--recurse",
        action="store_true",
        help="When an input is a directory, scan it recursively.",
    )
    parser.add_argument("--no-txt", action="store_true")
    parser.add_argument("--no-json", action="store_true")
    parser.add_argument("--no-srt", action="store_true")
    parser.add_argument("--no-xlsx", action="store_true")
    return parser


def _config_from_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> PipelineConfig:
    requested = args.model_id if args.model_id else args.model
    try:
        model_id = resolve_model_id(requested)
    except ValueError as exc:
        parser.error(str(exc))
    return PipelineConfig(
        model_id=model_id,
        output_dir=args.output_dir,
        language=args.language,
        device_preference=args.device,
        chunk_length_s=args.chunk_length,
        batch_size=args.batch_size,
        recurse=args.recurse,
        format_txt=not args.no_txt,
        format_json=not args.no_json,
        format_srt=not args.no_srt,
        format_xlsx=not args.no_xlsx,
    )


def _expand_inputs(
    inputs: list[Path], recurse: bool, parser: argparse.ArgumentParser
) -> list[Path]:
    media: list[Path] = []
    seen: set[Path] = set()
    for item in inputs:
        if item.is_dir():
            for path in discover_media_files(item, recurse=recurse):
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    media.append(path)
        elif item.is_file():
            if not is_supported_media(item):
                logger.warning("Skipping unsupported file extension: %s", item)
                continue
            resolved = item.resolve()
            if resolved not in seen:
                seen.add(resolved)
                media.append(item)
        else:
            parser.error(f"Input does not exist: {item}")
    return media


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = _config_from_args(args, parser)

    media_files = _expand_inputs(args.inputs, recurse=config.recurse, parser=parser)
    if not media_files:
        logger.warning("No supported media files found in inputs: %s", args.inputs)
        return

    pipeline = TranscriptionPipeline(config)
    pipeline.preflight_check()

    failures: list[dict[str, str]] = []
    successes: list[dict[str, object]] = []

    iterator = (
        tqdm(media_files, desc="Transcribing", unit="file")
        if len(media_files) > 1
        else media_files
    )

    for media_path in iterator:
        try:
            result = pipeline.transcribe_file(media_path)
            written = write_outputs(
                result=result,
                output_dir=config.output_dir,
                write_txt_file=config.format_txt,
                write_json_file=config.format_json,
                write_srt_file=config.format_srt,
                write_xlsx_file=config.format_xlsx,
            )
            successes.append(
                {
                    "input": str(media_path),
                    "outputs": [str(p) for p in written],
                    "device": result.device,
                }
            )
            logger.info("Transcribed: %s", media_path)
            for path in written:
                logger.info("Wrote: %s", path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed transcription for %s: %s", media_path, exc)
            failures.append({"input": str(media_path), "error": str(exc)})

    if len(media_files) > 1:
        summary_path = config.output_dir / "run_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "model_id": config.model_id,
                    "language": config.language,
                    "success_count": len(successes),
                    "failure_count": len(failures),
                    "successes": successes,
                    "failures": failures,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(
            "Finished. Success: %s | Failed: %s", len(successes), len(failures)
        )
        logger.info("Summary written to %s", summary_path)
