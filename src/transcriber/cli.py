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
from .io_audio import discover_media_files
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


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
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
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        language=args.language,
        device_preference=args.device,
        chunk_length_s=args.chunk_length,
        batch_size=args.batch_size,
        format_txt=not args.no_txt,
        format_json=not args.no_json,
        format_srt=not args.no_srt,
        format_xlsx=not args.no_xlsx,
    )


def single_main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe one media file with an Estonian Whisper model",
        parents=[_base_parser()],
    )
    parser.add_argument("input_file", type=Path)
    args = parser.parse_args()

    config = _config_from_args(args, parser)
    pipeline = TranscriptionPipeline(config)
    pipeline.preflight_check()

    result = pipeline.transcribe_file(args.input_file)
    written = write_outputs(
        result=result,
        output_dir=config.output_dir,
        write_txt_file=config.format_txt,
        write_json_file=config.format_json,
        write_srt_file=config.format_srt,
        write_xlsx_file=config.format_xlsx,
    )
    logger.info("Transcribed: %s", args.input_file)
    for path in written:
        logger.info("Wrote: %s", path)


def batch_main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch transcribe media files from data directory",
        parents=[_base_parser()],
    )
    parser.add_argument("--recurse", action="store_true", help="Scan directories recursively")
    args = parser.parse_args()

    config = _config_from_args(args, parser)
    config.recurse = args.recurse

    pipeline = TranscriptionPipeline(config)
    pipeline.preflight_check()

    media_files = discover_media_files(config.data_dir, recurse=config.recurse)
    if not media_files:
        logger.warning("No supported media files found in %s", config.data_dir)
        return

    failures: list[dict[str, str]] = []
    successes: list[dict[str, str]] = []

    for media_path in tqdm(media_files, desc="Transcribing", unit="file"):
        try:
            result = pipeline.transcribe_file(media_path)
            written = write_outputs(
                result=result,
                output_dir=config.output_dir,
                write_txt_file=config.format_txt,
                write_json_file=config.format_json,
                write_srt_file=config.format_srt,
            )
            successes.append(
                {
                    "input": str(media_path),
                    "outputs": [str(p) for p in written],
                    "device": result.device,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed transcription for %s: %s", media_path, exc)
            failures.append({"input": str(media_path), "error": str(exc)})

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

    logger.info("Batch finished. Success: %s | Failed: %s", len(successes), len(failures))
    logger.info("Summary written to %s", summary_path)
