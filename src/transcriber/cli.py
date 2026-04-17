from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm

from .config import PipelineConfig
from .io_audio import discover_media_files
from .outputs import write_outputs
from .pipeline import TranscriptionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--model-id", default="TalTechNLP/whisper-large-et")
    parser.add_argument("--language", default="et")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    parser.add_argument("--chunk-length", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-txt", action="store_true")
    parser.add_argument("--no-json", action="store_true")
    parser.add_argument("--no-srt", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        model_id=args.model_id,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        language=args.language,
        device_preference=args.device,
        chunk_length_s=args.chunk_length,
        batch_size=args.batch_size,
        format_txt=not args.no_txt,
        format_json=not args.no_json,
        format_srt=not args.no_srt,
    )


def single_main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe one media file with TalTechNLP Whisper ET",
        parents=[_base_parser()],
    )
    parser.add_argument("input_file", type=Path)
    args = parser.parse_args()

    config = _config_from_args(args)
    pipeline = TranscriptionPipeline(config)
    pipeline.preflight_check()

    result = pipeline.transcribe_file(args.input_file)
    written = write_outputs(
        result=result,
        output_dir=config.output_dir,
        write_txt_file=config.format_txt,
        write_json_file=config.format_json,
        write_srt_file=config.format_srt,
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

    config = _config_from_args(args)
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
