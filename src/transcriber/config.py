from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".mp3",
    ".mp4",
    ".m4a",
    ".wav",
    ".flac",
    ".ogg",
    ".aac",
    ".wma",
}


# Curated registry of known-good Estonian Whisper checkpoints.  Keys are short
# aliases used by the CLI / UI; values are full Hugging Face model ids.
MODEL_REGISTRY: dict[str, str] = {
    # Verbatim transcripts WITH punctuation and capitalization (current default).
    "verbatim": "TalTechNLP/whisper-large-v3-turbo-et-verbatim-2604",
    # Earlier verbatim model — kept for reproducibility of older runs.
    "verbatim-prev": "TalTechNLP/whisper-large-v3-turbo-et-verbatim",
    # Subtitle-style: punctuated/cased but rephrases & compresses speech.
    "subs": "TalTechNLP/whisper-large-v3-turbo-et-subs",
    # Original model used by this repo: unpunctuated, lowercase output.
    "legacy": "TalTechNLP/whisper-large-et",
}

DEFAULT_MODEL_ALIAS = "verbatim"


def resolve_model_id(value: str) -> str:
    """Resolve an alias from MODEL_REGISTRY or pass through a full HF id."""
    if value in MODEL_REGISTRY:
        return MODEL_REGISTRY[value]
    if "/" in value:
        return value
    raise ValueError(
        f"Unknown model alias '{value}'. Choose from "
        f"{sorted(MODEL_REGISTRY)} or pass a full HF model id like 'org/repo'."
    )


@dataclass(slots=True)
class PipelineConfig:
    model_id: str = MODEL_REGISTRY[DEFAULT_MODEL_ALIAS]
    output_dir: Path = Path("output")
    language: str = "et"
    device_preference: str = "gpu"
    chunk_length_s: int = 15
    batch_size: int = 1
    format_txt: bool = True
    format_json: bool = True
    format_srt: bool = True
    format_xlsx: bool = True
    recurse: bool = False
    extensions: set[str] = field(default_factory=lambda: set(SUPPORTED_EXTENSIONS))
