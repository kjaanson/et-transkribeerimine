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


@dataclass(slots=True)
class PipelineConfig:
    model_id: str = "TalTechNLP/whisper-large-et"
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    language: str = "et"
    device_preference: str = "mps"
    chunk_length_s: int = 15
    batch_size: int = 1
    format_txt: bool = True
    format_json: bool = True
    format_srt: bool = True
    recurse: bool = False
    extensions: set[str] = field(default_factory=lambda: set(SUPPORTED_EXTENSIONS))
