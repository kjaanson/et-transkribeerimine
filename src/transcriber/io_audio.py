from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .config import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000

# Extensions whose containers are natively readable by soundfile/librosa as-is.
_NATIVE_AUDIO = {".wav", ".flac", ".ogg"}


def is_supported_media(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def discover_media_files(data_dir: Path, recurse: bool = False) -> list[Path]:
    if recurse:
        candidates = sorted(p for p in data_dir.rglob("*") if p.is_file())
    else:
        candidates = sorted(p for p in data_dir.iterdir() if p.is_file())

    return [p for p in candidates if is_supported_media(p)]


def extract_audio_array(media_path: Path) -> tuple[np.ndarray, int]:
    """Convert any supported media file to a mono float32 NumPy array at 16 kHz.

    Uses ffmpeg for container decoding (MP4, M4A, MP3, etc.) so Transformers
    never has to touch the original container format.

    Returns:
        (samples, sample_rate) where samples is float32 in [-1.0, 1.0].
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        cmd = [
            "ffmpeg",
            "-y",                        # overwrite without asking
            "-i", str(media_path),
            "-ac", "1",                  # mono
            "-ar", str(TARGET_SAMPLE_RATE),  # 16 kHz
            "-sample_fmt", "s16",        # 16-bit PCM — standard wav before float conversion
            "-vn",                       # no video
            str(tmp_path),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed for {media_path}:\n"
                + result.stderr.decode(errors="replace")
            )

        import soundfile as sf

        samples, sample_rate = sf.read(str(tmp_path), dtype="float32", always_2d=False)
        logger.debug(
            "Normalized %s → %d samples @ %d Hz",
            media_path.name,
            len(samples),
            sample_rate,
        )
        return samples, sample_rate
    finally:
        tmp_path.unlink(missing_ok=True)
