from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from .config import PipelineConfig
from .devices import recommended_dtype, resolve_device
from .io_audio import extract_audio_array

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TranscriptionResult:
    input_path: Path
    model_id: str
    device: str
    language: str
    text: str
    segments: list[dict[str, Any]]


class TranscriptionPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device_preference)
        self.torch_dtype = recommended_dtype(self.device)
        self._pipe: Any | None = None

    def preflight_check(self) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg is not installed or not on PATH. Install with: brew install ffmpeg"
            )

    def _load_pipeline(self) -> Any:
        if self._pipe is not None:
            return self._pipe

        with tqdm(
            total=100,
            desc="Loading model",
            bar_format="{desc}: {percentage:3.0f}% {bar}",
            disable=False,
        ) as pbar:
            pbar.update(10)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.config.model_id,
                dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            pbar.update(40)
            model.to(self.device)
            pbar.update(20)

            processor = AutoProcessor.from_pretrained(self.config.model_id)
            pbar.update(20)

            # Fine-tuned checkpoints like TalTechNLP/whisper-large-et may ship with
            # generation_config missing no_timestamps_token_id and language mappings
            # required by modern transformers.  Patch what we need and use
            # forced_decoder_ids to set language+task (the native 'language=' kwarg
            # raises ValueError against outdated generation configs).
            no_ts_id: int = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
            model.generation_config.no_timestamps_token_id = no_ts_id
            # Build forced lang+task prefix, excluding the no_timestamps token so that
            # return_timestamps=True can produce segment-level timestamps.
            lang_task_ids = processor.get_decoder_prompt_ids(
                language=self.config.language, task="transcribe"
            )
            forced = [(pos, tok) for pos, tok in lang_task_ids if tok != no_ts_id]
            model.generation_config.forced_decoder_ids = forced

            self._pipe = pipeline(
                task="automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                dtype=self.torch_dtype,
                device=self.device,
            )
            pbar.update(10)

        return self._pipe

    def _run_asr(self, asr: Any, audio_input: dict[str, Any]) -> dict[str, Any]:
        sample_rate: int = audio_input["sampling_rate"]
        total_frames: int = len(audio_input["array"])
        total_seconds: float = total_frames / sample_rate

        pbar = tqdm(
            total=round(total_seconds, 1),
            desc="Transcribing",
            unit="s",
            bar_format="{desc}: {percentage:3.0f}% {bar} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        last_seek: list[int] = [0]

        def monitor_progress(progress_tensor: Any) -> None:  # noqa: ANN401
            # progress_tensor shape: (batch_size, 2) — [:, 0] = seek, [:, 1] = max_frames.
            try:
                current = int(progress_tensor[0][0].item())
            except (IndexError, TypeError, AttributeError):
                return
            elapsed_frames = max(0, current - last_seek[0])
            elapsed_s = round(elapsed_frames / sample_rate, 2)
            pbar.update(elapsed_s)
            last_seek[0] = current

        # Silence noisy transformers log messages that fire during every call:
        # - forced_decoder_ids deprecation (required workaround for outdated fine-tune)
        # - duplicate logits-processor warnings (side-effect of the same workaround)
        # - "did not predict an ending timestamp" (audio content observation, not a bug)
        _hush = [
            logging.getLogger("transformers.models.whisper.generation_whisper"),
            logging.getLogger("transformers.models.whisper.tokenization_whisper"),
            logging.getLogger("transformers.generation.utils"),
        ]
        _old_levels = [lg.level for lg in _hush]
        for lg in _hush:
            lg.setLevel(logging.ERROR)

        try:
            output = asr(
                audio_input,
                return_timestamps=True,
                generate_kwargs={"monitor_progress": monitor_progress},
            )
        finally:
            for lg, lvl in zip(_hush, _old_levels):
                lg.setLevel(lvl)
            pbar.n = round(total_seconds, 1)
            pbar.close()

        return output

    def transcribe_file(self, media_path: Path) -> TranscriptionResult:
        if not media_path.exists():
            raise FileNotFoundError(f"Input file not found: {media_path}")

        with tqdm(
            total=1, desc=f"Extracting: {media_path.name}", bar_format="{desc} {elapsed}s"
        ) as pbar:
            samples, sample_rate = extract_audio_array(media_path)
            pbar.update(1)

        audio_input = {"array": samples, "sampling_rate": sample_rate}

        asr = self._load_pipeline()

        try:
            output = self._run_asr(asr, audio_input)
        except RuntimeError as exc:
            # Some ops on MPS can fail for specific audio lengths. Retry on CPU automatically.
            if self.device == "mps":
                tqdm.write("⚠ MPS failed, retrying on CPU...")
                self.device = "cpu"
                self.torch_dtype = torch.float32
                self._pipe = None
                asr = self._load_pipeline()
                output = self._run_asr(asr, audio_input)
            else:
                raise

        text = str(output.get("text", "")).strip()
        raw_chunks = output.get("chunks") or []

        segments: list[dict[str, Any]] = []
        for chunk in raw_chunks:
            ts = chunk.get("timestamp") or (None, None)
            start = float(ts[0]) if ts[0] is not None else 0.0
            end = float(ts[1]) if ts[1] is not None else start + 0.1
            segments.append(
                {
                    "start": start,
                    "end": max(end, start),
                    "text": str(chunk.get("text", "")).strip(),
                }
            )

        if not segments and text:
            segments = [{"start": 0.0, "end": 5.0, "text": text}]

        return TranscriptionResult(
            input_path=media_path,
            model_id=self.config.model_id,
            device=self.device,
            language=self.config.language,
            text=text,
            segments=segments,
        )
