"""HuggingFace Space entry point for et-transkribeerimine.

The transcriber package lives in src/ (src-layout).  Adding src/ to sys.path
here means HF Spaces can find it without any code copying or packaging step,
while the existing CLI entry points defined in pyproject.toml remain untouched.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# src-layout: make the transcriber package importable in the HF Space runtime.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch  # noqa: E402  (must come after sys.path patch)
import gradio as gr  # noqa: E402

from transcriber.config import PipelineConfig  # noqa: E402
from transcriber.outputs import write_json, write_srt  # noqa: E402
from transcriber.pipeline import TranscriptionPipeline  # noqa: E402

MODEL_ID = "TalTechNLP/whisper-large-et"
SUPPORTED_FORMATS = "MP3, MP4, M4A, WAV, FLAC, OGG, AAC, WMA"

# Singleton pipeline – loaded lazily on first transcription request.
_pipeline: TranscriptionPipeline | None = None


def _get_pipeline() -> TranscriptionPipeline:
    global _pipeline
    if _pipeline is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config = PipelineConfig(model_id=MODEL_ID, device_preference=device)
        _pipeline = TranscriptionPipeline(config)
    return _pipeline


def transcribe(uploaded_file: object | None) -> tuple[str, list[list[str]], str | None, str | None]:
    if not uploaded_file:
        return "", [], None, None

    # gr.File yields a NamedString / tempfile wrapper — get the filesystem path.
    file_path = uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name

    pipe = _get_pipeline()
    result = pipe.transcribe_file(Path(file_path))

    # Segments table rows
    rows = [
        [f"{s['start']:.2f}", f"{s['end']:.2f}", s["text"]]
        for s in result.segments
    ]

    # Write downloadable output files to a temp directory
    tmp = Path(tempfile.mkdtemp())
    stem = Path(file_path).stem
    srt_path = tmp / f"{stem}.srt"
    json_path = tmp / f"{stem}.json"
    write_srt(result, srt_path)
    write_json(result, json_path)

    return result.text, rows, str(srt_path), str(json_path)


# ── UI ──────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Estonian Transcription") as demo:
    gr.Markdown(
        "# 🎙️ Estonian Audio Transcription\n"
        "Upload an audio or video file. The app uses "
        "[TalTechNLP/whisper-large-et](https://huggingface.co/TalTechNLP/whisper-large-et) "
        "to transcribe Estonian speech.\n\n"
        f"**Supported formats:** {SUPPORTED_FORMATS}  \n"
        "The model is downloaded on first use (~3 GB) and cached afterwards. "
        "Transcription on CPU is slower than real-time for long recordings."
    )

    audio_input = gr.File(
        label="Input audio / video",
        file_types=["audio", ".mp4", ".m4a", ".wma", ".aac"],
    )
    run_btn = gr.Button("Transcribe", variant="primary")

    transcript_box = gr.Textbox(
        label="Full transcript",
        lines=10,
        interactive=False,
        placeholder="Transcript will appear here after processing…",
    )

    segments_table = gr.Dataframe(
        headers=["Start (s)", "End (s)", "Text"],
        datatype=["str", "str", "str"],
        label="Segments with timestamps",
        wrap=True,
        interactive=False,
    )

    with gr.Row():
        srt_file = gr.File(label="Download SRT")
        json_file = gr.File(label="Download JSON")

    run_btn.click(
        fn=transcribe,
        inputs=[audio_input],
        outputs=[transcript_box, segments_table, srt_file, json_file],
    )

if __name__ == "__main__":
    demo.launch()
