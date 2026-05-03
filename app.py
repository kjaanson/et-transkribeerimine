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

from transcriber.config import (  # noqa: E402
    DEFAULT_MODEL_ALIAS,
    MODEL_REGISTRY,
    PipelineConfig,
    resolve_model_id,
)
from transcriber.outputs import write_json, write_srt, write_xlsx  # noqa: E402
from transcriber.pipeline import TranscriptionPipeline  # noqa: E402

SUPPORTED_FORMATS = "MP3, MP4, M4A, WAV, FLAC, OGG, AAC, WMA"

# One pipeline cached per model alias so switching back to a previously-used
# model is instant.
_pipelines: dict[str, TranscriptionPipeline] = {}


def _get_pipeline(model_alias: str) -> TranscriptionPipeline:
    if model_alias not in _pipelines:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config = PipelineConfig(
            model_id=resolve_model_id(model_alias),
            device_preference=device,
        )
        _pipelines[model_alias] = TranscriptionPipeline(config)
    return _pipelines[model_alias]


def transcribe(
    uploaded_file: object | None,
    model_alias: str,
) -> tuple[str, list[list[str]], str | None, str | None, str | None]:
    if not uploaded_file:
        return "", [], None, None, None

    # gr.File yields a NamedString / tempfile wrapper — get the filesystem path.
    file_path = uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name

    pipe = _get_pipeline(model_alias)
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
    xlsx_path = tmp / f"{stem}.xlsx"
    write_srt(result, srt_path)
    write_json(result, json_path)
    write_xlsx(result, xlsx_path)

    return result.text, rows, str(srt_path), str(json_path), str(xlsx_path)


# ── UI ──────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Estonian Transcription") as demo:
    gr.Markdown(
        "# 🎙️ Estonian Audio Transcription\n"
        "Upload an audio or video file and pick a TalTechNLP Whisper model "
        "to transcribe Estonian speech.\n\n"
        "**Models:**\n"
        "- `verbatim` — verbatim transcripts WITH punctuation and capitalization (default)\n"
        "- `verbatim-prev` — earlier verbatim model, kept for reproducibility\n"
        "- `subs` — subtitle-style: punctuated and compressed/rephrased\n"
        "- `legacy` — original `whisper-large-et`, lowercase and unpunctuated\n\n"
        f"**Supported formats:** {SUPPORTED_FORMATS}  \n"
        "Each model is downloaded on first use (~1.6 GB for the turbo variants) "
        "and cached afterwards. Transcription on CPU is slower than real-time "
        "for long recordings."
    )

    audio_input = gr.File(
        label="Input audio / video",
        file_types=["audio", ".mp4", ".m4a", ".wma", ".aac"],
    )
    model_dropdown = gr.Dropdown(
        choices=list(MODEL_REGISTRY.keys()),
        value=DEFAULT_MODEL_ALIAS,
        label="Model",
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
        xlsx_file = gr.File(label="Download XLSX")

    run_btn.click(
        fn=transcribe,
        inputs=[audio_input, model_dropdown],
        outputs=[transcript_box, segments_table, srt_file, json_file, xlsx_file],
    )

if __name__ == "__main__":
    demo.launch()
