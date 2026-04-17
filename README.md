---
title: Estonian Audio Transcription
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.12.0"
app_file: app.py
pinned: false
license: mit
---

# ET-transkribeerimine

Transcribe Estonian audio/video files with Hugging Face model `TalTechNLP/whisper-large-et`.

## Project Structure

```text
et-transkribeerimine/
├── data/                      # Input media files
├── output/                    # Generated transcripts (txt/json/srt)
├── scripts/
│   ├── transcribe.py          # Single-file CLI wrapper
│   └── batch_transcribe.py    # Batch CLI wrapper
├── src/transcriber/
│   ├── cli.py                 # Main CLI commands
│   ├── config.py              # Pipeline configuration
│   ├── devices.py             # MPS/CPU/CUDA device resolution
│   ├── io_audio.py            # Media discovery helpers
│   ├── outputs.py             # TXT/JSON/SRT writers
│   └── pipeline.py            # Whisper pipeline implementation
└── pyproject.toml
```

## Setup (uv)

1. Install system dependency:

```bash
brew install ffmpeg
```

2. Sync Python dependencies:

```bash
uv sync
```

This creates `.venv` and installs all dependencies from `pyproject.toml`.

## Usage

### Batch transcription (default workflow)

```bash
uv run transcribe-batch --data-dir data --output-dir output
```

Options:
- `--device mps|cpu|cuda` (default: `mps`)
- `--language et` (default: `et`)
- `--chunk-length 30`
- `--batch-size 1`
- `--recurse` to scan nested folders
- `--no-txt`, `--no-json`, `--no-srt` to disable formats

### Single file transcription

```bash
uv run transcribe-file data/9b565ae3-d95c-47a4-82c7-d3e942db9549.mp4 --output-dir output
```

## Outputs

For each input file, the pipeline writes:
- `<stem>.txt` plain transcript
- `<stem>.json` structured transcript with metadata and segments
- `<stem>.srt` subtitles

Batch runs also write:
- `output/run_summary.json` with success/failure report

## Device behavior

- Default device preference is `mps` for Apple Silicon speed.
- If MPS fails at runtime for an input, the pipeline retries that transcription on CPU automatically.
- You can force CPU with `--device cpu`.

## Notes

- First run downloads the model weights from Hugging Face, which can take time.
- Ensure your terminal has internet access for the initial download.
