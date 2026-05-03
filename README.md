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

Transcribe Estonian audio/video files with TalTechNLP Whisper models from Hugging Face.

Default model is `TalTechNLP/whisper-large-v3-turbo-et-verbatim-2604`, which produces
verbatim transcripts WITH punctuation and proper capitalization. The model can be
swapped via the `--model` alias flag (see below) or `--model-id` for any HF id.

## Project Structure

```text
et-transkribeerimine/
├── data/                      # Input media files
├── output/                    # Generated transcripts (txt/json/srt/xlsx/docx)
├── scripts/
│   ├── transcribe.py          # Single-file CLI wrapper
│   └── batch_transcribe.py    # Batch CLI wrapper
├── src/transcriber/
│   ├── cli.py                 # Main CLI commands
│   ├── config.py              # Pipeline configuration
│   ├── devices.py             # MPS/CPU/CUDA device resolution
│   ├── io_audio.py            # Media discovery helpers
│   ├── outputs.py             # TXT/JSON/SRT/XLSX/DOCX writers
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

The `transcribe-file` command accepts any number of files, directories, or a
mix. Directories are scanned for supported media (add `--recurse` for nested
folders). When more than one file is processed, a `run_summary.json` is also
written to the output directory.

```bash
# One file
uv run transcribe-file data/clip.mp3 --output-dir output

# Multiple files
uv run transcribe-file data/a.mp3 data/b.wav data/c.m4a --output-dir output

# Whole directory (with recursion)
uv run transcribe-file data --recurse --output-dir output
```

Options:
- `--model {verbatim,verbatim-prev,subs,legacy}` — model alias (default: `verbatim`)
- `--model-id <org/repo>` — full Hugging Face model id; overrides `--model`
- `--list-models` — print the alias table and exit
- `--device mps|cpu|cuda` (default: `mps`)
- `--language et` (default: `et`)
- `--chunk-length 30`
- `--batch-size 1`
- `--recurse` — when an input is a directory, scan it recursively
- `--no-txt`, `--no-json`, `--no-srt`, `--no-xlsx`, `--no-docx` to disable formats

### Model aliases

| Alias           | Hugging Face id                                              | Notes                                                       |
|-----------------|--------------------------------------------------------------|-------------------------------------------------------------|
| `verbatim`      | `TalTechNLP/whisper-large-v3-turbo-et-verbatim-2604`         | Default. Verbatim, punctuated, properly capitalized.        |
| `verbatim-prev` | `TalTechNLP/whisper-large-v3-turbo-et-verbatim`              | Earlier verbatim model, kept for reproducibility.           |
| `subs`          | `TalTechNLP/whisper-large-v3-turbo-et-subs`                  | Subtitle-style: punctuated, but rephrases/compresses speech.|
| `legacy`        | `TalTechNLP/whisper-large-et`                                | Original model. Lowercase, unpunctuated.                    |

## Outputs

For each input file, the pipeline writes:
- `<stem>.txt` plain transcript
- `<stem>.json` structured transcript with metadata and segments
- `<stem>.srt` subtitles
- `<stem>.xlsx` Excel workbook with one row per segment (columns: Segment, Start (s), End (s), Text)
- `<stem>.docx` Word document with one paragraph per segment (text only)

Multi-file runs also write:
- `output/run_summary.json` with success/failure report

## Device behavior

- Default device preference is `mps` for Apple Silicon speed.
- If MPS fails at runtime for an input, the pipeline retries that transcription on CPU automatically.
- You can force CPU with `--device cpu`.

## Notes

- First run downloads the model weights from Hugging Face, which can take time.
- Ensure your terminal has internet access for the initial download.
