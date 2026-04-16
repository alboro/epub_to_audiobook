# Architecture & Pipeline

## Overview

The pipeline transforms an e-book file into an audiobook through up to four stages:

```
EPUB/FB2
   │
   ▼  [parse]
Book Parser
   │  chapters: [(title, text), ...]
   ▼  [normalize]
Normalizer Chain   ←── SQLite resume state
   │  normalized text per chapter
   ▼  [synthesize]
TTS Provider       ←── chunked audio resume (SQLite)
   │  audio files per chapter / sentence
   ▼  [package]
m4b Packager (ffmpeg)
   │
   ▼
MyBook.m4b
```

---

## Pipeline Modes

| Mode | Stages run |
|---|---|
| `prepare` | parse → normalize → write `.txt` |
| `audio` | parse → (normalize) → TTS → write audio |
| `package` | detect audio files → pack to `.m4b` |
| `all` | parse → normalize → TTS → pack |

---

## Book Parsers

`audiobook_generator/book_parsers/`

| Parser | Format |
|---|---|
| `EpubBookParser` | `.epub` via `ebooklib` + `BeautifulSoup` |
| `Fb2BookParser` | `.fb2` via `lxml` |

Both parsers produce a list of `(title: str, text: str)` chapter tuples.
`chapter_mode` controls grouping: `documents` (one per XHTML/section) or `toc_sections` (grouped by TOC).

---

## Normalizer Chain

`audiobook_generator/normalizers/`

Normalizers are chained via `normalize_steps` config. Each step receives the output of the previous one.

### Step Naming Convention

- `simple_*` — language-agnostic text cleanup
- `tts_*` — TTS-agnostic but TTS-oriented transforms
- `ru_*` — Russian-specific deterministic normalizers  
- `ru_llm_*` — Russian-specific normalizers that call an LLM
- `openai` — generic LLM full-text rewrite

### Resumable Steps

Steps that involve LLM calls support chunked resume via SQLite:
- State is stored in `<output>/<text_run>/_state/normalization_progress.sqlite3`
- On rerun, completed chunks are skipped
- Changed input re-triggers only affected chunks

### Stress Paradox Guard

`ru_tts_stress_paradox_guard.py` — singleton service that tracks words where adding a stress mark
causes the TTS server to *mispronounce* the word. These words are excluded from LLM stress
disambiguation candidates. Currently injected into `ru_llm_stress_ambiguity`.

---

## TTS Providers

`audiobook_generator/tts_providers/`

All providers implement `BaseTTSProvider`. Key methods:
- `text_to_speech(text, output_file, audio_tags)` — synthesize one chapter
- `get_break_string()` — paragraph separator inserted by book parsers
- `get_output_file_extension()` — audio format extension
- `estimate_cost(chars)` — rough cost estimate for prompt

### Chunked Audio

When `chunked_audio = true`, each sentence is synthesised via `ChunkedAudioGenerator` and stored
in a SQLite-backed `AudioChunkStore`. Sentences already in the store are reused without TTS calls.

---

## Config & INI

`audiobook_generator/config/`

- `GeneralConfig` — flat dataclass populated from CLI args or UI form
- `IniConfigManager` — merges multiple INI files and env vars into args namespace
- `UiConfig` — Gradio UI state → `GeneralConfig` bridge

### Config load order

```
~/.config/epub_to_audiobook/config.ini   (global defaults)
<project>/config.local.ini               (machine-specific, gitignored)
<book_dir>/<book_stem>.ini               (per-book overrides)
--config PATH                            (explicit override)
CLI args                                 (always win)
```

---

## Output Layout

```
MyBook/
├── logs/
│   └── EtA_2026-04-16_001.log
├── text/
│   └── 001/                        ← first 'prepare' run
│       ├── 0001_Chapter_One.txt
│       ├── _state/
│       │   └── normalization_progress.sqlite3
│       └── _chapter_artifacts/
│           └── 0001_Chapter_One/
│               └── _normalizer_steps/
│                   ├── 01_ru_initials/
│                   │   ├── input.txt
│                   │   └── output.txt
│                   └── 02_ru_llm_stress_ambiguity/
│                       ├── input.txt
│                       ├── output.txt
│                       └── 00_choice_system_prompt.txt
├── wav/
│   └── 001/                        ← first 'audio' run
│       ├── 0001_Chapter_One.wav
│       └── chunks/
│           └── 0001_Chapter_One/
│               ├── abc123.wav      ← sentence chunk
│               └── def456.wav
└── MyBook.m4b
```

