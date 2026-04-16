# EPUB/FB2 to Audiobook Converter

A fork of [p0n1/epub_to_audiobook](https://github.com/p0n1/epub_to_audiobook).

**Mission:** Practical self-hosted audiobook production with focus on quality.
This fork extends the original toward a more complete workflow — richer input format support,
more TTS backend choices, and a sophisticated Russian text normalization pipeline tuned for
high-quality synthesis with XTTS and other self-hosted models.

Interesting finds from across the fork ecosystem are consolidated here rather than left scattered.
Useful improvements will be proposed back to upstream as pull requests over time.

**Supported input formats:** EPUB, FB2.

---

## What's new in this fork

### TTS Providers

In addition to the original Azure / Edge / OpenAI / Piper backends:

| Provider | Origin | Notes |
|---|---|---|
| **Qwen3** (`--tts qwen`) | [7enChan/reson](https://github.com/7enChan/reson) | Aliyun DashScope API. Supports Russian. Optional dep: `dashscope`. |
| **Gemini** (`--tts gemini`) | [7enChan/reson](https://github.com/7enChan/reson) | Google GenAI SDK. Multi-speaker map support. Optional dep: `google-genai`. |
| **Kokoro** (`--tts kokoro`) | [kroryan/epub_to_audiobook](https://github.com/kroryan/epub_to_audiobook) | [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) backend with voice mixing. |

### FB2 Input Support

Native parsing of FB2 fiction format — sections, poems, footnotes, metadata.

### Russian Text Normalization Pipeline

A pipeline of composable normalizer steps (see [Normalizer steps](#normalizer-steps) below).
Numbers, initials, abbreviations, stress disambiguation via LLM, proper nouns, safe sentence
splitting, and more.

Stress data is sourced from the bundled `tsnorm` dictionary and optionally from
[gramdict/zalizniak-2010](https://github.com/gramdict/zalizniak-2010) (~110k lemmas, CC BY-NC),
cached locally in SQLite.

### Structured Pipeline Modes

| Mode | What it does |
|---|---|
| `prepare` | Parse + normalize → write per-chapter `.txt` for review |
| `audio` | Synthesize from reviewed `.txt` or raw book text |
| `package` | Package existing audio files into a single `.m4b` |
| `all` | Full pipeline: normalize + synthesize + package |

### INI Config System

Settings are loaded in order (later sources override earlier ones):
1. Global user config: `~/.config/epub_to_audiobook/config.ini`
2. Project-local: `config.local.ini` (next to `main.py`, gitignored)
3. Per-book: `<book dir>/<book stem>.ini`
4. Explicit `--config PATH`
5. CLI arguments (always win)

### Chunked Audio Resume

`--chunked_audio` enables sentence-level synthesis with SQLite resume.
Each sentence is synthesised independently; already-synthesised chunks are
reused on reruns. Changed sentences are re-synthesised automatically.

---

## Quick Start

**macOS setup:**
```bash
bash recipes/macos/setup_macos.sh
```

See [docs/macos.md](docs/macos.md) for detailed setup notes.

**Run (with INI config for your setup):**
```bash
# Generate full audiobook
.venv/bin/python main.py "/path/to/MyBook.epub" --mode all

# Prepare chapter text for review first
.venv/bin/python main.py "/path/to/MyBook.epub" --mode prepare

# Synthesize from reviewed text
.venv/bin/python main.py "/path/to/MyBook.epub" --mode audio

# Package existing audio to m4b
.venv/bin/python main.py "/path/to/MyBook.epub" --mode package
```

---

## Normalizer Steps

Configure via `normalize_steps` in INI or `--normalize_steps` CLI flag.

| Step | Description |
|---|---|
| `simple_symbols` | Replaces `«»""—…` with simpler ASCII variants |
| `remove_endnotes` | Removes inline footnote numbers after words/punctuation |
| `remove_reference_numbers` | Removes bracketed references like `[3]` or `[12.1]` |
| `ru_initials` | Rewrites Russian initials into XTTS-friendly spoken forms |
| `ru_abbreviations` | Expands common Russian abbreviations |
| `tts_pronunciation_overrides` | Applies TTS-specific overrides for known problem words |
| `ru_stress_words` | Adds stress marks for curated list of frequently mispronounced words |
| `ru_llm_stress_ambiguity` | Sends only true homographs to LLM for contextual stress disambiguation |
| `ru_proper_nouns` | Adds stress marks to likely proper nouns using tsnorm |
| `ru_llm_proper_nouns` | Asks LLM to choose TTS-safe pronunciation for proper names |
| `ru_tsnorm` | Broader Russian stress + `ё` restoration via tsnorm backend |
| `tts_safe_split` | Splits overlong sentences to fit TTS model limits |
| `ru_numbers` | Expands numbers to words (`17-й` → `семнадцатый`, `№5` → `номер пять`) |
| `openai` | Full-text rewrite via OpenAI-compatible LLM |

**Recommended chain for Russian XTTS:**
```
simple_symbols,ru_initials,ru_numbers,ru_llm_stress_ambiguity,ru_llm_proper_nouns,tts_safe_split
```

---

## Recipes

Reusable launch configurations:

- `recipes/win_ru_xtts/` — Windows + Russian XTTS v2 via local xtts-api-server
- `recipes/macos/setup_macos.sh` — macOS environment setup

See `recipes/win_ru_xtts/config.ini.example` for a complete working INI example.

---

## Verified Self-Hosted TTS

- [alboro/xtts-win-jobs](https://github.com/alboro/xtts-win-jobs) — Windows-first OpenAI-compatible XTTS v2 server this fork is tested against.

---

## Output File Structure

```
MyBook.epub
MyBook/
├── logs/
├── text/001/           ← prepare mode output
│   ├── 0001_Chapter.txt
│   └── _state/normalization_progress.sqlite3
├── wav/001/            ← audio mode output
│   ├── 0001_Chapter.wav
│   └── chunks/0001_Chapter/  ← chunked_audio mode
└── MyBook.m4b          ← package mode output
```

---

## Web UI

```bash
.venv/bin/python main_ui.py
```

Opens a Gradio interface at `http://localhost:7860`.
