# EPUB to Audiobook Converter

This repository is a fork of [p0n1/epub_to_audiobook](https://github.com/p0n1/epub_to_audiobook).

The original project converts EPUB ebooks into audiobooks with multiple TTS backends.

This fork keeps that base and adds work aimed at a more practical self-hosted workflow:

- optional text normalization before TTS
- optional polling-based TTS flow for OpenAI-compatible job APIs
- optional `m4b` packaging with `ffmpeg`

The goal of this fork is still simple:

`EPUB -> audiobook`

Default behavior is intended to remain close to the original project unless the new options are explicitly enabled.

Recommended workflow for preprocessing and review:

1. `prepare_text`:
   Parse the EPUB into chapters, optionally normalize each chapter, and write per-chapter UTF-8 `.txt` files for manual review.
2. Review:
   Open the generated chapter `.txt` files, edit them as needed, and save the reviewed text.
3. TTS:
   Run the converter again with `--prepared_text_folder` to synthesize audio from the reviewed chapter files instead of re-normalizing the raw EPUB text.

This keeps chapter splitting and packaging from the tool, but moves quality control to a clean review step before audio generation.

Verified self-hosted TTS integration:

- [alboro/xtts-win-jobs](https://github.com/alboro/xtts-win-jobs) - the Windows-first XTTS v2 server/CLI this fork is currently tested against.

## Recipes

Reusable launch recipes live under `recipes/`.

Current Russian XTTS recipe:

```bash
python recipes/win_ru_xtts/run_book.py "/path/to/book.epub" --language ru-RU
```

Useful options:

- `--tts-base-url http://127.0.0.1:8020`
- `--voice-name reference_long`
- `--normalize-system-prompt-file /path/to/ru_prompt.txt`
- `--prepare-text`

Personal machine-specific wrappers should live outside the repo or under an ignored folder such as `.local/`.

Normalizer prompts can be customized per run from text files:

- `--normalize_system_prompt_file path.txt`
- `--normalize_user_prompt_file path.txt`
- `--normalize_steps simple_symbols,numbers_ru,tts_safe_split,llm`

The user prompt template supports these placeholders:

- `{chapter_title}`
- `{text}`

Built-in normalizer steps:

- `simple_symbols`
  Replaces risky typography with simpler ASCII variants:
  `«` -> `"`
  `»` -> `"`
  `—` -> `-`
  `…` -> `...`
- `numbers_ru`
  Deterministically expands many Russian number forms before TTS:
  plain integers, `№5`, ranges, common ordinal abbreviations like `17-й`,
  and ordinal-noun combinations like `XVII век` or `17 век`.
- `tts_safe_split`
  Deterministically rewrites overlong sentences into shorter ones before TTS.
  This is designed for XTTS-style models that become unstable on long Russian sentences.
  Use `--normalize_tts_safe_max_chars 160` to tune the target sentence length.
- `llm`
  Runs the OpenAI-compatible LLM normalizer.

Recommended chain for Russian XTTS:

- `simple_symbols,numbers_ru,llm,simple_symbols,tts_safe_split`

This lets deterministic steps handle typography, obvious numerals, and sentence length,
while the LLM focuses on the harder context-dependent rewrites.
