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

By default, this recipe now uses `--chapter-mode toc_sections`, so EPUBs with a usable table of contents are grouped by top-level TOC sections instead of one XHTML file per chapter.

Useful options:

- `--tts-base-url http://127.0.0.1:8020`
- `--voice-name reference_long`
- `--normalize-system-prompt-file /path/to/ru_prompt.txt`
- `--prepare-text`
- `--prepared-text-folder /path/to/reviewed_text`

Review-first workflow with reused chapter files:

1. Generate reviewable chapter text:
   `python recipes/win_ru_xtts/run_book.py "/path/to/book.epub" --language ru-RU --prepare-text --output-dir /path/to/run`
2. Edit the generated per-chapter `.txt` files in that output folder.
3. Synthesize from the reviewed text without re-normalizing it:
   `python recipes/win_ru_xtts/run_book.py "/path/to/book.epub" --language ru-RU --prepared-text-folder /path/to/run --output-dir /path/to/final_audio`

If you do want to run the normalizer chain on the reviewed text again, add `--normalize-reviewed-text`.

Personal machine-specific wrappers should live outside the repo or under an ignored folder such as `.local/`.

When you pass an explicit output directory, this fork keeps the whole run together there:

- copied source book in `_source/`
- run logs in `logs/`
- per-chapter text artifacts in `_chapter_artifacts/`
- preview text files and generated audio files in the output directory itself

Normalizer prompts can be customized per run from text files:

- `--normalize_system_prompt_file path.txt`
- `--normalize_user_prompt_file path.txt`
- `--normalize_steps simple_symbols,initials_ru,pronunciation_exceptions_ru,numbers_ru,tts_safe_split,llm`

The user prompt template supports these placeholders:

- `{chapter_title}`
- `{text}`

The default user prompt is now just `{text}`. This avoids leaking wrapper labels like `Chapter:` or `Text:` back into the normalized output.

LLM access is now treated as shared normalizer infrastructure, not as something reserved only for the `llm` step. The current `llm` step uses it directly, and future normalizer steps can reuse the same configured endpoint for narrow disambiguation requests without reimplementing client setup.

Shared LLM settings for any normalizer step that wants them:

- `--normalize_base_url`
- `--normalize_api_key`
- `--normalize_model`
- `--normalize_max_chars`
- `--normalize_system_prompt_file`
- `--normalize_user_prompt_file`

Built-in normalizer steps:

- `simple_symbols`
  Replaces risky typography with simpler ASCII variants:
  `«` -> `"`
  `»` -> `"`
  `—` -> `-`
  `…` -> `...`
- `initials_ru`
  Rewrites Russian initials before surnames into XTTS-friendlier spoken forms.
  Example: `Е. Д. Калашниковой` -> `Е-Дэ Калашниковой`.
- `pronunciation_exceptions_ru`
  Applies deterministic pronunciation overrides for known problematic Russian words.
  Built-in examples include `отель` -> `отэль`.
  Use `--normalize_pronunciation_exceptions_file path.txt` to add your own `source==replacement` rules.
- `numbers_ru`
  Deterministically expands many Russian number forms before TTS:
  plain integers, `№5`, ranges, common ordinal abbreviations like `17-й`,
  ordinal-noun combinations like `XVII век` or `17 век`,
  and book-structure patterns such as `1 глава` -> `одна глава`.
- `stress_words_ru`
  Adds stress marks only for a curated list of problem words instead of accenting the whole text.
  Built-in examples include `беды` -> `беды́`, `чудес` -> `чуде́с`, `чудеса` -> `чудеса́`.
  Use `--normalize_stress_exceptions_file path.txt` to extend the list.
- `proper_nouns_ru`
  Uses the existing `tsnorm` accent backend, but only on likely Russian proper nouns written with capital letters inside a sentence.
  This is meant as a compromise between no accents at all and accenting the whole book.
  It is especially useful for surnames, city names, and multi-part place names that XTTS tends to stress badly.
  The current heuristic intentionally skips most sentence-start words to avoid over-accenting ordinary prose.
- `tts_safe_split`
  Deterministically rewrites overlong sentences into shorter ones before TTS.
  This is designed for XTTS-style models that become unstable on long Russian sentences.
  Use `--normalize_tts_safe_max_chars 160` to tune the target sentence length.
- `tsnorm_ru`
  Runs the optional `tsnorm` backend for broader Russian stress and `ё` restoration.
  This is more aggressive than `stress_words_ru`, so it is best enabled deliberately rather than by default.
  It also needs the spaCy Russian model `ru_core_news_md`, which is now pinned in `requirements.txt`.
  Useful tuning flags:
  `--normalize_tsnorm_stress_yo`
  `--normalize_tsnorm_stress_monosyllabic`
  `--normalize_tsnorm_min_word_length`
- `llm`
  Runs the OpenAI-compatible LLM normalizer.

Recommended chain for Russian XTTS:

- `simple_symbols,initials_ru,pronunciation_exceptions_ru,numbers_ru,stress_words_ru,llm,simple_symbols,initials_ru,pronunciation_exceptions_ru,stress_words_ru,proper_nouns_ru,tts_safe_split`

This lets deterministic steps handle typography, obvious numerals, and sentence length,
while the LLM focuses on the harder context-dependent rewrites. The repeated post-LLM cleanup
steps are intentional: they restore initials, pronunciation overrides, sparse stress marks,
and likely proper-name accents if the LLM drifts back toward riskier forms.

Optional stronger accent chain:

- `simple_symbols,initials_ru,pronunciation_exceptions_ru,numbers_ru,tsnorm_ru,llm,simple_symbols,initials_ru,pronunciation_exceptions_ru,tsnorm_ru,tts_safe_split`

Notes:

- `tsnorm_ru` currently expects a Python `3.10-3.12` environment because the upstream package does not publish wheels for newer Python versions.
- The safest way to use accents with XTTS is still selective: start with `stress_words_ru`, then only opt into `tsnorm_ru` if that specific book needs broader help.
