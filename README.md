# EPUB/FB2 to Audiobook Converter

A fork of [p0n1/epub_to_audiobook](https://github.com/p0n1/epub_to_audiobook).

**Focus:** practical self-hosted audiobook production.
The original project is a solid base; this fork extends it toward a more complete workflow —
richer input format support, more TTS backend choices, and a Russian text normalization pipeline
tuned for high-quality synthesis.

Interesting work found across the fork ecosystem is consolidated here rather than left scattered.
Useful improvements will be proposed back to the upstream project as pull requests over time.

**Supported input formats:** EPUB, FB2.

## Added in this fork

### TTS providers

In addition to the original Azure / Edge / OpenAI / Piper backends:

| Provider | Source | Notes |
|---|---|---|
| **Qwen3** (`--tts qwen`) | [7enChan/reson](https://github.com/7enChan/reson) | Aliyun DashScope API. Supports Russian. Voices: Cherry, Ethan and others. Optional dep: `dashscope`. |
| **Gemini** (`--tts gemini`) | [7enChan/reson](https://github.com/7enChan/reson) | Google GenAI SDK, `gemini-2.5-pro-preview-tts`. Multi-speaker map support. Optional dep: `google-genai`. |
| **Kokoro** (`--tts kokoro`) | [kroryan/epub_to_audiobook](https://github.com/kroryan/epub_to_audiobook) | [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) backend with voice mixing. See `docker-compose.kokoro-example.yml`. |

### Russian text normalization

A pipeline of composable normalizer steps (see [Normalizer steps](#normalizer-steps) below):
numbers, initials, abbreviations, stress disambiguation via LLM, proper nouns, safe sentence splitting, and more.

Stress data is sourced from the bundled `tsnorm` dictionary and optionally from
[gramdict/zalizniak-2010](https://github.com/gramdict/zalizniak-2010) (~110k lemmas, CC BY-NC),
cached locally in SQLite.

### prepare_text workflow

New `prepare_text` workflow for reviewable chapter audio:

1. `prepare_text`:
   Parse the EPUB/FB2 into chapters, optionally normalize each chapter, and write per-chapter UTF-8 `.txt` files for manual review.
2. Review:
   Open the generated chapter `.txt` files, edit them as needed, and save the reviewed text.
3. TTS:
   Run the converter again with `--prepared_text_folder` to synthesize audio from the reviewed chapter files instead of re-normalizing the raw EPUB text.

This keeps chapter splitting and packaging from the tool, but moves quality control to a clean review step before audio generation.

## Quick Start

macOS:

```bash
bash scripts/setup_macos.sh
bash scripts/run_preview_macos.sh
```

More detailed notes and an OpenAI-compatible example live in [docs/macos.md](docs/macos.md).


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
- `--normalize-pronunciation-lexicon-db /path/to/ru_pronunciation_lexicon.sqlite3`

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
- normalization resume state in `_state/normalization_progress.sqlite3`
- preview text files and generated audio files in the output directory itself

For every normalized chapter, the detailed step-by-step artifacts now also include:

- `_chapter_artifacts/<chapter>/_normalizer_steps/<step>/input.txt`
- `_chapter_artifacts/<chapter>/_normalizer_steps/<step>/output.txt`
- prompt/config files for model-backed steps, such as `00_system_prompt.txt`, `01_user_prompt_template.txt`, and per-chunk `01_user_prompt.txt`

This means you can always inspect exactly what was sent to a model for a given chapter and step.

Normalization is now resumable at the step level, and chunked steps such as `llm` are resumable at the chunk level.

If a long LLM normalization run stops halfway through, rerunning the same command with the same output directory will reuse completed steps and completed chunks instead of starting the whole chapter from scratch.

Normalizer prompts can be customized per run from text files:

- `--normalize_system_prompt_file path.txt`
- `--normalize_user_prompt_file path.txt`
- `--normalize_steps simple_symbols,initials_ru,numbers_ru,stress_ambiguity_llm,llm,simple_symbols,initials_ru,proper_nouns_pronunciation_ru,tts_safe_split`

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

Shared pronunciation/stress lexicon settings:

- `--normalize_pronunciation_lexicon_db`

This SQLite database is a reusable local index of TTS-oriented word forms. On the first run that requests it, the project can build a cached database from the packaged `tsnorm` dictionary. The current schema is:

- `surface_form`
- `spoken_form`
- `lemma`
- `pos`
- `grammemes`
- `is_proper_name`
- `source`
- `confidence`

Here `spoken_form` means "the form we want to feed into TTS". For `tsnorm`-derived entries this currently means the same written word with stress marks inserted when the dictionary provides them. That is already useful for contextual stress disambiguation, even before a future phonetic/G2P source is added for cases like `Пейн -> Пэйн`.

Built-in normalizer steps:

- `simple_symbols`
  Replaces risky typography with simpler ASCII variants:
  `«` `»` `"` `"` → backtick `` ` ``
  `—` → `-`
  `…` → `...`
- `initials_ru`
  Rewrites Russian initials before surnames into XTTS-friendlier spoken forms.
  Example: `Е. Д. Калашниковой` -> `Е-Дэ Калашниковой`.
- `tts_pronunciation_overrides`
  Applies deterministic XTTS-specific pronunciation overrides for known problematic Russian words.
  This step is optional and is no longer part of the default recipe chain.
  Use `--normalize_tts_pronunciation_overrides_file path.txt` to add your own `source==replacement` rules.
- `numbers_ru`
  Deterministically expands many Russian number forms before TTS:
  plain integers, `№5`, ranges, common ordinal abbreviations like `17-й`,
  ordinal-noun combinations like `XVII век` or `17 век`,
  and book-structure patterns such as `1 глава` -> `одна глава`.
- `stress_words_ru`
  Adds stress marks only for a curated list of problem words instead of accenting the whole text.
  Built-in examples include `чудес` -> `чуде́с`, `чудеса` -> `чудеса́`.
  Use `--normalize_stress_exceptions_file path.txt` to extend the list.
  This step is kept as a legacy fallback, but the default recipe now prefers `stress_ambiguity_llm`.
- `stress_ambiguity_llm`
  Finds only known ambiguous Russian word forms, builds a small set of valid stress variants,
  and asks an OpenAI-compatible LLM to choose the best option from context.
  This is designed for real homographs such as `беды` or `поступи`, where a global replacement
  would be unsafe but full-text rewriting would be excessive.
  When `--normalize_pronunciation_lexicon_db` is set, this step can also pull ambiguous variants from the shared SQLite lexicon built from `tsnorm`.
  Use `--normalize_stress_ambiguity_file path.txt` to add more entries in the form
  `source==variant1|variant2`, with either combining acute accents or Silero-style plus notation
  like `б+еды|бед+ы`.
- `proper_nouns_ru`
  Uses the existing `tsnorm` accent backend, but only on likely Russian proper nouns written with capital letters inside a sentence.
  This is meant as a compromise between no accents at all and accenting the whole book.
  It is especially useful for surnames, city names, and multi-part place names that XTTS tends to stress badly.
  The current heuristic intentionally skips most sentence-start words to avoid over-accenting ordinary prose.
- `proper_nouns_pronunciation_ru`
  Collects likely proper names and named entities, builds several pronunciation variants,
  and asks an OpenAI-compatible LLM to choose the most TTS-safe option in context.
  This is narrower and safer than a free-form rewrite because the model chooses from explicit options
  instead of rewriting the whole sentence.
  Prompt artifacts and batch decisions are saved in chapter artifacts just like other resumable LLM-backed steps.
- `tts_safe_split`
  Deterministically rewrites overlong sentences into shorter ones before TTS.
  This is designed for XTTS-style models that become unstable on long Russian sentences.
Use `--normalize_tts_safe_max_chars 180` to tune the target sentence length.
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

- `simple_symbols,initials_ru,numbers_ru,stress_ambiguity_llm,llm,simple_symbols,initials_ru,proper_nouns_pronunciation_ru,tts_safe_split`

This lets deterministic steps handle typography, obvious numerals, and sentence length,
while the LLM focuses on the harder context-dependent rewrites. The repeated post-LLM cleanup
steps are intentional: they restore initials, sparse stress marks,
and contextual proper-name pronunciation if the LLM drifts back toward riskier forms.

Optional stronger accent chain:

- `simple_symbols,initials_ru,numbers_ru,tsnorm_ru,llm,simple_symbols,initials_ru,tsnorm_ru,tts_safe_split`

Notes:

- `tsnorm_ru` currently expects a Python `3.10-3.12` environment because the upstream package does not publish wheels for newer Python versions.
- The safest way to use accents with XTTS is still selective: prefer `stress_ambiguity_llm` for real homographs, keep `stress_words_ru` only as a narrow fallback, and only opt into `tsnorm_ru` if that specific book needs broader help.
