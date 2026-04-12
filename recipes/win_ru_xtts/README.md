# win_ru_xtts

This recipe is a simplified launcher for a Russian XTTS workflow.

It assumes:

- a compatible XTTS polling server is reachable at `--tts-base-url`
- the normalizer endpoint is provided through arguments or environment variables

Minimal usage:

```bash
python recipes/win_ru_xtts/run_book.py "/path/to/book.epub" --language ru-RU
```

Useful environment variables:

- `NORMALIZER_OPENAI_API_KEY`
- `NORMALIZER_OPENAI_BASE_URL`

Useful optional arguments:

- `--tts-base-url http://192.168.1.50:8020`
- `--voice-name reference_long`
- `--normalize-system-prompt-file /path/to/ru_prompt.txt`
- `--prepare-text`
- `--prepared-text-folder /path/to/reviewed_text`

Typical two-step review flow:

1. `--prepare-text` to generate chapter `.txt` files for review.
2. Edit those files.
3. Run again with `--prepared-text-folder` to use the reviewed files as the TTS source.

By default, `--prepared-text-folder` skips the normalizer chain on those reviewed files. Add `--normalize-reviewed-text` only if you explicitly want one more normalization pass.

This recipe is intentionally more general than a personal local wrapper. A personal wrapper can call it with machine-specific defaults.
