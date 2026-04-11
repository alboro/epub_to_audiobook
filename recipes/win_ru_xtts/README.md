# win_ru_xtts

This recipe is a simplified launcher for a Russian XTTS workflow.

It assumes:

- `epub_to_audiobook` runs on the current machine
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

This recipe is intentionally more general than a personal local wrapper. A personal wrapper can call it with machine-specific defaults.
