- Build a pronunciation lexicon/index for XTTS-specific Russian reading overrides, including cases where spelling and desired sounding differ. отель -> отэль, атеизм -> атэизм (и все формы)
- Evaluate whether `silero-stress` homograph lists can seed or validate `stress_ambiguity_llm` candidates before LLM selection.
- Add first-class multi-language support across recipes, normalizers, and backend selection instead of assuming Russian-only defaults.
- со временем сделаем интерфейс и возможность прослушивать прежние версии текста
- XTTS в обсуждениях рекомендуют синтезировать по одному предложению и убирать лишнюю пунктуацию там, где это допустимо.
- Не подавать в XTTS слишком короткие фразы как есть. Для фраз уровня “Да.”, “Нет.”, “Спасибо.”, “Хорошо.”, “Идём.” я бы тестировал:
  либо объединение с соседней репликой,
  либо добавление безопасного контекста,
  либо постобрезку хвоста по тишине.
- Add folder-watch automation: monitor a shared folder, enqueue new books automatically, and start generation when files appear.
- import MD & gemini tts & Qwen3 TTS support https://github.com/p0n1/epub_to_audiobook/compare/main...7enChan:reson:main
- CosyVoice 2 -- тоже очень сильный open-source вариант для русского.
   В официальном репозитории и модели указаны русский язык, zero-shot / cross-lingual voice cloning и упор на naturalness, speaker similarity и prosody. Если Qwen3-TTS по какой-то причине не зайдёт, CosyVoice 2 -- следующий кандидат для серьёзного теста.
- cuqui tts & ffmpeg & num2words https://github.com/p0n1/epub_to_audiobook/compare/main...kroryan:epub_to_audiobook:main