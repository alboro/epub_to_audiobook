- Build a pronunciation lexicon/index for XTTS-specific Russian reading overrides, including cases where spelling and desired sounding differ. отель -> отэль, атеизм -> атэизм (и все формы)
- XTTS: убирать лишнюю пунктуацию там, где это допустимо.
- Evaluate whether `silero-stress` homograph lists can seed or validate `stress_ambiguity_llm` candidates before LLM selection.
- унифицировать ini и webui, чтобы был один ini файл, но его бы переопределял инифайл, который внутри директории-для-книги, бекап-ини-файлы должны иметь другое расширение
- всё, что депрекейдед в проекте, следует удалить, если его не было в оригинальном проекте.
- Add first-class multi-language support across recipes, normalizers, and backend selection instead of assuming Russian-only defaults.
- со временем сделаем интерфейс и возможность прослушивать прежние версии текста
- Не подавать в XTTS слишком короткие фразы как есть. Для фраз уровня “Да.”, “Нет.”, “Спасибо.”, “Хорошо.”, “Идём.” я бы тестировал:
  либо объединение с соседней репликой,
  либо добавление безопасного контекста,
  либо постобрезку хвоста по тишине.
- Add folder-watch automation: monitor a shared folder, enqueue new books automatically, and start generation when files appear.
- CosyVoice 2 -- сильный open-source вариант для русского.
   В официальном репозитории и модели указаны русский язык, zero-shot / cross-lingual voice cloning и упор на naturalness, speaker similarity и prosody