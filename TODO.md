# TODO

- Expand `numbers_ru` to handle Russian years, dates, time, money, fractions, Roman numerals, and broader ordinal/case agreement patterns such as `1 глава / к 1 главе / из 1 главы`.
- Evaluate deterministic morphology helpers such as `pymorphy3` for harder numeric agreement.
- ~~Evaluate optional Russian text-normalization backends such as `runorm` and `saarus72/text_normalization`.~~ **Done** — evaluated both; `runorm.RuleNormalizer` integrated as optional backend in new `abbreviations_ru` step (common text abbreviations т.д./т.е./и пр. + ALL-CAPS acronym expansion США→эс-ша-а); `saarus72/russian_text_normalizer` (FRED-T5, ~1.2 GB, HuggingFace-only, no pip) not integrated — too heavy and already superseded by `numbers_ru` + `openai` LLM step; 20 new tests.
- Explore a separate `stress_ru` step for accents and `ё` restoration.
- Integrate `gramdict/zalizniak-2010` as a data source for stress variants and feed those variants into `stress_ambiguity_llm`.
- Build a pronunciation lexicon/index for XTTS-specific Russian reading overrides, including cases where spelling and desired sounding differ. отель -> отэль, атеизм -> атэизм (и все формы)
- Evaluate whether `silero-stress` homograph lists can seed or validate `stress_ambiguity_llm` candidates before LLM selection.
- Add first-class multi-language support across recipes, normalizers, and backend selection instead of assuming Russian-only defaults.
- Add folder-watch automation: monitor a shared folder, enqueue new books automatically, and start generation when files appear.
- XTTS в обсуждениях рекомендуют синтезировать по одному предложению и убирать лишнюю пунктуацию там, где это допустимо.
- Не подавать в XTTS слишком короткие фразы как есть. Для фраз уровня “Да.”, “Нет.”, “Спасибо.”, “Хорошо.”, “Идём.” я бы тестировал:
  либо объединение с соседней репликой,
  либо добавление безопасного контекста,
  либо постобрезку хвоста по тишине.