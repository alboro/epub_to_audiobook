# TODO

- Expand `numbers_ru` to handle Russian years, dates, time, money, fractions, Roman numerals, and broader ordinal/case agreement patterns such as `1 глава / к 1 главе / из 1 главы`.
- Explore a separate `stress_ru` step for accents and `ё` restoration.
- Evaluate deterministic morphology helpers such as `pymorphy3` for harder numeric agreement.
- Build a pronunciation lexicon/index for XTTS-specific Russian reading overrides, including cases where spelling and desired sounding differ.
- Integrate `gramdict/zalizniak-2010` as a data source for stress variants and feed those variants into `stress_ambiguity_llm`.
- Evaluate whether `silero-stress` homograph lists can seed or validate `stress_ambiguity_llm` candidates before LLM selection.
- Keep new normalizer steps PR-friendly so they can be proposed upstream later.
- Grow a project-owned XTTS exception lexicon for bad stresses and bad pronunciations discovered during listening.
- Explore optional Russian accent backends such as `silero-stress` without making them part of the default chain.
- Evaluate optional Russian text-normalization backends such as `runorm` and `saarus72/text_normalization`.
- Add first-class multi-language support across recipes, normalizers, and backend selection instead of assuming Russian-only defaults.
- Make the recipe path work cleanly from any OS, not only Windows.
- Support a network setup where macOS/Linux runs `epub_to_audiobook` and sends TTS jobs to a Windows XTTS server over LAN.
- Add folder-watch automation: monitor a shared folder, enqueue new books automatically, and start generation when files appear.
- Add native `fb2` support instead of requiring prior conversion to EPUB.
