# TODO

- Expand `numbers_ru` to handle Russian years, dates, and more ordinal/case patterns.
- Explore a separate `stress_ru` step for accents and `ё` restoration.
- Evaluate deterministic morphology helpers such as `pymorphy3` for harder numeric agreement.
- Keep new normalizer steps PR-friendly so they can be proposed upstream later.
- Grow a project-owned XTTS exception lexicon for bad stresses and bad pronunciations discovered during listening.
- Explore optional Russian accent backends such as `silero-stress` without making them part of the default chain.
- Evaluate optional Russian text-normalization backends such as `runorm` and `saarus72/text_normalization`.
- Add first-class multi-language support across recipes, normalizers, and backend selection instead of assuming Russian-only defaults.
- Make the recipe path work cleanly from any OS, not only Windows.
- Support a network setup where macOS/Linux runs `epub_to_audiobook` and sends TTS jobs to a Windows XTTS server over LAN.
- Add folder-watch automation: monitor a shared folder, enqueue new books automatically, and start generation when files appear.
- Add native `fb2` support instead of requiring prior conversion to EPUB.
