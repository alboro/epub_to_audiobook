# DONE

- Expand `numbers_ru` to handle Russian years, dates, time, money, fractions, Roman numerals, and broader ordinal/case agreement patterns such as `1 глава / к 1 главе / из 1 главы`.
- Evaluate deterministic morphology helpers such as `pymorphy3` for harder numeric agreement.
- ~~Evaluate optional Russian text-normalization backends such as `runorm` and `saarus72/text_normalization`.~~ **Done** — evaluated both; `runorm.RuleNormalizer` integrated as optional backend in new `abbreviations_ru` step (common text abbreviations т.д./т.е./и пр. + ALL-CAPS acronym expansion США→эс-ша-а); `saarus72/russian_text_normalizer` (FRED-T5, ~1.2 GB, HuggingFace-only, no pip) not integrated — too heavy and already superseded by `numbers_ru` + `openai` LLM step; 20 new tests.
- ~~Explore a separate `stress_ru` step for accents and `ё` restoration.~~ **Done** — confirmed `tsnorm_ru` (aliased as `stress_ru`) already covers both: unambiguous stress placement via `tsnorm` dictionary + ё restoration (`stress_yo=True`). Homographs left unstressed. 8 exploration tests added.
- ~~Integrate `gramdict/zalizniak-2010` as a data source for stress variants and feed those variants into `stress_ambiguity_llm`.~~ **Done** — `zalizniak_support.py` added: lazily downloads all Зализняк text files from GitHub (CC BY-NC), parses lemmas with embedded U+0301 stress marks, stores ~110k entries in the pronunciation lexicon DB as `source="zalizniak"`; `build_zalizniak_pronunciation_lexicon()` + `ensure_pronunciation_lexicon_db(include_zalizniak=True)` added; 25 tests (17 offline + 8 integration).
- ~~в тестах надо не `"`, а `` ` `` сделать, а то тесты падают.~~ **Done** — `simple_symbols_normalizer.py` оставлен с бэктиком (как изменил пользователь), тест обновлён.
- импортировать **Qwen3 TTS** из `7enChan/reson` (`qwen_tts_provider.py` + конфиг) — `dashscope` pip, поддержка RU, голоса Cherry/Ethan, язык `ru`. ~~**Done**~~ — `qwen_tts_provider.py` добавлен, `TTS_QWEN="qwen"` зарегистрирован, CLI-аргументы `--qwen_*`, вкладка Qwen в WebUI, `dashscope` в requirements.txt (закомментировано как optional).
- импортировать **Gemini TTS** из `7enChan/reson` (`gemini_tts_provider.py`) — `google-genai` pip, модель `gemini-2.5-pro-preview-tts`. ~~**Done**~~ — `gemini_tts_provider.py` добавлен (optional import `google-genai`), `TTS_GEMINI="gemini"` зарегистрирован, CLI-аргументы `--gemini_*`, вкладка Gemini в WebUI, `google-genai` в requirements.txt (закомментировано).
- импортировать **Kokoro TTS** из `kroryan/epub_to_audiobook` (`kokoro_tts_provider.py`) — Kokoro-FastAPI, voice mixing, у нас уже есть `docker-compose.kokoro-example.yml`. ~~**Done**~~ — `kokoro_tts_provider.py` добавлен (чистая адаптация без лишних зависимостей), `TTS_KOKORO="kokoro"` зарегистрирован, CLI-аргументы `--kokoro_*`, вкладка Kokoro в WebUI.
- импортировать **filename sanitizer** из `CroquetFlamingo/epub_to_audiobook` — фикс длинных заголовков глав. ~~**Done**~~ — `filename_sanitizer.py` уже есть в проекте и используется в `audiobook_generator.py` и `m4b_packager.py`.

## Таблица форков p0n1/epub_to_audiobook

> Базовый репо: **p0n1/epub_to_audiobook** (1954 ⭐, 207 форков). TTS: Azure, Edge, OpenAI, Piper. WebUI.

| Форк | Обновлён | Уникальные фичи | Импортировать? |
|------|----------|-----------------|----------------|
| **7enChan/reson** | 2025-10 | ➕ **Qwen3 TTS** (`qwen_tts_provider.py`) — Aliyun API, поддерживает **Russian**, Cherry/Ethan/etc. голоса, ~$0.011/1k chars<br>➕ **Gemini TTS** (`gemini_tts_provider.py`) — Google GenAI, `gemini-2.5-pro-preview-tts`, 30+ голосов, `google-genai` SDK<br>➕ **MiniMax TTS** (`minimax_tts_provider.py`) — `fal-ai/minimax`, Chinese-first, языковой буст для Russian | ✅ **Qwen3** (приоритет, поддержка RU)<br>✅ **Gemini** (мощный мультиязычный)<br>⏸ MiniMax (Chinese-first, низкий приоритет) |
| **kroryan/epub_to_audiobook** | 2025-10 | ➕ **Coqui/XTTS TTS** — local XTTS с SSL-фиксами, Spanish-focus, хакерский код<br>➕ **Kokoro TTS** — полный Kokoro-FastAPI провайдер с voice mixing, audio quality processor<br>➕ **num2words** интеграция — мультиязычная нормализация чисел<br>➕ voice presets system (`voice_presets/`)<br>➕ `tray_app.py` — системный трей<br>➕ аудио quality processor | ✅ **Kokoro** (у нас уже есть docker-compose.kokoro-example.yml)<br>⏸ Coqui (SSL-хаки, плохое качество кода)<br>⏸ num2words (у нас своя нормализация)<br>⏸ tray_app (LATER) |
| **aarongayle/epub_to_audiobook** | 2025-12 | ➕ **Gemini TTS** — своя реализация через Google GenAI SDK | ⏸ Дублирует 7enChan, взять из 7enChan |
| **Tspm1eca/epub_to_audiobook_Chinese** | 2025-11 | ➕ AI summary каждой главы + `auto_ebook.py`<br>➕ фикс обрезки длинного текста edge-tts | ⏸ Chinese-focus, нерелевантно |
| **CroquetFlamingo/epub_to_audiobook** | 2026-03 | ➕ filename sanitizer для длинных заголовков глав | ✅ filename sanitizer — полезная маленькая фича |

### Приоритет импорта

1. 🔴 **Qwen3 TTS** (`7enChan/reson`) — `dashscope` API, поддержка Russian — **самый высокий приоритет**
2. 🟠 **Gemini TTS** (`7enChan/reson`) — `google-genai`, мощный, мультиязычный
3. 🟡 **Kokoro TTS** (`kroryan`) — Kokoro-FastAPI провайдер (у нас уже docker-compose)
4. 🟢 filename sanitizer (`CroquetFlamingo`) — пара строк, легко

# TODO

- сейчас работать с текстом неудобно -- длинные куски аудио содержат ужасные ошибки, ревьюить такое неудобно, удобнее, если можно будет заменить предложение и перегенерить как бы всё.. но реально только одно предложение и будет перегенерировано при чём с сохранением старого куска для альтернативы и мапингом в БД старого куска и нового (на будущее). сборка аудиокниги в таком режиме тогда становится сложнее - надо сгруппировать предложения по главам, а потом уже собирать mb4 файл.
- ~~этапы создания книги недостаточно выражены -- 1-prepare (могут идти многократно и с ресьюмом) 2-звук 3-склеивание аудиокниги - всё это должны быть чётки режимы работы программы. режим 4 - сделать всё сразу по порядку. если режим не передан, то программа не запускается.~~ **Done** — `--mode {prepare,audio,package,all}` добавлен как required CLI-аргумент; `prepare` → пишет .txt для ревью, `audio` → TTS, `package` → упаковка существующих аудиофайлов в .m4b без запуска TTS, `all` → нормализация + TTS + упаковка; WebUI сохраняет обратную совместимость (mode=None); `args_test.py` обновлён, 11 тестов.
- сделать поддержку ini конфига, где легко можно обозревать все текущие настройки и чтоб они не повторялись в разных слоях много раз. из конфига в приложение настройки забирает компонент, где описаны все опции-директивы, они хранятся только внутри него, чтобы не нужно было переопределять ключи в разных слоях, упростить, если где-либо опции повторяются с разных case-ах, out директория по умолчанию -- рядом с исходным файлом. не нужно больше его помещать в _source. нужно помещать в папку ini файл, при создании можно генерить его на основе переданных параметров или если ничего не передано, то создавать из параметров де-факто, параметрами по умолачнию итд. ини файл фиксирует, с какими настройками запускается генерация, имя директории совпадает с именем файла книги, но без расширения. внутри папки один ini-файл -- text - там тексты (и бэкап ini-файла, с которым запускалось), wav - там звуки (и копия ini-файла, с которым запускалось), и папка или просто файл под именем книги, но с расширением аудиокниги. каждый прогон не переписывает текст или аудио, а создаёт новую порядковую подпапку в текстах и звуках. звуки генерятся на основе текстов, так что порядковый номер папки со звуками совпадает с порядковым номером папки с текстами. конечно, если переданы другие параметры, тогда генерируется там, где передано, речь об удобных умочаниях.
- есть система ресьюма генерации текстов. но нет со звуками. нынешний режим -- полностью слать текст в ттс-сервер. это ок, но нужен ещё один режим генерации звуков -- чанки и их учёт в sqlite. в частности -- разбивать на предложения и слать в ттс каждое прежложение отдельно. тогда хэш предложения - это имя звукофайла + расширение. потом будем делать прогон генерации звуков, разбивать текст на чанки-предложения и будет понятно, какое предложение сгенерировано уже на диске, а какое нужно отослать на ттс-сервер, чтобы догенерить (текст может меняться).
- это будет создавать ситуацию, что некоторые аудиочанки будет устаревать, их нужно чистить во время прогона, но не удалять. текст источник хэшей, а файловая система - источник аудиочанков. но в БД можно складывать как раз тексты с хешами и во время прогона отмечать, какие чанки-с-хешами устарели (были руками изменены в файле) и на какие именно они были заменены.. только как определять, на какие заменены? по положению новых хешей относительно остальных -- можно судить. связка старых и новых хешей для того, чтобы со временем (потом) можно было сравнивать альтернативные, но исправленные звучания между собой и выбирать лучшее.  

# LATER

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