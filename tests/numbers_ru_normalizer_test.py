import unittest
from unittest.mock import MagicMock
from pathlib import Path
import tempfile
import json

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer
from audiobook_generator.normalizers.initials_ru_normalizer import InitialsRuNormalizer
from audiobook_generator.normalizers.numbers_ru_normalizer import NumbersRuNormalizer
from audiobook_generator.normalizers.openai_normalizer import OpenAINormalizer
from audiobook_generator.normalizers.pipeline_runner import NormalizationPipelineRunner
from audiobook_generator.normalizers.proper_nouns_pronunciation_ru_normalizer import (
    ProperNounsPronunciationRuNormalizer,
)
from audiobook_generator.normalizers.proper_nouns_ru_normalizer import ProperNounsRuNormalizer
from audiobook_generator.normalizers.pronunciation_exceptions_ru_normalizer import (
    PronunciationExceptionsRuNormalizer,
)
from audiobook_generator.normalizers.ru_text_utils import (
    COMBINING_ACUTE,
    load_choice_mapping_file,
    plus_stress_to_combining_acute,
)
from audiobook_generator.normalizers.simple_symbols_normalizer import SimpleSymbolsNormalizer
from audiobook_generator.normalizers.stress_ambiguity_llm_normalizer import (
    StressAmbiguityLLMNormalizer,
)
from audiobook_generator.normalizers.tts_safe_split_normalizer import TTSSafeSplitNormalizer
from audiobook_generator.normalizers.stress_words_ru_normalizer import StressWordsRuNormalizer
from audiobook_generator.normalizers.llm_support import (
    NormalizerLLMChoiceService,
    NormalizerLLMChoiceItem,
    NormalizerLLMChoiceOption,
)


def make_config(**overrides):
    values = dict(
        input_file="examples/The_Life_and_Adventures_of_Robinson_Crusoe.epub",
        output_folder="output",
        preview=False,
        output_text=False,
        prepare_text=False,
        prepared_text_folder=None,
        log="INFO",
        no_prompt=True,
        worker_count=1,
        use_pydub_merge=False,
        package_m4b=False,
        m4b_filename=None,
        m4b_bitrate="64k",
        ffmpeg_path="ffmpeg",
        title_mode="auto",
        chapter_mode="documents",
        newline_mode="double",
        chapter_start=1,
        chapter_end=-1,
        remove_endnotes=False,
        remove_reference_numbers=False,
        search_and_replace_file="",
        tts="openai",
        language="ru-RU",
        voice_name="reference",
        output_format="wav",
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        openai_api_key=None,
        openai_base_url=None,
        openai_max_chars=0,
        openai_enable_polling=False,
        openai_submit_url=None,
        openai_status_url_template=None,
        openai_download_url_template=None,
        openai_job_id_path="id",
        openai_job_status_path="status",
        openai_job_download_url_path="download_url",
        openai_job_done_values="done,completed,succeeded,success",
        openai_job_failed_values="failed,error,cancelled",
        openai_poll_interval=5,
        openai_poll_timeout=60,
        openai_poll_request_timeout=60,
        openai_poll_max_errors=3,
        instructions=None,
        speed=1.0,
        normalize=True,
        normalize_steps="numbers_ru",
        normalize_provider="openai",
        normalize_model="gpt-5.4",
        normalize_prompt_file=None,
        normalize_system_prompt_file=None,
        normalize_user_prompt_file=None,
        normalize_api_key=None,
        normalize_base_url=None,
        normalize_max_chars=4000,
        normalize_tts_safe_max_chars=180,
        normalize_pronunciation_exceptions_file=None,
        normalize_stress_exceptions_file=None,
        normalize_stress_ambiguity_file=None,
        normalize_tsnorm_stress_yo=True,
        normalize_tsnorm_stress_monosyllabic=False,
        normalize_tsnorm_min_word_length=2,
        break_duration="1250",
        voice_rate=None,
        voice_volume=None,
        voice_pitch=None,
        proxy=None,
        piper_path="piper",
        piper_docker_image="lscr.io/linuxserver/piper:latest",
        piper_speaker=0,
        piper_noise_scale=None,
        piper_noise_w_scale=None,
        piper_length_scale=1.0,
        piper_sentence_silence=0.2,
    )
    values.update(overrides)
    return GeneralConfig(MagicMock(**values))


class DummyNormalizer(BaseNormalizer):
    def validate_config(self):
        return None

    def normalize(self, text: str, chapter_title: str = "") -> str:
        return text


class DummyChunkedNormalizer(BaseNormalizer):
    STEP_NAME = "dummy_chunked"

    def __init__(self, config: GeneralConfig):
        self.process_calls = []
        super().__init__(config)

    def validate_config(self):
        return None

    def normalize(self, text: str, chapter_title: str = "") -> str:
        return text

    def supports_chunked_resume(self) -> bool:
        return True

    def plan_processing_units(self, text: str, chapter_title: str = "") -> list[str]:
        return [part for part in text.split("|") if part]

    def process_unit(
        self,
        unit: str,
        *,
        chapter_title: str = "",
        unit_index: int,
        unit_count: int,
    ) -> str:
        self.process_calls.append((unit_index, unit))
        return unit.upper()

    def get_step_artifacts(self, text: str, chapter_title: str = "") -> dict[str, str]:
        return {"00_system_prompt.txt": "dummy system"}

    def get_unit_artifacts(
        self,
        unit: str,
        *,
        chapter_title: str = "",
        unit_index: int,
        unit_count: int,
    ) -> dict[str, str]:
        return {"01_user_prompt.txt": f"prompt for {unit}"}


class FakeLLM:
    def __init__(self, *, settings, response_text):
        self.settings = settings
        self.response_text = response_text
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        return self.response_text


class TestNumbersRuNormalizer(unittest.TestCase):
    def test_plain_cardinal(self):
        normalizer = NumbersRuNormalizer(make_config())
        self.assertEqual(normalizer.normalize("У меня 3 яблока."), "У меня три яблока.")

    def test_number_sign(self):
        normalizer = NumbersRuNormalizer(make_config())
        self.assertEqual(normalizer.normalize("Смотри №5."), "Смотри номер пять.")

    def test_arabic_century(self):
        normalizer = NumbersRuNormalizer(make_config())
        self.assertEqual(normalizer.normalize("17 век"), "семнадцатый век")

    def test_roman_century(self):
        normalizer = NumbersRuNormalizer(make_config())
        self.assertEqual(normalizer.normalize("XVII век"), "семнадцатый век")

    def test_ordinal_suffix(self):
        normalizer = NumbersRuNormalizer(make_config())
        self.assertEqual(normalizer.normalize("17-й век"), "семнадцатый век")

    def test_cardinal_with_feminine_book_noun(self):
        normalizer = NumbersRuNormalizer(make_config())
        self.assertEqual(normalizer.normalize("1 глава"), "одна глава")

    def test_cardinal_with_dative_book_noun(self):
        normalizer = NumbersRuNormalizer(make_config())
        self.assertEqual(normalizer.normalize("к 1 главе"), "к одной главе")


class TestDeterministicRuNormalizers(unittest.TestCase):
    def test_initials_ru(self):
        normalizer = InitialsRuNormalizer(make_config(normalize_steps="initials_ru"))
        self.assertEqual(
            normalizer.normalize("вкратце о Е. Д. Калашниковой"),
            "вкратце о Е-Дэ-Калашниковой",
        )

    def test_initials_ru_with_extra_spaces(self):
        normalizer = InitialsRuNormalizer(make_config(normalize_steps="initials_ru"))
        self.assertEqual(
            normalizer.normalize("вкратце о Е.   Д.   Калашниковой"),
            "вкратце о Е-Дэ-Калашниковой",
        )

    def test_pronunciation_exceptions_ru(self):
        normalizer = PronunciationExceptionsRuNormalizer(
            make_config(normalize_steps="pronunciation_exceptions_ru")
        )
        self.assertEqual(
            normalizer.normalize("Отель расположен рядом."),
            "Отэль расположен рядом.",
        )

    def test_stress_words_ru(self):
        normalizer = StressWordsRuNormalizer(make_config(normalize_steps="stress_words_ru"))
        self.assertEqual(
            normalizer.normalize("Это одно из чудес, а не все чудеса."),
            "Это одно из чуде́с, а не все чудеса́.",
        )

    def test_stress_words_ru_leaves_ambiguous_word_unchanged(self):
        normalizer = StressWordsRuNormalizer(make_config(normalize_steps="stress_words_ru"))
        self.assertEqual(
            normalizer.normalize("И после беды пришли новые беды."),
            "И после беды пришли новые беды.",
        )

    def test_simple_symbols_aggressive_cleanup_preserves_letters_digits_and_stress(self):
        normalizer = SimpleSymbolsNormalizer(make_config(normalize_steps="simple_symbols"))
        self.assertEqual(
            normalizer.normalize('Текст\u200b с\u00a0мусором • и™ акце́нтом… «цитата» — 42'),
            'Текст с мусором и акце́нтом... "цитата" - 42',
        )

    def test_plus_stress_to_combining_acute(self):
        self.assertEqual(
            plus_stress_to_combining_acute("б+еды"),
            f"бе{COMBINING_ACUTE}ды",
        )

    def test_load_choice_mapping_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "ambiguities.txt"
            mapping_path.write_text(
                "беды==б+еды|бед+ы\nпоступи==п+оступи|поступ+и\n",
                encoding="utf-8",
            )
            mapping = load_choice_mapping_file(str(mapping_path))
            self.assertEqual(
                mapping["беды"],
                (f"бе{COMBINING_ACUTE}ды", f"беды{COMBINING_ACUTE}"),
            )


class TestTTSSafeSplitNormalizer(unittest.TestCase):
    def test_avoids_single_word_tail_split(self):
        normalizer = TTSSafeSplitNormalizer(
            make_config(
                normalize_steps="tts_safe_split",
                normalize_tts_safe_max_chars=160,
            )
        )
        result = normalizer.normalize(
            'О, если б голова моя была водой, и очи мои - Исто́чниками, льющимися, как жидкие небеса. Тогда́ бы я дал волю могучему потоку И оплакал бы потопом род человеческий'
        )
        self.assertNotIn("род. Человеческий", result)
        self.assertIn("род человеческий", result)


class TestSharedNormalizerLLMSupport(unittest.TestCase):
    def test_default_user_prompt_is_plain_text_only(self):
        normalizer = DummyNormalizer(
            make_config(
                normalize_steps="simple_symbols",
                normalize_base_url="http://127.0.0.1:1234/v1",
            )
        )
        llm = normalizer.get_normalizer_llm()
        self.assertEqual(llm.settings.user_prompt_template, "{text}")
        self.assertEqual(
            llm.render_user_prompt(chapter_title="Глава 1", text="Привет, мир."),
            "Привет, мир.",
        )

    def test_llm_runtime_is_shared_for_same_config(self):
        config = make_config(
            normalize_steps="simple_symbols",
            normalize_base_url="http://127.0.0.1:1234/v1",
        )
        first = DummyNormalizer(config)
        second = DummyNormalizer(config)
        self.assertIs(first.get_normalizer_llm(), second.get_normalizer_llm())
        self.assertTrue(first.has_normalizer_llm())

    def test_openai_small_chunk_merge(self):
        merged = OpenAINormalizer._merge_small_chunks(
            ["A" * 3900, "B" * 900, "C" * 3800],
            4000,
        )
        self.assertEqual(len(merged), 2)
        self.assertIn("B" * 900, merged[0])

    def test_choice_response_parser_accepts_json_fences(self):
        parsed = NormalizerLLMChoiceService.parse_choice_response(
            """```json
            {"selections":[{"id":"item-1","option_id":"phonetic"}]}
            ```"""
        )
        self.assertEqual(parsed, {"item-1": "phonetic"})

    def test_choice_response_parser_accepts_cacheable_and_custom_text(self):
        parsed = NormalizerLLMChoiceService.parse_choice_response_objects(
            """{
              "selections": [
                {
                  "id": "item-1",
                  "custom_text": "То́мас Пэйн",
                  "cacheable": true,
                  "reason": "Stable surname pronunciation"
                }
              ]
            }"""
        )
        self.assertEqual(parsed["item-1"].custom_text, "То́мас Пэйн")
        self.assertTrue(parsed["item-1"].cacheable)
        self.assertEqual(parsed["item-1"].reason, "Stable surname pronunciation")

    def test_choice_service_reuses_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = MagicMock(
                max_chars=4000,
                choice_cache_path=str(Path(temp_dir) / "choice_cache.json"),
                model="gpt-5.4",
                provider="openai",
                base_url="http://127.0.0.1:1234/v1",
            )
            response = json.dumps(
                {
                    "selections": [
                        {
                            "id": "item-1",
                            "custom_text": "То́мас Пэйн",
                            "cacheable": True,
                            "reason": "Stable pronunciation",
                        }
                    ]
                },
                ensure_ascii=False,
            )
            llm = FakeLLM(settings=settings, response_text=response)
            service = NormalizerLLMChoiceService(llm)
            item = NormalizerLLMChoiceItem(
                item_id="item-1",
                source_text="Томас Пейн",
                context="Томас Пейн писал эссе.",
                note=None,
                options=(
                    NormalizerLLMChoiceOption("original", "Томас Пейн"),
                    NormalizerLLMChoiceOption("guided", "То́мас Пэйн"),
                ),
            )

            first = service.choose_batch([item], target_language="ru-RU")
            second = service.choose_batch([item], target_language="ru-RU")

            self.assertEqual(first["item-1"].custom_text, "То́мас Пэйн")
            self.assertEqual(second["item-1"].source, "cache")
            self.assertEqual(llm.calls, 1)


class TestProperNounsRuNormalizer(unittest.TestCase):
    def test_accents_internal_proper_nouns(self):
        normalizer = ProperNounsRuNormalizer(make_config(normalize_steps="proper_nouns_ru"))
        normalizer.backend = lambda text: {
            "Фицджеральда": "Фицджера́льда",
            "Калашниковой": "Кала́шниковой",
            "Нью-Йорка": "Нью-Йо́рка",
        }.get(text, text)
        self.assertEqual(
            normalizer.normalize("Раньше уже говорилось вкратце о Е. Д. Калашниковой и Фицджеральда из Нью-Йорка."),
            "Раньше уже говорилось вкратце о Е. Д. Кала́шниковой и Фицджера́льда из Нью-Йо́рка.",
        )

    def test_skips_sentence_start_words(self):
        normalizer = ProperNounsRuNormalizer(make_config(normalize_steps="proper_nouns_ru"))
        normalizer.backend = lambda text: text + COMBINING_ACUTE
        self.assertEqual(
            normalizer.normalize("Для ясности автор этой книжки не лингвист. Но Фицджеральда он помнит."),
            "Для ясности автор этой книжки не лингвист. Но Фицджеральда́ он помнит.",
        )


class TestProperNounsPronunciationRuNormalizer(unittest.TestCase):
    def test_builds_pronunciation_variants_and_applies_selected_options(self):
        normalizer = ProperNounsPronunciationRuNormalizer(
            make_config(
                normalize_steps="proper_nouns_pronunciation_ru",
                normalize_base_url="http://127.0.0.1:1234/v1",
            )
        )

        def fake_choose_batch(items, **kwargs):
            selections = {}
            for item in items:
                if item.source_text == "Томас Пейн":
                    selections[item.item_id] = "guided"
                elif item.source_text == "Лев Толстой":
                    selections[item.item_id] = "guided"
                else:
                    selections[item.item_id] = "original"
            return selections

        normalizer.choice_service.choose_batch = fake_choose_batch
        result = normalizer.normalize(
            'Томас Пейн писал о вере, а позже Лев Толстой спорил с газетой "Таймс".'
        )

        self.assertIn("Пэйн", result)
        self.assertIn(f"Лев Толсто{COMBINING_ACUTE}й", result)
        self.assertIn(f"То{COMBINING_ACUTE}мас Пэйн", result)
        self.assertIn('"Таймс"', result)

    def test_prompt_artifacts_include_context_and_options(self):
        normalizer = ProperNounsPronunciationRuNormalizer(
            make_config(
                normalize_steps="proper_nouns_pronunciation_ru",
                normalize_base_url="http://127.0.0.1:1234/v1",
            )
        )
        text = 'В лондонскую газету "Таймс" писал Томас Пейн.'
        units = normalizer.plan_processing_units(text, chapter_title="Test")
        self.assertTrue(units)
        artifacts = normalizer.get_unit_artifacts(
            units[0],
            chapter_title="Test",
            unit_index=1,
            unit_count=len(units),
        )
        self.assertIn("Томас Пейн", artifacts["01_choice_user_prompt.txt"])
        self.assertIn("Пэйн", artifacts["01_choice_user_prompt.txt"])

    def test_skips_generic_leading_sentence_word_in_candidate(self):
        normalizer = ProperNounsPronunciationRuNormalizer(
            make_config(
                normalize_steps="proper_nouns_pronunciation_ru",
                normalize_base_url="http://127.0.0.1:1234/v1",
            )
        )
        units = normalizer.plan_processing_units("Когда Моисей сказал народу.", chapter_title="Test")
        self.assertEqual(len(units), 1)
        artifacts = normalizer.get_unit_artifacts(
            units[0],
            chapter_title="Test",
            unit_index=1,
            unit_count=1,
        )
        self.assertIn('"source_text": "Моисей"', artifacts["01_choice_user_prompt.txt"])
        self.assertIn('"context": "Когда Моисей сказал народу"', artifacts["01_choice_user_prompt.txt"])

    def test_collapses_double_stress_marks_from_backend(self):
        normalizer = ProperNounsPronunciationRuNormalizer(
            make_config(
                normalize_steps="proper_nouns_pronunciation_ru",
                normalize_base_url="http://127.0.0.1:1234/v1",
            )
        )
        normalizer.backend = lambda text: {
            "Новый": f"Но{COMBINING_ACUTE}вый",
            "Завет": f"За{COMBINING_ACUTE}ве{COMBINING_ACUTE}т",
        }.get(text, text)
        self.assertEqual(
            normalizer._accent_phrase("Новый Завет"),
            f"Но{COMBINING_ACUTE}вый Заве{COMBINING_ACUTE}т",
        )

    def test_post_step_artifacts_include_selection_stats(self):
        normalizer = ProperNounsPronunciationRuNormalizer(
            make_config(
                normalize_steps="proper_nouns_pronunciation_ru",
                normalize_base_url="http://127.0.0.1:1234/v1",
            )
        )
        normalizer.choice_service.choose_batch = lambda items, **kwargs: {
            items[0].item_id: MagicMock(
                option_id="guided",
                custom_text=None,
                cacheable=True,
                reason="Stable pronunciation",
                source="llm",
                has_custom_text=False,
                resolved_option_id=lambda: "guided",
            )
        }
        result = normalizer.normalize("Томас Пейн писал эссе.")
        artifacts = normalizer.get_post_step_artifacts(
            input_text="Томас Пейн писал эссе.",
            output_text=result,
            chapter_title="Test",
        )
        self.assertIn("changed_candidates", artifacts["93_selection_stats.json"])
        self.assertIn("Томас Пейн", artifacts["92_selection_report.txt"])


class TestStressAmbiguityLLMNormalizer(unittest.TestCase):
    def test_selects_contextual_variants_for_same_surface_form(self):
        normalizer = StressAmbiguityLLMNormalizer(
            make_config(
                normalize_steps="stress_ambiguity_llm",
                normalize_base_url="http://127.0.0.1:1234/v1",
            )
        )

        def fake_choose_batch(items, **kwargs):
            selections = {}
            for item in items:
                if item.item_id.endswith("0001"):
                    selections[item.item_id] = "variant_2"
                elif item.item_id.endswith("0002"):
                    selections[item.item_id] = "variant_1"
                else:
                    selections[item.item_id] = "original"
            return selections

        normalizer.choice_service.choose_batch = fake_choose_batch
        result = normalizer.normalize("После беды пришли новые беды.")
        self.assertEqual(
            result,
            f"После беды{COMBINING_ACUTE} пришли новые бе{COMBINING_ACUTE}ды.",
        )

    def test_skips_already_accented_word(self):
        normalizer = StressAmbiguityLLMNormalizer(
            make_config(
                normalize_steps="stress_ambiguity_llm",
                normalize_base_url="http://127.0.0.1:1234/v1",
            )
        )
        units = normalizer.plan_processing_units(
            f"После беды{COMBINING_ACUTE} пришли новые бе{COMBINING_ACUTE}ды.",
            chapter_title="Test",
        )
        self.assertEqual(units, [])

    def test_post_step_artifacts_include_reports(self):
        normalizer = StressAmbiguityLLMNormalizer(
            make_config(
                normalize_steps="stress_ambiguity_llm",
                normalize_base_url="http://127.0.0.1:1234/v1",
            )
        )
        normalizer.choice_service.choose_batch = lambda items, **kwargs: {
            items[0].item_id: MagicMock(
                option_id="variant_2",
                custom_text=None,
                cacheable=False,
                reason="Context says genitive singular",
                source="llm",
                has_custom_text=False,
                resolved_option_id=lambda: "variant_2",
            )
        }
        result = normalizer.normalize("После беды.")
        artifacts = normalizer.get_post_step_artifacts(
            input_text="После беды.",
            output_text=result,
            chapter_title="Test",
        )
        self.assertIn("changed_candidates", artifacts["93_selection_stats.json"])
        self.assertIn("После беды", artifacts["92_selection_report.txt"])


class TestNormalizationPipelineRunner(unittest.TestCase):
    def test_resumes_chunked_units_and_saves_prompt_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_config(output_folder=temp_dir, normalize_steps="dummy_chunked")
            artifact_dir = Path(temp_dir) / "_chapter_artifacts" / "0001_Test"
            first = DummyChunkedNormalizer(config)
            runner = NormalizationPipelineRunner(config=config, artifact_dir=artifact_dir)
            normalized, trace = runner.run(first, "alpha|beta", "Test")

            self.assertEqual(normalized, "ALPHA\n\nBETA")
            self.assertEqual(first.process_calls, [(1, "alpha"), (2, "beta")])
            self.assertEqual(trace, [("dummy_chunked", "ALPHA\n\nBETA")])
            self.assertTrue((artifact_dir / "_normalizer_steps" / "01_dummy_chunked" / "00_system_prompt.txt").is_file())
            self.assertTrue((artifact_dir / "_normalizer_steps" / "01_dummy_chunked" / "chunks" / "0001" / "01_user_prompt.txt").is_file())
            self.assertTrue((artifact_dir / "_normalizer_steps" / "01_dummy_chunked" / "90_changes.md").is_file())
            self.assertTrue((artifact_dir / "_normalizer_steps" / "01_dummy_chunked" / "91_changes.diff").is_file())
            self.assertTrue((artifact_dir / "00_normalizer_change_summary.md").is_file())

            second = DummyChunkedNormalizer(config)
            second_runner = NormalizationPipelineRunner(config=config, artifact_dir=artifact_dir)
            resumed, resumed_trace = second_runner.run(second, "alpha|beta", "Test")

            self.assertEqual(resumed, "ALPHA\n\nBETA")
            self.assertEqual(resumed_trace, [("dummy_chunked", "ALPHA\n\nBETA")])
            self.assertEqual(second.process_calls, [])


if __name__ == "__main__":
    unittest.main()
