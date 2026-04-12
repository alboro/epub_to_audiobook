import unittest
from unittest.mock import MagicMock

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer
from audiobook_generator.normalizers.initials_ru_normalizer import InitialsRuNormalizer
from audiobook_generator.normalizers.numbers_ru_normalizer import NumbersRuNormalizer
from audiobook_generator.normalizers.proper_nouns_ru_normalizer import ProperNounsRuNormalizer
from audiobook_generator.normalizers.pronunciation_exceptions_ru_normalizer import (
    PronunciationExceptionsRuNormalizer,
)
from audiobook_generator.normalizers.ru_text_utils import COMBINING_ACUTE
from audiobook_generator.normalizers.stress_words_ru_normalizer import StressWordsRuNormalizer


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
        normalize_tts_safe_max_chars=160,
        normalize_pronunciation_exceptions_file=None,
        normalize_stress_exceptions_file=None,
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
            normalizer.normalize("Это одно из чудес и больших беды."),
            "Это одно из чуде́с и больших беды́.",
        )


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


if __name__ == "__main__":
    unittest.main()
