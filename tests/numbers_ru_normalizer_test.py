import unittest
from unittest.mock import MagicMock

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.numbers_ru_normalizer import NumbersRuNormalizer


def make_config():
    args = MagicMock(
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
    return GeneralConfig(args)


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


if __name__ == "__main__":
    unittest.main()
