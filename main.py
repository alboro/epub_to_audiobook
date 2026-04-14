import argparse
from pathlib import Path

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core.audiobook_generator import AudiobookGenerator
from audiobook_generator.normalizers.base_normalizer import get_supported_normalizers
from audiobook_generator.tts_providers.base_tts_provider import (
    get_supported_tts_providers,
)
from audiobook_generator.utils.log_handler import setup_logging, generate_unique_log_path


def handle_args():
    parser = argparse.ArgumentParser(description="Convert text book to audiobook")
    parser.add_argument("input_file", help="Path to the input book file (EPUB or FB2)")
    parser.add_argument("output_folder", help="Path to the output folder")
    parser.add_argument(
        "--tts",
        choices=get_supported_tts_providers(),
        default=get_supported_tts_providers()[0],
        help="Choose TTS provider (default: azure). azure: Azure Cognitive Services, openai: OpenAI TTS API. When using azure, environment variables MS_TTS_KEY and MS_TTS_REGION must be set. When using openai, environment variable OPENAI_API_KEY must be set.",
    )
    parser.add_argument(
        "--log",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Log level (default: INFO), can be DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Enable preview mode. The script will run parsing and normalization, save text artifacts, and stop before sending anything to TTS.",
    )
    parser.add_argument(
        "--no_prompt",
        action="store_true",
        help="Don't ask the user if they wish to continue after estimating the cloud cost for TTS. Useful for scripting.",
    )
    parser.add_argument(
        "--language",
        default="en-US",
        help="Language for the text-to-speech service (default: en-US). For Azure TTS (--tts=azure), check https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts#text-to-speech for supported languages. For OpenAI TTS (--tts=openai), their API detects the language automatically. But setting this will also help on splitting the text into chunks with different strategies in this tool, especially for Chinese characters. For Chinese books, use zh-CN, zh-TW, or zh-HK.",
    )
    parser.add_argument(
        "--newline_mode",
        choices=["single", "double", "none"],
        default="double",
        help="Choose the mode of detecting new paragraphs: 'single', 'double', or 'none'. 'single' means a single newline character, while 'double' means two consecutive newline characters. 'none' means all newline characters will be replace with blank so paragraphs will not be detected. (default: double, works for most ebooks but will detect less paragraphs for some ebooks)",
    )
    parser.add_argument(
        "--title_mode",
        choices=["auto", "tag_text", "first_few"],
        default="auto",
        help="Choose the parse mode for chapter title, 'tag_text' search 'title','h1','h2','h3' tag for title, 'first_few' set first 60 characters as title, 'auto' auto apply the best mode for current chapter.",
    )
    parser.add_argument(
        "--chapter_mode",
        choices=["documents", "toc_sections"],
        default="documents",
        help=(
            "Choose how book content is grouped into chapters. "
            "'documents' keeps one chapter per XHTML document (EPUB) or per leaf section (FB2). "
            "'toc_sections' groups EPUB documents by top-level table-of-contents sections when possible "
            "(not applicable for FB2, falls back to per-section mode)."
        ),
    )
    parser.add_argument(
        "--chapter_start",
        default=1,
        type=int,
        help="Chapter start index (default: 1, starting from 1)",
    )
    parser.add_argument(
        "--chapter_end",
        default=-1,
        type=int,
        help="Chapter end index (default: -1, meaning to the last chapter)",
    )
    parser.add_argument(
        "--output_text",
        action="store_true",
        help="Enable Output Text. This will export a plain text file for each chapter specified and write the files to the output folder specified.",
    )
    parser.add_argument(
        "--prepare_text",
        action="store_true",
        help="Prepare per-chapter UTF-8 text files for review and stop before TTS. If --normalize is enabled, the exported text is normalized first. Chapter text artifacts are also saved.",
    )
    parser.add_argument(
        "--prepared_text_folder",
        help="Use reviewed per-chapter .txt files from this folder as the TTS source instead of the raw text extracted from the EPUB.",
    )
    parser.add_argument(
        "--remove_endnotes",
        action="store_true",
        help="This will remove endnote numbers from the end or middle of sentences. This is useful for academic books.",
    )

    parser.add_argument(
        "--remove_reference_numbers",
        action="store_true",
        help="This will remove reference numbers from the end or middle of sentences (e.g [3] or [12.1]). Also useful for academic books."
    )

    parser.add_argument(
        "--search_and_replace_file",
        default="",
        help="""Path to a file that contains 1 regex replace per line, to help with fixing pronunciations, etc. The format is:
        <search>==<replace>
        Note that you may have to specify word boundaries, to avoid replacing parts of words.
        """,
    )

    parser.add_argument(
        "--worker_count",
        type=int,
        default=1,
        help="Specifies the number of parallel workers to use for audiobook generation. "
        "Increasing this value can significantly speed up the process by processing multiple chapters simultaneously. "
        "Note: Chapters may not be processed in sequential order, but this will not affect the final audiobook.",
    )

    parser.add_argument(
        "--use_pydub_merge",
        action="store_true",
        help="Use pydub to merge audio segments of one chapter into single file instead of direct write. "
        "Currently only supported for OpenAI and Azure TTS. "
        "Direct write is faster but might skip audio segments if formats differ. "
        "Pydub merge is slower but more reliable for different audio formats. It requires ffmpeg to be installed first. "
        "You can use this option to avoid the issue of skipping audio segments in some cases. "
        "However, it's recommended to use direct write for most cases as it's faster. "
        "Only use this option if you encounter issues with direct write.",
    )
    parser.add_argument(
        "--package_m4b",
        action="store_true",
        help="Package generated chapter audio files into a single m4b audiobook with chapter markers.",
    )
    parser.add_argument(
        "--m4b_filename",
        help="Optional output filename for the packaged m4b file.",
    )
    parser.add_argument(
        "--m4b_bitrate",
        default="64k",
        help="AAC bitrate for m4b packaging (default: 64k).",
    )
    parser.add_argument(
        "--ffmpeg_path",
        default="ffmpeg",
        help="Path to ffmpeg binary used for m4b packaging.",
    )

    parser.add_argument(
        "--voice_name",
        help="Various TTS providers has different voice names, look up for your provider settings.",
    )

    parser.add_argument(
        "--output_format",
        help="Output format for the text-to-speech service. Supported format depends on selected TTS provider",
    )

    parser.add_argument(
        "--model_name",
        help="Various TTS providers has different neural model names",
    )

    openai_tts_group = parser.add_argument_group(title="openai specific")
    openai_tts_group.add_argument(
        "--speed",
        default=1.0,
        type=float,
        help="The speed of the generated audio. Select a value from 0.25 to 4.0. 1.0 is the default.",
    )

    openai_tts_group.add_argument(
        "--instructions",
        help="Instructions for the TTS model. Only supported for 'gpt-4o-mini-tts' model.",
    )
    openai_tts_group.add_argument(
        "--openai_api_key",
        help="Optional API key override for OpenAI-compatible TTS servers.",
    )
    openai_tts_group.add_argument(
        "--openai_base_url",
        help="Optional base URL override for OpenAI-compatible TTS servers.",
    )
    openai_tts_group.add_argument(
        "--openai_max_chars",
        default=1800,
        type=int,
        help="Local chunk size before sending text to the OpenAI TTS provider. Set to 0 or a negative value to disable local chunking.",
    )
    openai_tts_group.add_argument(
        "--openai_enable_polling",
        action="store_true",
        help="Use submit/poll/download workflow instead of standard synchronous OpenAI TTS response handling.",
    )
    openai_tts_group.add_argument(
        "--openai_submit_url",
        help="Submit endpoint for polling-based OpenAI-compatible TTS servers.",
    )
    openai_tts_group.add_argument(
        "--openai_status_url_template",
        help="Status URL template for polling-based TTS servers, for example '/tts/jobs/{job_id}'.",
    )
    openai_tts_group.add_argument(
        "--openai_download_url_template",
        help="Optional download URL template for completed jobs, for example '/tts/jobs/{job_id}/audio'.",
    )
    openai_tts_group.add_argument(
        "--openai_job_id_path",
        default="id",
        help="Dot path to job id in submit response JSON (default: id).",
    )
    openai_tts_group.add_argument(
        "--openai_job_status_path",
        default="status",
        help="Dot path to job status in polling response JSON (default: status).",
    )
    openai_tts_group.add_argument(
        "--openai_job_download_url_path",
        default="download_url",
        help="Dot path to download URL in polling response JSON (default: download_url).",
    )
    openai_tts_group.add_argument(
        "--openai_job_done_values",
        default="done,completed,succeeded,success",
        help="Comma-separated status values that mean the polling job is complete.",
    )
    openai_tts_group.add_argument(
        "--openai_job_failed_values",
        default="failed,error,cancelled",
        help="Comma-separated status values that mean the polling job has failed.",
    )
    openai_tts_group.add_argument(
        "--openai_poll_interval",
        default=120,
        type=int,
        help="Polling interval in seconds for job-based OpenAI-compatible TTS servers.",
    )
    openai_tts_group.add_argument(
        "--openai_poll_timeout",
        default=14400,
        type=int,
        help="Maximum time in seconds to wait for a polling TTS job before failing.",
    )
    openai_tts_group.add_argument(
        "--openai_poll_request_timeout",
        default=120,
        type=int,
        help="HTTP timeout in seconds for each individual polling or download request (default: 120).",
    )
    openai_tts_group.add_argument(
        "--openai_poll_max_errors",
        default=10,
        type=int,
        help="How many consecutive transient polling/download HTTP errors to tolerate before failing (default: 10).",
    )

    normalizer_group = parser.add_argument_group(title="normalizer specific")
    normalizer_group.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize chapter text before sending it to TTS.",
    )
    normalizer_group.add_argument(
        "--normalize_steps",
        help=(
            "Comma-separated normalizer steps to apply in order. Example: "
            "simple_symbols,initials_ru,numbers_ru,stress_ambiguity_llm,llm,"
            "simple_symbols,initials_ru,proper_nouns_pronunciation_ru,tts_safe_split"
        ),
    )
    normalizer_group.add_argument(
        "--normalize_provider",
        choices=get_supported_normalizers(),
        default="openai",
        help="Choose normalizer provider (default: openai).",
    )
    normalizer_group.add_argument(
        "--normalize_model",
        help="Model name for the LLM normalizer.",
    )
    normalizer_group.add_argument(
        "--normalize_prompt_file",
        help="Optional text file with a custom system prompt for the normalizer. Kept for backwards compatibility.",
    )
    normalizer_group.add_argument(
        "--normalize_system_prompt_file",
        help="Optional text file with a custom system prompt for the normalizer.",
    )
    normalizer_group.add_argument(
        "--normalize_user_prompt_file",
        help="Optional text file with a custom user prompt template for the normalizer. Available placeholders: {chapter_title}, {text}.",
    )
    normalizer_group.add_argument(
        "--normalize_api_key",
        help="Optional API key override for the normalizer endpoint.",
    )
    normalizer_group.add_argument(
        "--normalize_base_url",
        help="Optional base URL override for the normalizer endpoint.",
    )
    normalizer_group.add_argument(
        "--normalize_max_chars",
        default=4000,
        type=int,
        help="Maximum characters per normalization request. Use a negative value to disable local splitting.",
    )
    normalizer_group.add_argument(
        "--normalize_tts_safe_max_chars",
        default=180,
        type=int,
        help="Maximum characters per sentence for the deterministic tts_safe_split normalizer (default: 180).",
    )
    normalizer_group.add_argument(
        "--normalize_pronunciation_exceptions_file",
        help=(
            "Optional UTF-8 file with per-line pronunciation overrides in the form "
            "'source==replacement'. Legacy alias for --normalize_tts_pronunciation_overrides_file."
        ),
    )
    normalizer_group.add_argument(
        "--normalize_tts_pronunciation_overrides_file",
        help=(
            "Optional UTF-8 file with per-line XTTS pronunciation overrides in the form "
            "'source==replacement'. Use this with tts_pronunciation_overrides."
        ),
    )
    normalizer_group.add_argument(
        "--normalize_pronunciation_lexicon_db",
        help=(
            "Optional SQLite path for the shared pronunciation/stress lexicon. "
            "If omitted, a cached project-local database is created automatically."
        ),
    )
    normalizer_group.add_argument(
        "--normalize_stress_exceptions_file",
        help=(
            "Optional UTF-8 file with per-line stress overrides in the form "
            "'source==replacement'. Use this with stress_words_ru or tsnorm_ru."
        ),
    )
    normalizer_group.add_argument(
        "--normalize_stress_ambiguity_file",
        help=(
            "Optional UTF-8 file with per-line ambiguity variants in the form "
            "'source==variant1|variant2'. Variants may use combining acute accents or "
            "Silero-style plus notation such as 'б+еды|бед+ы'. Use this with stress_ambiguity_llm."
        ),
    )
    normalizer_group.add_argument(
        "--normalize_tsnorm_stress_yo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When tsnorm_ru is enabled, restore or keep 'ё' where the backend can determine it (default: on).",
    )
    normalizer_group.add_argument(
        "--normalize_tsnorm_stress_monosyllabic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When tsnorm_ru is enabled, also add stress marks to monosyllabic words (default: off).",
    )
    normalizer_group.add_argument(
        "--normalize_tsnorm_min_word_length",
        default=2,
        type=int,
        help="Minimum token length for tsnorm_ru stress processing (default: 2).",
    )

    edge_tts_group = parser.add_argument_group(title="edge specific")
    edge_tts_group.add_argument(
        "--voice_rate",
        help="""
            Speaking rate of the text. Valid relative values range from -50%%(--xxx='-50%%') to +100%%. 
            For negative value use format --arg=value,
        """,
    )

    edge_tts_group.add_argument(
        "--voice_volume",
        help="""
            Volume level of the speaking voice. Valid relative values floor to -100%%.
            For negative value use format --arg=value,
        """,
    )

    edge_tts_group.add_argument(
        "--voice_pitch",
        help="""
            Baseline pitch for the text.Valid relative values like -80Hz,+50Hz, pitch changes should be within 0.5 to 1.5 times the original audio.
            For negative value use format --arg=value,
        """,
    )

    edge_tts_group.add_argument(
        "--proxy",
        help="Proxy server for the TTS provider. Format: http://[username:password@]proxy.server:port",
    )

    azure_edge_tts_group = parser.add_argument_group(title="azure/edge specific")
    azure_edge_tts_group.add_argument(
        "--break_duration",
        default="1250",
        help="Break duration in milliseconds for the different paragraphs or sections (default: 1250, means 1.25 s). Valid values range from 0 to 5000 milliseconds for Azure TTS.",
    )

    piper_tts_group = parser.add_argument_group(title="piper specific")
    piper_tts_group.add_argument(
        "--piper_path",
        default="piper",
        help="Path to the Piper TTS executable",
    )
    piper_tts_group.add_argument(
        "--piper_docker_image",
        default="lscr.io/linuxserver/piper:latest",
        help="Piper Docker image name (if using Docker)",
    )
    piper_tts_group.add_argument(
        "--piper_speaker",
        default=0,
        help="Piper speaker id, used for multi-speaker models",
    )
    piper_tts_group.add_argument(
        "--piper_sentence_silence",
        default=0.2,
        help="Seconds of silence after each sentence",
    )
    piper_tts_group.add_argument(
        "--piper_length_scale",
        default=1.0,
        help="Phoneme length, a.k.a. speaking rate",
    )

    args = parser.parse_args()
    return GeneralConfig(args)


def main(config=None, log_file=None):
    if not config: # config passed from UI, or uses args if CLI
        config = handle_args()

    if log_file:
        # If log_file is provided (e.g., from UI), use it directly as a Path object.
        # The UI passes an absolute path string.
        effective_log_file = Path(log_file)
    else:
        # Otherwise (e.g., CLI usage without a specific log file from UI),
        # keep logs inside the selected output folder so each run stays self-contained.
        base_dir = Path(config.output_folder) if getattr(config, "output_folder", None) else None
        effective_log_file = generate_unique_log_path("EtA", base_dir=base_dir)
    
    # Ensure config.log_file is updated, as it's used by AudiobookGenerator for worker processes.
    config.log_file = effective_log_file

    setup_logging(config.log, str(effective_log_file))

    AudiobookGenerator(config).run()


if __name__ == "__main__":
    main()
