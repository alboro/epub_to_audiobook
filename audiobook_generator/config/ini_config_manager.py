"""INI config manager for epub_to_audiobook.

Provides read/write support for INI-format configuration files using Python's
built-in configparser.  All setting definitions live here so they are not
duplicated across main.py, general_config.py, and other layers.

Merge priority when loading: CLI args > INI file > built-in defaults.
"""

from __future__ import annotations

import configparser
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical section assignments for every config field.
# This is the single source of truth for which section a setting belongs to.
# ---------------------------------------------------------------------------
FIELD_SECTIONS: Dict[str, str] = {
    # [general] ----------------------------------------------------------------
    "mode": "general",
    "language": "general",
    "log": "general",
    "no_prompt": "general",
    "worker_count": "general",
    "use_pydub_merge": "general",
    "chapter_start": "general",
    "chapter_end": "general",
    "newline_mode": "general",
    "title_mode": "general",
    "chapter_mode": "general",
    "remove_endnotes": "general",
    "remove_reference_numbers": "general",
    "search_and_replace_file": "general",
    "preview": "general",
    "output_text": "general",
    "prepare_text": "general",
    "prepared_text_folder": "general",
    # [tts] --------------------------------------------------------------------
    "tts": "tts",
    "voice_name": "tts",
    "model_name": "tts",
    "output_format": "tts",
    "speed": "tts",
    "instructions": "tts",
    # [tts.openai] -------------------------------------------------------------
    "openai_api_key": "tts.openai",
    "openai_base_url": "tts.openai",
    "openai_max_chars": "tts.openai",
    "openai_enable_polling": "tts.openai",
    "openai_submit_url": "tts.openai",
    "openai_status_url_template": "tts.openai",
    "openai_download_url_template": "tts.openai",
    "openai_job_id_path": "tts.openai",
    "openai_job_status_path": "tts.openai",
    "openai_job_download_url_path": "tts.openai",
    "openai_job_done_values": "tts.openai",
    "openai_job_failed_values": "tts.openai",
    "openai_poll_interval": "tts.openai",
    "openai_poll_timeout": "tts.openai",
    "openai_poll_request_timeout": "tts.openai",
    "openai_poll_max_errors": "tts.openai",
    # [tts.azure_edge] ---------------------------------------------------------
    "break_duration": "tts.azure_edge",
    # [tts.edge] ---------------------------------------------------------------
    "voice_rate": "tts.edge",
    "voice_volume": "tts.edge",
    "voice_pitch": "tts.edge",
    "proxy": "tts.edge",
    # [tts.piper] --------------------------------------------------------------
    "piper_path": "tts.piper",
    "piper_docker_image": "tts.piper",
    "piper_speaker": "tts.piper",
    "piper_noise_scale": "tts.piper",
    "piper_noise_w_scale": "tts.piper",
    "piper_length_scale": "tts.piper",
    "piper_sentence_silence": "tts.piper",
    # [tts.qwen] ---------------------------------------------------------------
    "qwen_api_key": "tts.qwen",
    "qwen_language_type": "tts.qwen",
    "qwen_stream": "tts.qwen",
    "qwen_request_timeout": "tts.qwen",
    # [tts.gemini] -------------------------------------------------------------
    "gemini_api_key": "tts.gemini",
    "gemini_sample_rate": "tts.gemini",
    "gemini_channels": "tts.gemini",
    "gemini_audio_encoding": "tts.gemini",
    "gemini_temperature": "tts.gemini",
    "gemini_speaker_map": "tts.gemini",
    # [tts.kokoro] -------------------------------------------------------------
    "kokoro_base_url": "tts.kokoro",
    "kokoro_volume_multiplier": "tts.kokoro",
    # [normalize] --------------------------------------------------------------
    "normalize": "normalize",
    "normalize_steps": "normalize",
    "normalize_provider": "normalize",
    "normalize_model": "normalize",
    "normalize_api_key": "normalize",
    "normalize_base_url": "normalize",
    "normalize_max_chars": "normalize",
    "normalize_system_prompt_file": "normalize",
    "normalize_user_prompt_file": "normalize",
    "normalize_tts_safe_max_chars": "normalize",
    "normalize_tts_pronunciation_overrides_file": "normalize",
    "normalize_pronunciation_lexicon_db": "normalize",
    "normalize_stress_exceptions_file": "normalize",
    "normalize_stress_ambiguity_file": "normalize",
    "normalize_tsnorm_stress_yo": "normalize",
    "normalize_tsnorm_stress_monosyllabic": "normalize",
    "normalize_tsnorm_min_word_length": "normalize",
    # [m4b] --------------------------------------------------------------------
    "package_m4b": "m4b",
    "chunked_audio": "m4b",
    "m4b_filename": "m4b",
    "m4b_bitrate": "m4b",
    "ffmpeg_path": "m4b",
}

# Fields that are boolean flags (no value, just presence).
BOOL_FIELDS = {
    "no_prompt", "use_pydub_merge", "preview", "output_text", "prepare_text",
    "normalize", "remove_endnotes", "remove_reference_numbers",
    "openai_enable_polling",
    "normalize_tsnorm_stress_yo", "normalize_tsnorm_stress_monosyllabic",
    "qwen_stream", "package_m4b",
}


def load_ini(path: str | Path) -> Dict[str, Any]:
    """Parse an INI file and return a flat {field_name: value} dict.

    Section names are stripped; field names must be unique across sections
    (which they are, by design in FIELD_SECTIONS).
    """
    cp = configparser.ConfigParser(interpolation=None)
    cp.read(str(path), encoding="utf-8")
    result: Dict[str, Any] = {}
    for section in cp.sections():
        for key, raw_value in cp.items(section):
            result[key] = raw_value
    return result


def save_ini(path: str | Path, config, *, include_paths: bool = True) -> None:
    """Write a GeneralConfig snapshot to an INI file.

    Only non-None, non-False fields are written to keep the file readable.
    Sensitive fields like API keys are included but the file is local/private.
    """
    cp = configparser.ConfigParser(interpolation=None)

    # Always write resolved paths at the top.
    if include_paths:
        cp.add_section("paths")
        if getattr(config, "input_file", None):
            cp.set("paths", "input_file", str(config.input_file))
        if getattr(config, "output_folder", None):
            cp.set("paths", "output_folder", str(config.output_folder))

    for field, section in FIELD_SECTIONS.items():
        value = getattr(config, field, None)
        if value is None:
            continue
        if value is False and field in BOOL_FIELDS:
            continue  # skip False flags to reduce noise
        if not cp.has_section(section):
            cp.add_section(section)
        cp.set(section, field, str(value))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as fh:
        cp.write(fh)
    logger.debug("Config snapshot written to %s", path)


def merge_ini_into_args(args: Any, ini_values: Dict[str, Any]) -> None:
    """Merge INI-loaded values into an argparse Namespace (in-place).

    CLI-provided values take priority: a field is only set from INI if the
    current argparse value is None (i.e. the user did not pass it on the CLI).
    """
    for key, raw_value in ini_values.items():
        current = getattr(args, key, None)
        if current is not None:
            continue  # CLI wins
        # Coerce boolean strings
        if isinstance(raw_value, str):
            if raw_value.lower() in ("true", "yes", "1"):
                setattr(args, key, True)
            elif raw_value.lower() in ("false", "no", "0"):
                setattr(args, key, False)
            else:
                setattr(args, key, raw_value)
        else:
            setattr(args, key, raw_value)

