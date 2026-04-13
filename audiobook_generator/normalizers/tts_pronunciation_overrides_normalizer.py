from __future__ import annotations

import logging
import re

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer
from audiobook_generator.normalizers.ru_text_utils import (
    is_russian_language,
    load_mapping_file,
    preserve_case,
)

logger = logging.getLogger(__name__)

BUILTIN_TTS_PRONUNCIATION_OVERRIDES = {
    "отель": "отэль",
    "отеля": "отэля",
    "отелю": "отэлю",
    "отелем": "отэлем",
    "отеле": "отэле",
    "отели": "отэли",
    "отелей": "отэлей",
    "отелям": "отэлям",
    "отелями": "отэлями",
    "отелях": "отэлях",
}


class TTSPronunciationOverridesNormalizer(BaseNormalizer):
    STEP_NAME = "tts_pronunciation_overrides"

    def __init__(self, config: GeneralConfig):
        self.replacements = BUILTIN_TTS_PRONUNCIATION_OVERRIDES.copy()
        self.replacements.update(
            {
                source.lower(): replacement
                for source, replacement in load_mapping_file(
                    config.normalize_tts_pronunciation_overrides_file
                ).items()
            }
        )
        super().__init__(config)

    def validate_config(self):
        return None

    def normalize(self, text: str, chapter_title: str = "") -> str:
        if not is_russian_language(self.config.language):
            logger.info(
                "tts_pronunciation_overrides skipped for chapter '%s' because language is '%s'",
                chapter_title,
                self.config.language,
            )
            return text

        normalized = text
        replacements = 0
        for source, replacement in sorted(self.replacements.items(), key=lambda item: len(item[0]), reverse=True):
            pattern = self._build_pattern(source)
            normalized, count = pattern.subn(
                lambda match: preserve_case(match.group(0), replacement),
                normalized,
            )
            replacements += count

        logger.info(
            "tts_pronunciation_overrides normalizer applied to chapter '%s': %s replacements",
            chapter_title,
            replacements,
        )
        return normalized

    def _build_pattern(self, source: str) -> re.Pattern[str]:
        escaped = re.escape(source)
        if re.fullmatch(r"[а-яё-]+", source, re.IGNORECASE):
            return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)
        return re.compile(escaped, re.IGNORECASE)
