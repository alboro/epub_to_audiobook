import logging

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer

logger = logging.getLogger(__name__)

SYMBOL_REPLACEMENTS = {
    "\u00ab": '"',
    "\u00bb": '"',
    "\u2014": "-",
    "\u2026": "...",
}


class SimpleSymbolsNormalizer(BaseNormalizer):
    STEP_NAME = "simple_symbols"

    def __init__(self, config: GeneralConfig):
        super().__init__(config)

    def validate_config(self):
        return None

    def normalize(self, text: str, chapter_title: str = "") -> str:
        normalized = text
        replacement_count = 0
        for source, target in SYMBOL_REPLACEMENTS.items():
            count = normalized.count(source)
            if count:
                normalized = normalized.replace(source, target)
                replacement_count += count

        logger.info(
            "Simple symbols normalizer applied to chapter '%s': %s replacements",
            chapter_title,
            replacement_count,
        )
        return normalized
