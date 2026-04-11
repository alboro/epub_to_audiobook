from __future__ import annotations

import logging
import re

from sentencex import segment

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer

logger = logging.getLogger(__name__)

DEFAULT_SAFE_MAX_CHARS = 160
MIN_SPLIT_FRACTION = 0.45
LEFT_TRIM_CHARS = " \t\r\n,;:-–—"
RIGHT_TRIM_CHARS = " \t\r\n,;:-–—"
SENTENCE_END_CHARS = ".!?"

PRIORITY_PATTERNS = (
    re.compile(r"[;:](?=\s|$)"),
    re.compile(r",\s+(?=(?:а также|однако|но|зато|поэтому|причем|притом|при этом|затем|потом)\b)", re.IGNORECASE),
    re.compile(
        r",\s+(?=(?:котор(?:ый|ая|ое|ые|ого|ому|ым|ых|ую|ой|ою)|обосновывающ\w*|существующ\w*|позволяющ\w*|делающ\w*|создающ\w*)\b)",
        re.IGNORECASE,
    ),
    re.compile(r",(?=\s|$)"),
    re.compile(r"\s-\s"),
)


class TTSSafeSplitNormalizer(BaseNormalizer):
    STEP_NAME = "tts_safe_split"

    def __init__(self, config: GeneralConfig):
        self.max_chars = config.normalize_tts_safe_max_chars or DEFAULT_SAFE_MAX_CHARS
        super().__init__(config)

    def validate_config(self):
        if self.max_chars < 40:
            raise ValueError("normalize_tts_safe_max_chars must be at least 40")

    def normalize(self, text: str, chapter_title: str = "") -> str:
        parts = re.split(r"(\n\s*\n+)", text)
        normalized_parts = []
        sentence_count = 0
        inserted_splits = 0

        for part in parts:
            if not part:
                continue
            if re.fullmatch(r"\n\s*\n+", part):
                normalized_parts.append(part)
                continue

            normalized_paragraph, paragraph_sentences, paragraph_splits = self._normalize_paragraph(part)
            normalized_parts.append(normalized_paragraph)
            sentence_count += paragraph_sentences
            inserted_splits += paragraph_splits

        logger.info(
            "TTS safe split normalizer applied to chapter '%s': %s sentence parts, %s inserted splits, max_chars=%s",
            chapter_title,
            sentence_count,
            inserted_splits,
            self.max_chars,
        )
        return "".join(normalized_parts).strip()

    def _normalize_paragraph(self, paragraph: str) -> tuple[str, int, int]:
        compact = re.sub(r"\s+", " ", paragraph).strip()
        if not compact:
            return "", 0, 0

        language = (self.config.language or "ru").split("-")[0].lower()
        sentences = [item.strip() for item in segment(language, compact) if item and item.strip()]
        if not sentences:
            sentences = [compact]

        safe_sentences = []
        inserted_splits = 0
        for sentence in sentences:
            split_parts = self._split_long_sentence(sentence)
            safe_sentences.extend(split_parts)
            inserted_splits += max(0, len(split_parts) - 1)

        return " ".join(safe_sentences).strip(), len(safe_sentences), inserted_splits

    def _split_long_sentence(self, sentence: str) -> list[str]:
        pending = [sentence.strip()]
        result = []

        while pending:
            current = pending.pop(0).strip()
            if not current:
                continue
            if len(current) <= self.max_chars:
                result.append(self._finalize_sentence(current))
                continue

            split_index = self._find_split_index(current)
            if split_index is None:
                result.append(self._finalize_sentence(current))
                continue

            left = current[:split_index].rstrip(LEFT_TRIM_CHARS)
            right = current[split_index:].lstrip(RIGHT_TRIM_CHARS)
            if not left or not right:
                result.append(self._finalize_sentence(current))
                continue

            result.append(self._finalize_sentence(left))
            pending.insert(0, self._normalize_sentence_start(right))

        return result

    def _find_split_index(self, sentence: str) -> int | None:
        if len(sentence) <= self.max_chars:
            return None

        min_index = max(20, int(self.max_chars * MIN_SPLIT_FRACTION))
        window = sentence[: self.max_chars + 1]

        for pattern in PRIORITY_PATTERNS:
            matches = [match for match in pattern.finditer(window) if match.end() >= min_index]
            if matches:
                return matches[-1].end()

        space_index = window.rfind(" ")
        if space_index >= min_index:
            return space_index + 1

        return self.max_chars

    def _normalize_sentence_start(self, sentence: str) -> str:
        sentence = sentence.lstrip()
        if not sentence:
            return ""

        chars = list(sentence)
        for index, char in enumerate(chars):
            if char.isalpha():
                chars[index] = char.upper()
                break
            if char.isdigit():
                break
        return "".join(chars)

    def _finalize_sentence(self, sentence: str) -> str:
        sentence = sentence.strip()
        if not sentence:
            return ""
        if sentence[-1] in SENTENCE_END_CHARS:
            return sentence
        return f"{sentence}."
