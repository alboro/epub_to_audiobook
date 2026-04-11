from __future__ import annotations

import logging
import re

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer

logger = logging.getLogger(__name__)

try:
    from num2words import num2words
except ImportError:  # pragma: no cover - dependency validation handles this at runtime
    num2words = None


LIST_ITEM_PATTERN = re.compile(r"(^|\n)(\s*)(\d+)\.(?=\s)", re.MULTILINE)
NUMBER_SIGN_PATTERN = re.compile(r"№\s*(\d+)")
DECIMAL_PATTERN = re.compile(r"(?<![\w/])(-?\d+[.,]\d+)(?![\w/])")
RANGE_PATTERN = re.compile(r"(?<![\w/])(\d+)\s*([–—-])\s*(\d+)(?![\w/])")
ORDINAL_PATTERN = re.compile(
    r"(?<![\w/])(?P<number>-?\d+)(?P<sep>-?)(?P<suffix>ый|ий|ой|й|ая|я|ое|ее|е|ого|го|ому|му|ыми|ими|ых|их|ую|ю|ом|ем|ым|им|ые|ие)(?![\w/])",
    re.IGNORECASE,
)
CARDINAL_PATTERN = re.compile(r"(?<![\w/])(-?\d+)(?![\w/])")

ORDINAL_NOUN_FORMS = {
    "век": ("m", "n"),
    "века": ("m", "g"),
    "веку": ("m", "d"),
    "веком": ("m", "i"),
    "веке": ("m", "p"),
    "столетие": ("n", "n"),
    "столетия": ("n", "g"),
    "столетию": ("n", "d"),
    "столетием": ("n", "i"),
    "столетии": ("n", "p"),
    "глава": ("f", "n"),
    "главы": ("f", "g"),
    "главе": ("f", "d"),
    "главу": ("f", "a"),
    "главой": ("f", "i"),
    "главою": ("f", "i"),
    "том": ("m", "n"),
    "тома": ("m", "g"),
    "тому": ("m", "d"),
    "томом": ("m", "i"),
    "томе": ("m", "p"),
    "раздел": ("m", "n"),
    "раздела": ("m", "g"),
    "разделу": ("m", "d"),
    "разделом": ("m", "i"),
    "разделе": ("m", "p"),
    "пункт": ("m", "n"),
    "пункта": ("m", "g"),
    "пункту": ("m", "d"),
    "пунктом": ("m", "i"),
    "пункте": ("m", "p"),
    "страница": ("f", "n"),
    "страницы": ("f", "g"),
    "странице": ("f", "d"),
    "страницу": ("f", "a"),
    "страницей": ("f", "i"),
    "страницею": ("f", "i"),
    "книга": ("f", "n"),
    "книги": ("f", "g"),
    "книге": ("f", "d"),
    "книгу": ("f", "a"),
    "книгой": ("f", "i"),
    "книгою": ("f", "i"),
    "номер": ("m", "n"),
    "номера": ("m", "g"),
    "номеру": ("m", "d"),
    "номером": ("m", "i"),
    "номере": ("m", "p"),
}
ORDINAL_NOUN_PATTERN = re.compile(
    r"\b([IVXLCDM]+|\d+)\s+(" + "|".join(re.escape(item) for item in sorted(ORDINAL_NOUN_FORMS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

ORDINAL_SUFFIX_CONFIG = {
    "й": {"to": "ordinal", "case": "n", "gender": "m"},
    "ый": {"to": "ordinal", "case": "n", "gender": "m"},
    "ий": {"to": "ordinal", "case": "n", "gender": "m"},
    "ой": {"to": "ordinal", "case": "n", "gender": "m"},
    "ая": {"to": "ordinal", "case": "n", "gender": "f"},
    "я": {"to": "ordinal", "case": "n", "gender": "f"},
    "ое": {"to": "ordinal", "case": "n", "gender": "n"},
    "ее": {"to": "ordinal", "case": "n", "gender": "n"},
    "е": {"to": "ordinal", "case": "n", "gender": "n"},
    "ого": {"to": "ordinal", "case": "g", "gender": "m"},
    "го": {"to": "ordinal", "case": "g", "gender": "m"},
    "ому": {"to": "ordinal", "case": "d", "gender": "m"},
    "му": {"to": "ordinal", "case": "d", "gender": "m"},
    "ым": {"to": "ordinal", "case": "i", "gender": "m"},
    "им": {"to": "ordinal", "case": "i", "gender": "m"},
    "ом": {"to": "ordinal", "case": "p", "gender": "m"},
    "ем": {"to": "ordinal", "case": "p", "gender": "m"},
    "ую": {"to": "ordinal", "case": "a", "gender": "f"},
    "ю": {"to": "ordinal", "case": "a", "gender": "f"},
    "ые": {"to": "ordinal", "case": "n", "plural": True},
    "ие": {"to": "ordinal", "case": "n", "plural": True},
    "ых": {"to": "ordinal", "case": "g", "plural": True},
    "их": {"to": "ordinal", "case": "g", "plural": True},
    "ыми": {"to": "ordinal", "case": "i", "plural": True},
    "ими": {"to": "ordinal", "case": "i", "plural": True},
}
SAFE_NO_HYPHEN_SUFFIXES = {"й", "я", "е", "ая", "ое", "ее", "го", "му", "ю"}


class NumbersRuNormalizer(BaseNormalizer):
    STEP_NAME = "numbers_ru"

    def __init__(self, config: GeneralConfig):
        super().__init__(config)

    def validate_config(self):
        if num2words is None:
            raise ImportError(
                "numbers_ru requires the 'num2words' package. Install dependencies again after updating requirements.txt."
            )

    def normalize(self, text: str, chapter_title: str = "") -> str:
        language = (self.config.language or "ru").split("-")[0].lower()
        if language != "ru":
            logger.info(
                "numbers_ru skipped for chapter '%s' because language is '%s'",
                chapter_title,
                language,
            )
            return text

        normalized = text
        replacements = 0

        normalized, count = LIST_ITEM_PATTERN.subn(self._replace_list_item, normalized)
        replacements += count
        normalized, count = NUMBER_SIGN_PATTERN.subn(self._replace_number_sign, normalized)
        replacements += count
        normalized, count = ORDINAL_NOUN_PATTERN.subn(self._replace_ordinal_noun, normalized)
        replacements += count
        normalized, count = ORDINAL_PATTERN.subn(self._replace_ordinal, normalized)
        replacements += count
        normalized, count = DECIMAL_PATTERN.subn(self._replace_decimal, normalized)
        replacements += count
        normalized, count = RANGE_PATTERN.subn(self._replace_range, normalized)
        replacements += count
        normalized, count = CARDINAL_PATTERN.subn(self._replace_cardinal, normalized)
        replacements += count

        logger.info(
            "numbers_ru normalizer applied to chapter '%s': %s replacements",
            chapter_title,
            replacements,
        )
        return normalized

    def _replace_list_item(self, match: re.Match[str]) -> str:
        prefix, indent, raw_number = match.groups()
        spoken = self._to_words(int(raw_number), to="ordinal", gender="n")
        return f"{prefix}{indent}{spoken.capitalize()}."

    def _replace_number_sign(self, match: re.Match[str]) -> str:
        raw_number = match.group(1)
        return f"номер {self._to_words(int(raw_number))}"

    def _replace_ordinal_noun(self, match: re.Match[str]) -> str:
        raw_number, noun = match.groups()
        noun_key = noun.lower()
        noun_config = ORDINAL_NOUN_FORMS.get(noun_key)
        if noun_config is None:
            return match.group(0)

        gender, case = noun_config
        number = self._parse_number_token(raw_number)
        if number is None:
            return match.group(0)

        spoken = self._to_words(number, to="ordinal", gender=gender, case=case)
        return f"{spoken} {noun}"

    def _replace_ordinal(self, match: re.Match[str]) -> str:
        raw_number = match.group("number")
        separator = match.group("sep")
        suffix = match.group("suffix").lower()
        if not separator and suffix not in SAFE_NO_HYPHEN_SUFFIXES:
            return match.group(0)

        config = ORDINAL_SUFFIX_CONFIG.get(suffix)
        if config is None:
            return match.group(0)

        return self._to_words(int(raw_number), **config)

    def _replace_decimal(self, match: re.Match[str]) -> str:
        raw_number = match.group(1).replace(",", ".")
        return self._to_words(raw_number)

    def _replace_range(self, match: re.Match[str]) -> str:
        left, dash, right = match.groups()
        return f"{self._to_words(int(left))} {dash} {self._to_words(int(right))}"

    def _replace_cardinal(self, match: re.Match[str]) -> str:
        raw_number = match.group(1)
        return self._to_words(int(raw_number))

    def _parse_number_token(self, raw_number: str) -> int | None:
        if raw_number.isdigit():
            return int(raw_number)
        if re.fullmatch(r"[IVXLCDM]+", raw_number, re.IGNORECASE):
            return self._roman_to_int(raw_number)
        return None

    def _roman_to_int(self, value: str) -> int:
        numerals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
        total = 0
        previous = 0
        for char in reversed(value.upper()):
            current = numerals[char]
            if current < previous:
                total -= current
            else:
                total += current
                previous = current
        return total

    def _to_words(self, value, **kwargs) -> str:
        spoken = num2words(value, lang="ru", **kwargs)
        return re.sub(r"\s+", " ", str(spoken)).strip()
