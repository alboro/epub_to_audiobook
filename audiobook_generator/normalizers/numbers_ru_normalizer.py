from __future__ import annotations

import logging
import re

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer
from audiobook_generator.normalizers.ru_text_utils import is_russian_language

logger = logging.getLogger(__name__)

try:
    from num2words import num2words
except ImportError:  # pragma: no cover - dependency validation handles this at runtime
    num2words = None

try:
    from pymorphy3 import MorphAnalyzer
except ImportError:  # pragma: no cover - dependency validation handles this at runtime
    MorphAnalyzer = None


LIST_ITEM_PATTERN = re.compile(r"(^|\n)(\s*)(\d+)\.(?=\s)", re.MULTILINE)
NUMBER_SIGN_PATTERN = re.compile(r"№\s*(\d+)")
DECIMAL_PATTERN = re.compile(r"(?<![\w/])(-?\d+[.,]\d+)(?![\w/])")
RANGE_PATTERN = re.compile(r"(?<![\w/])(\d+)\s*([–—-])\s*(\d+)(?![\w/])")
ORDINAL_PATTERN = re.compile(
    r"(?<![\w/])(?P<number>-?\d+)(?P<sep>-?)(?P<suffix>ый|ий|ой|й|ая|я|ое|ее|е|ого|го|ому|му|ыми|ими|ых|их|ую|ю|ом|ем|ым|им|ые|ие)(?![\w/])",
    re.IGNORECASE,
)
CARDINAL_WITH_NOUN_PATTERN = re.compile(
    r"(?<![\w/])(?P<number>[12])\s+(?P<noun>[А-Яа-яЁё-]+)"
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
    "часть": ("f", "n"),
    "части": ("f", "g"),
    "частью": ("f", "i"),
}
ORDINAL_NOUN_PATTERN = re.compile(
    r"\b([IVXLCDM]+|\d+)\s+(" + "|".join(re.escape(item) for item in sorted(ORDINAL_NOUN_FORMS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

ALLOWED_CARDINAL_NOUN_LEMMAS = {
    "глава",
    "том",
    "раздел",
    "пункт",
    "страница",
    "книга",
    "часть",
    "номер",
}

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
PYMORPHY_GENDER_MAP = {"masc": "m", "femn": "f", "neut": "n"}
PYMORPHY_CASE_MAP = {
    "nomn": "n",
    "gent": "g",
    "datv": "d",
    "accs": "a",
    "ablt": "i",
    "loct": "p",
    "loc2": "p",
}
NUMBER_WITH_NOUN_FORMS = {
    1: {
        ("n", "m"): "один",
        ("n", "f"): "одна",
        ("n", "n"): "одно",
        ("g", "m"): "одного",
        ("g", "f"): "одной",
        ("g", "n"): "одного",
        ("d", "m"): "одному",
        ("d", "f"): "одной",
        ("d", "n"): "одному",
        ("a", "m"): "один",
        ("a", "f"): "одну",
        ("a", "n"): "одно",
        ("i", "m"): "одним",
        ("i", "f"): "одной",
        ("i", "n"): "одним",
        ("p", "m"): "одном",
        ("p", "f"): "одной",
        ("p", "n"): "одном",
    },
    2: {
        ("n", "m"): "два",
        ("n", "f"): "две",
        ("n", "n"): "два",
        ("g", "m"): "двух",
        ("g", "f"): "двух",
        ("g", "n"): "двух",
        ("d", "m"): "двум",
        ("d", "f"): "двум",
        ("d", "n"): "двум",
        ("a", "m"): "два",
        ("a", "f"): "две",
        ("a", "n"): "два",
        ("i", "m"): "двумя",
        ("i", "f"): "двумя",
        ("i", "n"): "двумя",
        ("p", "m"): "двух",
        ("p", "f"): "двух",
        ("p", "n"): "двух",
    },
}


class NumbersRuNormalizer(BaseNormalizer):
    STEP_NAME = "numbers_ru"

    def __init__(self, config: GeneralConfig):
        self.morph = MorphAnalyzer() if MorphAnalyzer is not None else None
        super().__init__(config)

    def validate_config(self):
        if num2words is None:
            raise ImportError(
                "numbers_ru requires the 'num2words' package. Install dependencies again after updating requirements.txt."
            )

    def normalize(self, text: str, chapter_title: str = "") -> str:
        if not is_russian_language(self.config.language):
            logger.info(
                "numbers_ru skipped for chapter '%s' because language is '%s'",
                chapter_title,
                self.config.language,
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
        if self.morph is not None:
            normalized, count = CARDINAL_WITH_NOUN_PATTERN.subn(self._replace_cardinal_with_noun, normalized)
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
        if raw_number in {"1", "2"} and self._best_noun_parse(noun) is not None:
            return match.group(0)

        noun_config = ORDINAL_NOUN_FORMS.get(noun.lower())
        if noun_config is None:
            return match.group(0)

        number = self._parse_number_token(raw_number)
        if number is None:
            return match.group(0)

        gender, case = noun_config
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

    def _replace_cardinal_with_noun(self, match: re.Match[str]) -> str:
        raw_number = match.group("number")
        noun = match.group("noun")
        if self.morph is None:
            return match.group(0)

        parse = self._best_noun_parse(noun)
        if parse is None:
            return match.group(0)

        spoken = self._to_agreed_cardinal(int(raw_number), parse)
        return f"{spoken} {noun}"

    def _replace_cardinal(self, match: re.Match[str]) -> str:
        raw_number = match.group(1)
        return self._to_words(int(raw_number))

    def _best_noun_parse(self, noun: str):
        for parse in self.morph.parse(noun):
            if "NOUN" not in parse.tag:
                continue
            if parse.normal_form not in ALLOWED_CARDINAL_NOUN_LEMMAS:
                continue
            return parse
        return None

    def _to_agreed_cardinal(self, number: int, noun_parse) -> str:
        gender = PYMORPHY_GENDER_MAP.get(noun_parse.tag.gender)
        if gender is None and "ms-f" in str(noun_parse.tag):
            gender = "f"
        case = PYMORPHY_CASE_MAP.get(noun_parse.tag.case)
        direct_form = NUMBER_WITH_NOUN_FORMS.get(number, {}).get((case or "n", gender or "m"))
        if direct_form:
            return direct_form

        kwargs = {}
        if gender:
            kwargs["gender"] = gender
        if case:
            kwargs["case"] = case
        try:
            return self._to_words(number, **kwargs)
        except TypeError:
            kwargs.pop("case", None)
            return self._to_words(number, **kwargs)

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
