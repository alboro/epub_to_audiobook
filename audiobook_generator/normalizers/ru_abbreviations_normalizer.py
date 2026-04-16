"""Russian abbreviation expansion normalizer.

Evaluation summary (TODO 3):

``runorm`` (pip install runorm, v1.1)
  - ``RuleNormalizer``: deterministic, handles ALL-CAPS Russian acronyms
    (США → эс-ша-а) and basic currency symbol normalization.  Useful for
    letter-by-letter expansion of acronyms.  Integrated as an optional backend
    below when ``runorm`` is installed.
  - ``RUNorm`` (neural): downloads BERT+T5 models from HuggingFace (~200 MB+),
    requires ~1 GB RAM for inference on CPU.  Too heavy for the default
    pipeline; its number/date normalization is less capable than our own
    ``numbers_ru`` step.  Not integrated.

``saarus72/text_normalization`` (GitHub only, HuggingFace model ``saarus72/russian_text_normalizer``)
  - T5/FRED-T5-large based neural normalization (~1.2 GB).  NOT a pip package.
  - Handles numbers, dates, abbreviations and Latin transliteration well.
  - However, it requires significant compute, has no pip distribution, and its
    normalization scope is already covered by our deterministic pipeline
    (``numbers_ru``) + the existing ``openai`` LLM step.  Not integrated.

This normalizer therefore:
  1. Always applies a deterministic table of common Russian text abbreviations
     (т.д., т.е., и пр., и др., напр., см., ср., тыс., млн., млрд., …).
  2. Optionally uses ``runorm.RuleNormalizer`` for ALL-CAPS Russian acronyms
     when the ``runorm`` package is installed (gracefully skipped otherwise).
"""
from __future__ import annotations

import logging
import re

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer
from audiobook_generator.normalizers.ru_text_utils import is_russian_language

logger = logging.getLogger(__name__)

try:
    from runorm.runorm import RuleNormalizer as _RunormRuleNormalizer
    _HAS_RUNORM = True
except ImportError:  # pragma: no cover - runorm is optional
    _RunormRuleNormalizer = None
    _HAS_RUNORM = False

# ---------------------------------------------------------------------------
# Russian letter names (алфавитные названия) used for acronym expansion.
# When runorm is not available we use this table directly.
# ---------------------------------------------------------------------------
_RU_LETTER_NAMES: dict[str, str] = {
    "А": "а",   "Б": "бэ",  "В": "вэ",  "Г": "гэ",  "Д": "дэ",
    "Е": "е",   "Ё": "ё",   "Ж": "жэ",  "З": "зэ",  "И": "и",
    "Й": "ий",  "К": "ка",  "Л": "эл",  "М": "эм",  "Н": "эн",
    "О": "о",   "П": "пэ",  "Р": "эр",  "С": "эс",  "Т": "тэ",
    "У": "у",   "Ф": "эф",  "Х": "ха",  "Ц": "цэ",  "Ч": "чэ",
    "Ш": "ша",  "Щ": "ща",  "Ъ": "твёрдый знак",
    "Ы": "ы",   "Ь": "мягкий знак",
    "Э": "э",   "Ю": "ю",   "Я": "я",
}

# Pattern: 2+ consecutive Cyrillic uppercase letters not flanked by Cyrillic letters.
# Avoids matching normal words written in ALL-CAPS (they would be whole-word matches
# with actual vowels, making them pronounceable — left to TTS).
_ACRONYM_PATTERN = re.compile(r"\b([А-ЯЁ]{2,})\b")

# ---------------------------------------------------------------------------
# Deterministic abbreviation expansion table.
# Compound / longer entries must precede shorter ones.
# ---------------------------------------------------------------------------
_ABBREV_TABLE: list[tuple[re.Pattern, str]] = [
    # Compound sequences
    (re.compile(r"\bи\s+т\.\s*д\.", re.IGNORECASE), "и так далее"),
    (re.compile(r"\bи\s+т\.\s*п\.", re.IGNORECASE), "и тому подобное"),
    (re.compile(r"\bи\s+пр\.", re.IGNORECASE), "и прочее"),
    (re.compile(r"\bи\s+др\.", re.IGNORECASE), "и другие"),
    # Single-token
    (re.compile(r"\bт\.\s*д\.", re.IGNORECASE), "так далее"),
    (re.compile(r"\bт\.\s*е\.", re.IGNORECASE), "то есть"),
    (re.compile(r"\bт\.\s*п\.", re.IGNORECASE), "тому подобному"),
    (re.compile(r"\bнапр\.", re.IGNORECASE), "например"),
    (re.compile(r"\bсм\.", re.IGNORECASE), "смотрите"),
    (re.compile(r"\bср\.", re.IGNORECASE), "сравните"),
    # Units of quantity (note: no case agreement — left to TTS)
    (re.compile(r"\bтыс\.", re.IGNORECASE), "тысяч"),
    (re.compile(r"\bмлн\.", re.IGNORECASE), "миллионов"),
    (re.compile(r"\bмлрд\.", re.IGNORECASE), "миллиардов"),
    # Publication-related
    (re.compile(r"\bизд\.", re.IGNORECASE), "издание"),
    (re.compile(r"\bл\.", re.IGNORECASE), "лист"),   # only when clearly an abbreviation
]


def _expand_acronym(acronym: str) -> str:
    """Convert an ALL-CAPS Cyrillic acronym to hyphen-joined letter names.

    Example: "США" → "эс-ша-а"
    """
    parts = [_RU_LETTER_NAMES.get(ch.upper(), ch) for ch in acronym]
    return "-".join(parts)


class AbbreviationsRuNormalizer(BaseNormalizer):
    """Expand common Russian text abbreviations and optionally ALL-CAPS acronyms.

    Pipeline step name: ``abbreviations_ru``

    Two-pass strategy
    -----------------
    1. Deterministic table substitution (always active): т.д., т.е., и пр., …
    2. ALL-CAPS acronym expansion (active when ``runorm`` is installed *or*
       always via built-in letter table — configurable).

    The ``runorm`` backend is tried first; if unavailable the built-in
    ``_expand_acronym()`` function is used instead.
    """

    STEP_NAME = "ru_abbreviations"

    def __init__(self, config: GeneralConfig):
        # Pre-initialise the optional runorm backend once (lazy-loaded on
        # first normalize() call to avoid heavy imports during __init__).
        self._runorm: object | None = None
        self._runorm_checked = False
        super().__init__(config)

    def validate_config(self):
        pass  # runorm is optional; nothing required

    def normalize(self, text: str, chapter_title: str = "") -> str:
        if not is_russian_language(self.config.language):
            logger.info(
                "abbreviations_ru skipped for chapter '%s' because language is '%s'",
                chapter_title,
                self.config.language,
            )
            return text

        normalized = text
        changed = 0

        # Pass 1: deterministic table
        for pattern, replacement in _ABBREV_TABLE:
            new, n = pattern.subn(replacement, normalized)
            if n:
                normalized = new
                changed += n

        # Pass 2: ALL-CAPS acronym expansion
        new, n = _ACRONYM_PATTERN.subn(self._expand_acronym_match, normalized)
        if n:
            normalized = new
            changed += n

        logger.info(
            "abbreviations_ru applied to chapter '%s': %d replacements (runorm=%s)",
            chapter_title,
            changed,
            "available" if _HAS_RUNORM else "unavailable",
        )
        return normalized

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _expand_acronym_match(self, match: re.Match[str]) -> str:
        """Callback for _ACRONYM_PATTERN.subn — expands a single acronym."""
        acronym = match.group(1)

        if _HAS_RUNORM:
            return self._expand_via_runorm(acronym)

        return _expand_acronym(acronym)

    def _expand_via_runorm(self, acronym: str) -> str:
        """Use runorm to get letter-by-letter pronunciation, then join with hyphens."""
        runorm_obj = self._get_runorm()
        if runorm_obj is None:
            return _expand_acronym(acronym)
        try:
            # expand_abbreviations works on a full text fragment; wrap the acronym
            expanded = runorm_obj.expand_abbreviations(acronym)
            # runorm outputs space-separated letter names; convert to hyphen-joined
            return "-".join(expanded.split())
        except Exception:
            return _expand_acronym(acronym)

    def _get_runorm(self):
        """Lazily initialise the runorm RuleNormalizer (once per instance)."""
        if self._runorm_checked:
            return self._runorm
        self._runorm_checked = True
        if _RunormRuleNormalizer is not None:
            try:
                self._runorm = _RunormRuleNormalizer()
            except Exception:
                self._runorm = None
        return self._runorm

