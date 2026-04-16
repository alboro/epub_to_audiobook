"""
TTS Stress Paradox Guard
========================
Some words have a "stress paradox": adding a stress mark makes the TTS server
read the word *worse* than without any mark. For example, placing an accent on
"Томас" results in the TTS reading "Том**А**с" — an ugly artefact.

This service keeps a list of such words and provides two functions:
1. `filter_for_llm(candidates)` — remove paradox words from LLM stress candidates
   so the LLM normalizer never wastes tokens on them.
2. `strip_paradox_stress(text)` — final post-processing pass that removes stress
   marks from all inflected forms of the paradox words.

Extend the built-in list by subclassing or by passing extra_words to the constructor.
"""
from __future__ import annotations

import re
import logging
from typing import Iterable

logger = logging.getLogger(__name__)

# Words that should NEVER receive stress marks before TTS.
# The TTS server reads them correctly as-is, but mispronounces them when stressed.
# Include the base form; inflected forms are matched by a loose suffix pattern.
_BUILTIN_PARADOX_WORDS: list[str] = [
    "томас",
    "томаса",
    "томасу",
    "томасом",
    "томасе",
    # Add more as discovered
]

# Combining acute accent (U+0301) used by this project for stress marks
_STRESS_MARK = "\u0301"


class TTSStressParadoxGuard:
    """Tracks words that must not receive stress marks before TTS."""

    def __init__(self, extra_words: Iterable[str] = ()):
        self._words: set[str] = {w.lower() for w in _BUILTIN_PARADOX_WORDS}
        self._words.update(w.lower() for w in extra_words)

    def add_words(self, words: Iterable[str]) -> None:
        self._words.update(w.lower() for w in words)

    def is_paradox_word(self, word: str) -> bool:
        return word.lower().replace(_STRESS_MARK, "") in self._words

    def filter_candidates(self, candidates: dict[str, list[str]]) -> dict[str, list[str]]:
        """Remove paradox words from a {word: [variant1, variant2]} LLM candidate dict."""
        filtered = {}
        for word, variants in candidates.items():
            clean = word.lower().replace(_STRESS_MARK, "")
            if clean in self._words:
                logger.debug("Stress paradox guard: skipping LLM for '%s'", word)
            else:
                filtered[word] = variants
        return filtered

    def strip_stress_from_paradox_words(self, text: str) -> str:
        """Remove stress marks from all paradox words in the text."""
        result = text
        for word in sorted(self._words, key=len, reverse=True):
            # Match the word (case-insensitive) possibly with stress mark after any vowel
            pattern = re.compile(
                re.escape(word).replace(r"а", r"а\u0301?")
                .replace(r"е", r"е\u0301?")
                .replace(r"и", r"и\u0301?")
                .replace(r"о", r"о\u0301?")
                .replace(r"у", r"у\u0301?")
                .replace(r"ы", r"ы\u0301?")
                .replace(r"э", r"э\u0301?")
                .replace(r"ю", r"ю\u0301?")
                .replace(r"я", r"я\u0301?"),
                re.IGNORECASE | re.UNICODE,
            )
            # Actually, simpler: just strip U+0301 from any occurrence of the base word
            # Find all case-insensitive occurrences and strip stress marks
            result = _strip_stress_from_word_occurrences(result, word)
        return result


def _strip_stress_from_word_occurrences(text: str, base_word: str) -> str:
    """Strip U+0301 from any occurrence (ignoring existing stress marks) of base_word."""
    # Build pattern that matches base_word allowing optional U+0301 after any char
    escaped = ""
    for ch in base_word:
        escaped += re.escape(ch) + "\u0301?"
    pattern = re.compile(escaped, re.IGNORECASE | re.UNICODE)

    def _strip(m: re.Match) -> str:
        return m.group(0).replace(_STRESS_MARK, "")

    return pattern.sub(_strip, text)


# Singleton
_guard: TTSStressParadoxGuard | None = None


def get_paradox_guard() -> TTSStressParadoxGuard:
    global _guard
    if _guard is None:
        _guard = TTSStressParadoxGuard()
    return _guard

