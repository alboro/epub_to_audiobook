"""
Tests for TTSStressParadoxGuard.

Covers:
- Loading words from config list (with/without stress marks)
- Inflected forms expansion via pymorphy3
- is_paradox_word() for base and all inflected forms
- apply_paradox_overrides(): strip stress from paradox words
- apply_paradox_overrides(): enforce canonical stress for exact base form
- filter_candidates(): paradox words excluded from LLM candidate dict
"""
from __future__ import annotations

import unittest

from audiobook_generator.normalizers.ru_text_utils import COMBINING_ACUTE, strip_combining_acute
from audiobook_generator.normalizers.ru_tts_stress_paradox_guard import TTSStressParadoxGuard


# Helpers
def stressed(word: str) -> str:
    """Add combining acute after first vowel — for building simple test expectations."""
    vowels = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
    result = list(word)
    for i, ch in enumerate(result):
        if ch in vowels:
            result.insert(i + 1, COMBINING_ACUTE)
            break
    return "".join(result)


class TestParadoxGuardNoStress(unittest.TestCase):
    """Words listed WITHOUT stress → strip stress from all their forms."""

    def setUp(self):
        # "Томас" without stress mark
        self.guard = TTSStressParadoxGuard(["Томас"])

    def test_is_paradox_base_form(self):
        self.assertTrue(self.guard.is_paradox_word("томас"))

    def test_is_paradox_case_insensitive(self):
        self.assertTrue(self.guard.is_paradox_word("Томас"))
        self.assertTrue(self.guard.is_paradox_word("ТОМАС"))

    def test_is_paradox_genitive_form(self):
        # pymorphy3 should produce: томаса, томасу, томасом, томасе, томасы etc.
        self.assertTrue(self.guard.is_paradox_word("томаса"))
        self.assertTrue(self.guard.is_paradox_word("томасу"))

    def test_non_paradox_word(self):
        self.assertFalse(self.guard.is_paradox_word("пушкин"))
        self.assertFalse(self.guard.is_paradox_word("слово"))

    def test_apply_strips_stress_from_base(self):
        # Normalizers put stress on "Томас" → paradox guard removes it
        text = "То" + COMBINING_ACUTE + "мас написал"
        result = self.guard.apply_paradox_overrides(text)
        self.assertNotIn(COMBINING_ACUTE, result.split()[0])
        self.assertIn("Томас", result)

    def test_apply_strips_stress_from_inflected(self):
        # "Томасу" with bogus stress added by normalizer
        text = "Тома" + COMBINING_ACUTE + "су Пейну"
        result = self.guard.apply_paradox_overrides(text)
        self.assertNotIn(COMBINING_ACUTE, result.split()[0])

    def test_apply_preserves_non_paradox_stress(self):
        # Stress on non-paradox word should be untouched
        text = "Том" + COMBINING_ACUTE + "ас и Пу" + COMBINING_ACUTE + "шкин"
        # "Пушкин" is not in the list → its stress preserved
        result = self.guard.apply_paradox_overrides(text)
        self.assertIn("Пу" + COMBINING_ACUTE + "шкин", result)

    def test_apply_preserves_case(self):
        # Capital T must be preserved
        text = "Томасу Пейну"
        result = self.guard.apply_paradox_overrides(text)
        self.assertTrue(result.startswith("Томасу") or result.startswith("томасу"))
        self.assertIn("Т", result[0])  # Capital preserved

    def test_filter_candidates_excludes_paradox(self):
        candidates = {
            "томас": ["То" + COMBINING_ACUTE + "мас", "Тома" + COMBINING_ACUTE + "с"],
            "слово": ["сло" + COMBINING_ACUTE + "во"],
        }
        filtered = self.guard.filter_candidates(candidates)
        self.assertNotIn("томас", filtered)
        self.assertIn("слово", filtered)

    def test_filter_candidates_excludes_inflected_form(self):
        candidates = {
            "томасу": ["Тома" + COMBINING_ACUTE + "су", "То" + COMBINING_ACUTE + "масу"],
            "пейну": ["Пе" + COMBINING_ACUTE + "йну"],
        }
        filtered = self.guard.filter_candidates(candidates)
        self.assertNotIn("томасу", filtered)


class TestParadoxGuardWithCanonicalStress(unittest.TestCase):
    """Words listed WITH stress mark → canonical stress enforced for that form; others stripped."""

    def setUp(self):
        # "Т+омас" means stress on О (+ is placed BEFORE the stressed letter)
        self.guard = TTSStressParadoxGuard(["Т+омас"])

    def test_is_paradox_base_form(self):
        self.assertTrue(self.guard.is_paradox_word("томас"))

    def test_is_paradox_inflected(self):
        self.assertTrue(self.guard.is_paradox_word("томаса"))

    def test_apply_enforces_canonical_stress_on_base(self):
        # If text has wrong stress or no stress on "Томас", guard enforces "То́мас"
        text = "Тома" + COMBINING_ACUTE + "с"  # wrong: stress on А
        result = self.guard.apply_paradox_overrides(text)
        # Should now have stress on О
        self.assertIn("о" + COMBINING_ACUTE, result.lower())

    def test_apply_strips_stress_from_other_forms(self):
        # "Томасу" is an inflected form not directly in config → strip stress
        text = "Тома" + COMBINING_ACUTE + "су"
        result = self.guard.apply_paradox_overrides(text)
        self.assertNotIn(COMBINING_ACUTE, strip_combining_acute(result).replace(result.replace(COMBINING_ACUTE, ""), ""))
        # Simpler assertion:
        self.assertNotIn(COMBINING_ACUTE, result)


class TestParadoxGuardMultipleWords(unittest.TestCase):
    """Multiple words in config."""

    def setUp(self):
        self.guard = TTSStressParadoxGuard(["Томас", "Пейн"])

    def test_both_words_detected(self):
        self.assertTrue(self.guard.is_paradox_word("томас"))
        self.assertTrue(self.guard.is_paradox_word("пейн"))

    def test_apply_strips_both(self):
        text = "То" + COMBINING_ACUTE + "мас Пе" + COMBINING_ACUTE + "йн"
        result = self.guard.apply_paradox_overrides(text)
        # No stress marks should remain on either word
        for word in result.split():
            clean = strip_combining_acute(word).lower().rstrip(".,;:")
            if clean in ("томас", "пейн"):
                self.assertNotIn(COMBINING_ACUTE, word, f"Stress not stripped from '{word}'")


class TestParadoxGuardEmptyList(unittest.TestCase):
    def setUp(self):
        self.guard = TTSStressParadoxGuard([])

    def test_empty_list_no_paradox(self):
        self.assertFalse(self.guard.is_paradox_word("томас"))

    def test_apply_returns_text_unchanged(self):
        text = "То" + COMBINING_ACUTE + "мас"
        self.assertEqual(text, self.guard.apply_paradox_overrides(text))

    def test_filter_candidates_unchanged(self):
        candidates = {"томас": ["То" + COMBINING_ACUTE + "мас"]}
        self.assertEqual(candidates, self.guard.filter_candidates(candidates))


class TestParadoxGuardFromConfig(unittest.TestCase):
    """Test the from_config() factory that parses a comma-separated INI value."""

    def test_from_config_none(self):
        guard = TTSStressParadoxGuard.from_config(None)
        self.assertFalse(guard.is_paradox_word("томас"))

    def test_from_config_empty(self):
        guard = TTSStressParadoxGuard.from_config("")
        self.assertFalse(guard.is_paradox_word("томас"))

    def test_from_config_comma_separated(self):
        guard = TTSStressParadoxGuard.from_config("Томас, Пейн")
        self.assertTrue(guard.is_paradox_word("томас"))
        self.assertTrue(guard.is_paradox_word("пейн"))

    def test_from_config_with_stress(self):
        guard = TTSStressParadoxGuard.from_config("Т+омас")  # stress on О
        self.assertTrue(guard.is_paradox_word("томас"))


if __name__ == "__main__":
    unittest.main()

