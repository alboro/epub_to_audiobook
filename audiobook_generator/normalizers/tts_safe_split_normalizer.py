from __future__ import annotations

import logging
import re

from sentencex import segment

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer

logger = logging.getLogger(__name__)

DEFAULT_SAFE_MAX_CHARS = 180
MIN_SPLIT_FRACTION = 0.45
MIN_SPLIT_FRAGMENT_CHARS = 24
MIN_SPLIT_FRAGMENT_WORDS = 2
# Sentences shorter than this will be merged with the next sentence to avoid TTS instability
MIN_TTS_SAFE_CHARS = 12
LEFT_TRIM_CHARS = " \t\r\n,;:-–—"
RIGHT_TRIM_CHARS = " \t\r\n,;:-–—"
SENTENCE_END_CHARS = ".!?"

PRIORITY_PATTERNS = (
    # 0. Existing sentence boundary: period/!/? followed by space and capital letter.
    #    This is always the best split point — avoids breaking clauses mid-sentence.
    re.compile(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z«\"])", re.UNICODE),
    re.compile(r"[;:](?=\s|$)"),
    # Conjunctions after punctuation only — prevents splitting "Ветхого и Нового" (no comma).
    # Exclude "и" followed by adverbial particles ("и более того", "и даже", "и при этом" etc.)
    re.compile(
        r"(?<=[,;])\s+(?=(?:а|но|однако)\b)"
        r"|(?<=[,;])\s+(?=и\s+(?!(?:более|даже|тем|при|всё|всего|ещё|только|притом|при этом)\b))",
        re.IGNORECASE,
    ),
    re.compile(r",\s+(?=(?:а также|однако|но|зато|поэтому|причем|притом|при этом|затем|потом)\b)", re.IGNORECASE),
    re.compile(
        r",\s+(?=(?:котор(?:ый|ая|ое|ые|ого|ому|ым|ых|ую|ой|ою)|обосновывающ\w*|существующ\w*|позволяющ\w*|делающ\w*|создающ\w*)\b)",
        re.IGNORECASE,
    ),
    re.compile(r",(?=\s|$)"),
    re.compile(r"\s-\s"),
)


_DEFAULT_SAFE_SPLIT_SYSTEM_PROMPT = (
    "You are a text-to-speech pre-processor. "
    "You receive a JSON list of sentences that have been algorithmically split for TTS. "
    "Your task: review punctuation and improve natural speech flow where needed. "
    "Rules:\n"
    "- Keep the content and meaning unchanged.\n"
    "- Only adjust punctuation marks (commas, dashes, semicolons, colons).\n"
    "- Do NOT merge or split sentences.\n"
    "- Return a JSON array of the same number of strings, in the same order.\n"
    "- If no change is needed for a sentence, return it as-is."
)


def _get_safe_split_prompt(config) -> str:
    custom = getattr(config, "normalize_safe_split_system_prompt", None)
    if custom and isinstance(custom, str):
        return custom
    return _DEFAULT_SAFE_SPLIT_SYSTEM_PROMPT


class TTSSafeSplitNormalizer(BaseNormalizer):
    STEP_NAME = "tts_llm_safe_split"

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
        result = "".join(normalized_parts).strip()
        result = self._llm_refine(result, chapter_title=chapter_title)
        return result

    def _llm_refine(self, text: str, *, chapter_title: str = "") -> str:
        """Optional LLM pass to refine punctuation in split sentences."""
        if not self.has_normalizer_llm():
            return text
        sentences = [s for s in re.split(r"(?<=[.!?]) +", text) if s.strip()]
        if not sentences:
            return text
        try:
            llm = self.get_normalizer_llm()
            import json as _json
            prompt = _json.dumps(sentences, ensure_ascii=False)
            system_prompt = _get_safe_split_prompt(self.config)
            response = llm.complete(
                prompt,
                system_prompt=system_prompt,
                model=self.config.normalize_model,
                temperature=0,
            )
            refined = _json.loads(response.strip())
            if isinstance(refined, list) and len(refined) == len(sentences):
                logger.info(
                    "TTS safe split LLM refinement applied to chapter '%s': %s sentences",
                    chapter_title,
                    len(refined),
                )
                return " ".join(s.strip() for s in refined if s.strip())
        except Exception as exc:
            logger.warning(
                "TTS safe split LLM refinement skipped for chapter '%s': %s",
                chapter_title,
                exc,
            )
        return text

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

        # Merge very short sentences with neighbors to avoid TTS instability.
        # Prefer merging with the PREVIOUS sentence (better semantic coherence):
        # "Или нет." is a reply to the previous question, not a preamble to the next.
        merged_sentences = []
        i = 0
        while i < len(safe_sentences):
            current = safe_sentences[i]
            if len(current) < MIN_TTS_SAFE_CHARS:
                # Try merge with PREVIOUS first (fits and previous exists)
                if (
                    merged_sentences
                    and len(merged_sentences[-1]) + 1 + len(current) <= self.max_chars
                ):
                    prev = merged_sentences.pop()
                    # If the short sentence already ends with ! or ?, preserve it;
                    # only strip trailing period (or nothing) before re-finalising.
                    if current and current[-1] in "!?":
                        merged_sentences.append(f"{prev} {current}")
                    else:
                        base_curr = current.rstrip(".").rstrip()
                        merged_sentences.append(f"{prev} {base_curr}.")
                    i += 1
                elif i + 1 < len(safe_sentences):
                    # Fall back to merge with NEXT
                    next_sent = safe_sentences[i + 1]
                    if current and current[-1] in "!?":
                        merged_sentences.append(f"{current} {next_sent}")
                    else:
                        base = current.rstrip(".").rstrip()
                        merged_sentences.append(f"{base} {next_sent}")
                    i += 2
                else:
                    merged_sentences.append(current)
                    i += 1
            else:
                merged_sentences.append(current)
                i += 1

        return " ".join(merged_sentences).strip(), len(merged_sentences), inserted_splits

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
            if not left or not right or not self._is_acceptable_split(left, right):
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
            candidate_indexes = [
                match.end()
                for match in pattern.finditer(window)
                if match.end() >= min_index
            ]
            split_index = self._select_best_candidate(sentence, candidate_indexes)
            if split_index is not None:
                return split_index

        space_indexes = [
            index + 1
            for index, char in enumerate(window)
            if char == " " and index + 1 >= min_index
        ]
        split_index = self._select_best_candidate(sentence, space_indexes)
        if split_index is not None:
            return split_index

        return self.max_chars

    def _select_best_candidate(self, sentence: str, candidate_indexes: list[int]) -> int | None:
        for split_index in sorted(candidate_indexes, reverse=True):
            left = sentence[:split_index].rstrip(LEFT_TRIM_CHARS)
            right = sentence[split_index:].lstrip(RIGHT_TRIM_CHARS)
            if left and right and self._is_acceptable_split(left, right):
                return split_index
        return None

    def _is_acceptable_split(self, left: str, right: str) -> bool:
        if not left or not right:
            return False

        left_words = [word for word in left.split() if word]
        right_words = [word for word in right.split() if word]
        if len(left) < MIN_SPLIT_FRAGMENT_CHARS and len(left_words) < MIN_SPLIT_FRAGMENT_WORDS:
            return False
        if len(right) < MIN_SPLIT_FRAGMENT_CHARS and len(right_words) < MIN_SPLIT_FRAGMENT_WORDS:
            return False

        # Avoid leaving a dangling short function word (preposition/article) at end of left.
        # e.g. "...третьего года. По" → "По" is only 2 chars and belongs to the next phrase.
        if left_words:
            last_word = left_words[-1].strip('.,!?;:\'"»«')
            if len(last_word) <= 3:
                return False

        return True

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
