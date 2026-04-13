from typing import List

from audiobook_generator.config.general_config import GeneralConfig

NORMALIZER_OPENAI = "openai"
NORMALIZER_LLM = "llm"
NORMALIZER_SIMPLE_SYMBOLS = "simple_symbols"
NORMALIZER_TTS_SAFE_SPLIT = "tts_safe_split"
NORMALIZER_NUMBERS_RU = "numbers_ru"
NORMALIZER_INITIALS_RU = "initials_ru"
NORMALIZER_PRONUNCIATION_EXCEPTIONS_RU = "pronunciation_exceptions_ru"
NORMALIZER_STRESS_WORDS_RU = "stress_words_ru"
NORMALIZER_TSNORM_RU = "tsnorm_ru"
NORMALIZER_PROPER_NOUNS_RU = "proper_nouns_ru"


class BaseNormalizer:
    STEP_VERSION = 1

    def __init__(self, config: GeneralConfig):
        self.config = config
        self.validate_config()

    def validate_config(self):
        raise NotImplementedError

    def normalize(self, text: str, chapter_title: str = "") -> str:
        raise NotImplementedError

    def get_step_name(self) -> str:
        return getattr(self, "STEP_NAME", self.__class__.__name__.lower())

    def get_resume_signature(self) -> dict:
        return {
            "step_name": self.get_step_name(),
            "step_version": getattr(self, "STEP_VERSION", 1),
        }

    def supports_chunked_resume(self) -> bool:
        return False

    def plan_processing_units(self, text: str, chapter_title: str = "") -> list[str]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support chunked processing units"
        )

    def process_unit(
        self,
        unit: str,
        *,
        chapter_title: str = "",
        unit_index: int,
        unit_count: int,
    ) -> str:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support chunked processing units"
        )

    def merge_processed_units(
        self,
        processed_units: list[str],
        *,
        chapter_title: str = "",
    ) -> str:
        return "\n\n".join(chunk for chunk in processed_units if chunk).strip()

    def get_step_artifacts(self, text: str, chapter_title: str = "") -> dict[str, str]:
        return {}

    def get_unit_artifacts(
        self,
        unit: str,
        *,
        chapter_title: str = "",
        unit_index: int,
        unit_count: int,
    ) -> dict[str, str]:
        return {}

    def get_normalizer_llm(self):
        cached_runtime = getattr(self.config, "_normalizer_llm_runtime", None)
        if cached_runtime is None:
            from audiobook_generator.normalizers.llm_support import NormalizerLLM

            cached_runtime = NormalizerLLM(self.config)
            setattr(self.config, "_normalizer_llm_runtime", cached_runtime)
        return cached_runtime

    def has_normalizer_llm(self) -> bool:
        return self.get_normalizer_llm().is_available

    def normalize_with_trace(self, text: str, chapter_title: str = "") -> tuple[str, list[tuple[str, str]]]:
        normalized = self.normalize(text, chapter_title=chapter_title)
        return normalized, [(self.get_step_name(), normalized)]


def get_supported_normalizers() -> List[str]:
    return [
        NORMALIZER_OPENAI,
        NORMALIZER_SIMPLE_SYMBOLS,
        NORMALIZER_INITIALS_RU,
        NORMALIZER_PRONUNCIATION_EXCEPTIONS_RU,
        NORMALIZER_STRESS_WORDS_RU,
        NORMALIZER_PROPER_NOUNS_RU,
        NORMALIZER_TSNORM_RU,
        NORMALIZER_TTS_SAFE_SPLIT,
        NORMALIZER_NUMBERS_RU,
    ]


def get_normalizer(config: GeneralConfig) -> BaseNormalizer:
    if not config.normalize:
        raise ValueError("Text normalization is disabled")
    steps = _resolve_normalizer_steps(config)
    normalizers = [_create_normalizer(step, config) for step in steps]

    if len(normalizers) == 1:
        return normalizers[0]

    return ChainNormalizer(config=config, normalizers=normalizers, steps=steps)


class ChainNormalizer(BaseNormalizer):
    def __init__(self, config: GeneralConfig, normalizers: List[BaseNormalizer], steps: List[str]):
        self.normalizers = normalizers
        self.steps = steps
        super().__init__(config)

    def validate_config(self):
        if not self.normalizers:
            raise ValueError("At least one normalizer step must be configured")

    def normalize(self, text: str, chapter_title: str = "") -> str:
        normalized = text
        for step_name, normalizer in zip(self.steps, self.normalizers):
            normalized = normalizer.normalize(normalized, chapter_title=chapter_title)
        return normalized

    def iter_steps(self):
        return list(zip(self.steps, self.normalizers))

    def normalize_with_trace(self, text: str, chapter_title: str = "") -> tuple[str, list[tuple[str, str]]]:
        normalized = text
        trace = []
        for step_name, normalizer in zip(self.steps, self.normalizers):
            if hasattr(normalizer, "normalize_with_trace"):
                normalized, step_trace = normalizer.normalize_with_trace(
                    normalized,
                    chapter_title=chapter_title,
                )
            else:
                normalized = normalizer.normalize(normalized, chapter_title=chapter_title)
                step_trace = [(step_name, normalized)]

            if not step_trace:
                step_trace = [(step_name, normalized)]

            for traced_name, traced_text in step_trace:
                trace.append((traced_name or step_name, traced_text))

        return normalized, trace


def _resolve_normalizer_steps(config: GeneralConfig) -> List[str]:
    if config.normalize_steps:
        steps = [normalize_step_name(step) for step in config.normalize_steps.split(",") if step.strip()]
        if not steps:
            raise ValueError("normalize_steps must contain at least one normalizer step")
        return steps

    return [normalize_step_name(config.normalize_provider or NORMALIZER_OPENAI)]


def normalize_step_name(step: str) -> str:
    lowered = step.strip().lower()
    aliases = {
        NORMALIZER_OPENAI: NORMALIZER_OPENAI,
        NORMALIZER_LLM: NORMALIZER_OPENAI,
        NORMALIZER_SIMPLE_SYMBOLS: NORMALIZER_SIMPLE_SYMBOLS,
        "symbols": NORMALIZER_SIMPLE_SYMBOLS,
        "safe_symbols": NORMALIZER_SIMPLE_SYMBOLS,
        NORMALIZER_INITIALS_RU: NORMALIZER_INITIALS_RU,
        "initials": NORMALIZER_INITIALS_RU,
        "ru_initials": NORMALIZER_INITIALS_RU,
        NORMALIZER_PRONUNCIATION_EXCEPTIONS_RU: NORMALIZER_PRONUNCIATION_EXCEPTIONS_RU,
        "pronunciation": NORMALIZER_PRONUNCIATION_EXCEPTIONS_RU,
        "pronunciation_exceptions": NORMALIZER_PRONUNCIATION_EXCEPTIONS_RU,
        "ru_pronunciation": NORMALIZER_PRONUNCIATION_EXCEPTIONS_RU,
        NORMALIZER_STRESS_WORDS_RU: NORMALIZER_STRESS_WORDS_RU,
        "stress_words": NORMALIZER_STRESS_WORDS_RU,
        "stress_overrides": NORMALIZER_STRESS_WORDS_RU,
        NORMALIZER_PROPER_NOUNS_RU: NORMALIZER_PROPER_NOUNS_RU,
        "proper_nouns": NORMALIZER_PROPER_NOUNS_RU,
        "proper_names": NORMALIZER_PROPER_NOUNS_RU,
        "stress_names": NORMALIZER_PROPER_NOUNS_RU,
        NORMALIZER_TSNORM_RU: NORMALIZER_TSNORM_RU,
        "tsnorm": NORMALIZER_TSNORM_RU,
        "stress_ru": NORMALIZER_TSNORM_RU,
        NORMALIZER_TTS_SAFE_SPLIT: NORMALIZER_TTS_SAFE_SPLIT,
        "safe_split": NORMALIZER_TTS_SAFE_SPLIT,
        "split": NORMALIZER_TTS_SAFE_SPLIT,
        "tts_split": NORMALIZER_TTS_SAFE_SPLIT,
        NORMALIZER_NUMBERS_RU: NORMALIZER_NUMBERS_RU,
        "numbers": NORMALIZER_NUMBERS_RU,
        "ru_numbers": NORMALIZER_NUMBERS_RU,
    }
    normalized = aliases.get(lowered)
    if not normalized:
        raise ValueError(f"Invalid normalizer step: {step}")
    return normalized


def _create_normalizer(step: str, config: GeneralConfig) -> BaseNormalizer:
    if step == NORMALIZER_OPENAI:
        from audiobook_generator.normalizers.openai_normalizer import OpenAINormalizer

        return OpenAINormalizer(config)
    if step == NORMALIZER_SIMPLE_SYMBOLS:
        from audiobook_generator.normalizers.simple_symbols_normalizer import SimpleSymbolsNormalizer

        return SimpleSymbolsNormalizer(config)
    if step == NORMALIZER_INITIALS_RU:
        from audiobook_generator.normalizers.initials_ru_normalizer import InitialsRuNormalizer

        return InitialsRuNormalizer(config)
    if step == NORMALIZER_PRONUNCIATION_EXCEPTIONS_RU:
        from audiobook_generator.normalizers.pronunciation_exceptions_ru_normalizer import (
            PronunciationExceptionsRuNormalizer,
        )

        return PronunciationExceptionsRuNormalizer(config)
    if step == NORMALIZER_STRESS_WORDS_RU:
        from audiobook_generator.normalizers.stress_words_ru_normalizer import StressWordsRuNormalizer

        return StressWordsRuNormalizer(config)
    if step == NORMALIZER_PROPER_NOUNS_RU:
        from audiobook_generator.normalizers.proper_nouns_ru_normalizer import ProperNounsRuNormalizer

        return ProperNounsRuNormalizer(config)
    if step == NORMALIZER_TSNORM_RU:
        from audiobook_generator.normalizers.tsnorm_ru_normalizer import TSNormRuNormalizer

        return TSNormRuNormalizer(config)
    if step == NORMALIZER_TTS_SAFE_SPLIT:
        from audiobook_generator.normalizers.tts_safe_split_normalizer import TTSSafeSplitNormalizer

        return TTSSafeSplitNormalizer(config)
    if step == NORMALIZER_NUMBERS_RU:
        from audiobook_generator.normalizers.numbers_ru_normalizer import NumbersRuNormalizer

        return NumbersRuNormalizer(config)
    raise ValueError(f"Invalid normalizer step: {step}")

