from typing import List

from audiobook_generator.config.general_config import GeneralConfig

NORMALIZER_OPENAI = "openai"


class BaseNormalizer:
    def __init__(self, config: GeneralConfig):
        self.config = config
        self.validate_config()

    def validate_config(self):
        raise NotImplementedError

    def normalize(self, text: str, chapter_title: str = "") -> str:
        raise NotImplementedError


def get_supported_normalizers() -> List[str]:
    return [NORMALIZER_OPENAI]


def get_normalizer(config: GeneralConfig) -> BaseNormalizer:
    if not config.normalize:
        raise ValueError("Text normalization is disabled")
    if config.normalize_provider == NORMALIZER_OPENAI:
        from audiobook_generator.normalizers.openai_normalizer import OpenAINormalizer

        return OpenAINormalizer(config)
    raise ValueError(f"Invalid normalizer provider: {config.normalize_provider}")

