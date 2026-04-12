import logging

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer
from audiobook_generator.utils.utils import split_text

logger = logging.getLogger(__name__)


class OpenAINormalizer(BaseNormalizer):
    STEP_NAME = "llm"

    def __init__(self, config: GeneralConfig):
        config.normalize_provider = config.normalize_provider or "openai"
        config.normalize_model = config.normalize_model or "gpt-4.1-mini"
        config.normalize_max_chars = config.normalize_max_chars or 4000

        super().__init__(config)

    def validate_config(self):
        llm = self.get_normalizer_llm()
        if self.config.normalize_max_chars == 0:
            raise ValueError("Normalizer max chars must be positive")
        if not llm.is_available:
            raise ValueError(
                "The 'llm' normalizer requires a configured LLM endpoint. "
                "Set normalize_base_url / normalize_api_key or the corresponding environment variables."
            )
        if not llm.settings.system_prompt:
            raise ValueError("Normalizer system prompt must not be empty")
        if "{text}" not in llm.settings.user_prompt_template:
            raise ValueError("Normalizer user prompt template must contain the {text} placeholder")

    def normalize(self, text: str, chapter_title: str = "") -> str:
        llm = self.get_normalizer_llm()
        max_chars = llm.settings.max_chars
        chunks = [text] if not max_chars or max_chars < 0 else split_text(text, max_chars, self.config.language)
        normalized_chunks = []

        for idx, chunk in enumerate(chunks, start=1):
            logger.info(
                "Normalizing chapter '%s' chunk %s/%s, length=%s",
                chapter_title,
                idx,
                len(chunks),
                len(chunk),
            )
            user_prompt = llm.render_user_prompt(chapter_title=chapter_title, text=chunk)
            normalized = llm.complete(
                user_prompt=user_prompt,
                model=self.config.normalize_model,
                temperature=0,
            )
            normalized_chunks.append(normalized.strip())

        return "\n\n".join(chunk for chunk in normalized_chunks if chunk).strip()
