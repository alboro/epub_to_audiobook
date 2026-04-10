import logging
import os
from pathlib import Path

from openai import OpenAI

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.normalizers.base_normalizer import BaseNormalizer
from audiobook_generator.utils.utils import split_text

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = """You normalize ebook text for text-to-speech.

Rules:
- Preserve meaning and structure.
- Do not summarize, censor, or add new facts.
- Keep paragraph breaks where they help narration.
- Fix obvious formatting artifacts and awkward punctuation.
- Expand only symbols or shorthand that would sound bad when read aloud.
- Return only the normalized text."""


class OpenAINormalizer(BaseNormalizer):
    def __init__(self, config: GeneralConfig):
        config.normalize_provider = config.normalize_provider or "openai"
        config.normalize_model = config.normalize_model or "gpt-4.1-mini"
        config.normalize_max_chars = config.normalize_max_chars or 4000

        self.prompt = DEFAULT_PROMPT
        if config.normalize_prompt_file:
            self.prompt = Path(config.normalize_prompt_file).read_text(encoding="utf-8").strip()

        super().__init__(config)

        base_url = (
            config.normalize_base_url
            or os.getenv("NORMALIZER_OPENAI_BASE_URL")
            or config.openai_base_url
            or os.getenv("OPENAI_BASE_URL")
        )
        api_key = (
            config.normalize_api_key
            or os.getenv("NORMALIZER_OPENAI_API_KEY")
            or config.openai_api_key
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key and base_url:
            api_key = "dummy"

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=4,
        )

    def validate_config(self):
        if self.config.normalize_max_chars == 0:
            raise ValueError("Normalizer max chars must be positive")

    def normalize(self, text: str, chapter_title: str = "") -> str:
        max_chars = self.config.normalize_max_chars
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
            response = self.client.chat.completions.create(
                model=self.config.normalize_model,
                temperature=0,
                messages=[
                    {"role": "system", "content": self.prompt},
                    {
                        "role": "user",
                        "content": f"Chapter title: {chapter_title}\n\nText:\n{chunk}",
                    },
                ],
            )
            normalized = response.choices[0].message.content or ""
            normalized_chunks.append(normalized.strip())

        return "\n\n".join(chunk for chunk in normalized_chunks if chunk).strip()
