from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from audiobook_generator.config.general_config import GeneralConfig

DEFAULT_SYSTEM_PROMPT = """You normalize ebook text for text-to-speech.

Rules:
- Preserve meaning and structure.
- Do not summarize, censor, or add new facts.
- Keep paragraph breaks where they help narration.
- Fix obvious formatting artifacts and awkward punctuation.
- Expand only symbols or shorthand that would sound bad when read aloud.
- Never output labels, headings, field names, or wrapper text such as "Chapter", "Text", or "Title".
- Return only the normalized text."""
DEFAULT_USER_PROMPT_TEMPLATE = "{text}"


@dataclass(frozen=True)
class NormalizerLLMSettings:
    provider: str
    model: str
    base_url: str | None
    api_key: str | None
    max_chars: int
    system_prompt: str
    user_prompt_template: str


class NormalizerLLM:
    def __init__(self, config: GeneralConfig):
        self.config = config
        self.settings = resolve_normalizer_llm_settings(config)
        self.client = self._build_client()

    @property
    def is_available(self) -> bool:
        return self.client is not None

    def ensure_available(self):
        if self.is_available:
            return
        raise ValueError(
            "LLM-backed normalizer features are not available. "
            "Provide normalize_api_key / normalize_base_url or the matching environment variables."
        )

    def render_user_prompt(self, *, chapter_title: str, text: str) -> str:
        try:
            return self.settings.user_prompt_template.format(
                chapter_title=chapter_title,
                text=text,
            )
        except KeyError as exc:
            raise ValueError(
                "Unknown placeholder in normalize_user_prompt_file. "
                "Supported placeholders: {chapter_title}, {text}."
            ) from exc

    def complete(
        self,
        *,
        user_prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0,
    ) -> str:
        self.ensure_available()
        response = self.client.chat.completions.create(
            model=model or self.settings.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt or self.settings.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def _build_client(self):
        if self.settings.provider != "openai":
            return None

        api_key = self.settings.api_key
        if not api_key and not self.settings.base_url:
            return None
        if not api_key and self.settings.base_url:
            api_key = "dummy"

        return OpenAI(
            api_key=api_key,
            base_url=self.settings.base_url,
            max_retries=4,
        )


def resolve_normalizer_llm_settings(config: GeneralConfig) -> NormalizerLLMSettings:
    provider = config.normalize_provider or "openai"
    model = config.normalize_model or "gpt-4.1-mini"
    max_chars = config.normalize_max_chars or 4000

    system_prompt_path = config.normalize_system_prompt_file or config.normalize_prompt_file
    system_prompt = DEFAULT_SYSTEM_PROMPT
    if system_prompt_path:
        system_prompt = Path(system_prompt_path).read_text(encoding="utf-8").strip()

    user_prompt_template = DEFAULT_USER_PROMPT_TEMPLATE
    if config.normalize_user_prompt_file:
        user_prompt_template = Path(config.normalize_user_prompt_file).read_text(
            encoding="utf-8"
        ).strip()

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

    return NormalizerLLMSettings(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_chars=max_chars,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
    )
