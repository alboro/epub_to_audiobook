from __future__ import annotations

import re
from pathlib import Path

COMBINING_ACUTE = "\u0301"
CYRILLIC_WORD_PATTERN = re.compile(r"[А-Яа-яЁё-]+")


def is_russian_language(language: str | None) -> bool:
    return (language or "ru").split("-")[0].lower() == "ru"


def strip_combining_acute(text: str) -> str:
    return text.replace(COMBINING_ACUTE, "")


def preserve_case(source: str, replacement: str) -> str:
    if not source:
        return replacement
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def load_mapping_file(path: str | None) -> dict[str, str]:
    if not path:
        return {}

    mapping: dict[str, str] = {}
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise ValueError(
                f"Invalid mapping line {line_number} in {path!r}. "
                "Expected the format 'source==replacement'."
            )
        source, replacement = line.split("==", 1)
        source = source.strip()
        replacement = replacement.strip()
        if not source:
            raise ValueError(
                f"Invalid mapping line {line_number} in {path!r}. Source must not be empty."
            )
        mapping[source] = replacement
    return mapping
