"""Chunked TTS audio generation with sentence-level resume.

Usage
-----
Instead of sending an entire chapter text as one TTS call, this module:

1. Splits the chapter text into sentences.
2. Computes a content-hash for each sentence (includes voice + model so
   changing those forces re-synthesis).
3. Skips sentences whose audio file already exists on disk.
4. Calls the TTS provider for each missing sentence.
5. Marks any old sentences for this chapter that are no longer present as
   "superseded" (files are kept on disk for future comparison).
6. Concatenates all chunk audio files into the chapter output file.

Enable with ``--chunked_audio`` CLI flag.
"""
from __future__ import annotations

import difflib
import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

from audiobook_generator.core.audio_chunk_store import (
    AudioChunkStore,
    STATUS_SUPERSEDED,
    STATUS_SYNTHESIZED,
)
from audiobook_generator.core.audio_tags import AudioTags

logger = logging.getLogger(__name__)

# Minimum sentence length to synthesise (skip whitespace-only fragments).
MIN_SENTENCE_CHARS = 3


def _sentence_hash(sentence_text: str, voice_name: str, model_name: str) -> str:
    """Content-addressed hash that changes when text OR voice/model changes."""
    key = f"{sentence_text.strip()}|{voice_name or ''}|{model_name or ''}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def split_into_sentences(text: str, language: str = "ru") -> List[str]:
    """Split *text* into sentences using sentencex.

    Falls back to splitting on double-newlines if sentencex is not available.
    """
    try:
        from sentencex import segment  # type: ignore
        lang = language.split("-")[0]  # "ru-RU" → "ru"
        sentences = list(segment(lang, text))
    except Exception:
        # Simple fallback: split on paragraph boundaries
        sentences = [s.strip() for s in text.split("\n\n") if s.strip()]
    return [s for s in sentences if len(s.strip()) >= MIN_SENTENCE_CHARS]


def _merge_audio_files(chunk_paths: List[str], output_path: str) -> None:
    """Concatenate audio chunk files into *output_path* using pydub."""
    try:
        from pydub import AudioSegment  # type: ignore

        combined: Optional[AudioSegment] = None
        for path in chunk_paths:
            seg = AudioSegment.from_file(path)
            combined = seg if combined is None else combined + seg
        if combined is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fmt = out.suffix.lstrip(".") or "mp3"
            combined.export(str(out), format=fmt)
        logger.debug("Merged %d chunks into %s", len(chunk_paths), output_path)
    except ImportError:
        raise RuntimeError(
            "pydub is required for chunked audio merging. "
            "Install it with: pip install pydub"
        )


class ChunkedAudioGenerator:
    """Per-chapter chunked TTS synthesiser with SQLite resume support."""

    def __init__(
        self,
        *,
        config,
        chunk_store: AudioChunkStore,
        tts_provider,
        run_id: str,
        chunks_base_dir: str,
    ):
        self.config = config
        self.store = chunk_store
        self.tts_provider = tts_provider
        self.run_id = run_id
        self.chunks_base_dir = Path(chunks_base_dir)

    def _chunk_dir(self, chapter_key: str) -> Path:
        """Return (and create) the directory for this chapter's chunk files."""
        d = self.chunks_base_dir / chapter_key
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _chunk_path(self, chapter_key: str, sentence_hash: str) -> str:
        ext = self.tts_provider.get_output_file_extension()
        return str(self._chunk_dir(chapter_key) / f"{sentence_hash}.{ext}")

    def process_chapter(
        self,
        *,
        chapter_idx: int,
        chapter_key: str,
        text_for_tts: str,
        output_file: str,
        audio_tags: AudioTags,
    ) -> bool:
        """Synthesise all sentences for *chapter_key*, then merge into *output_file*.

        Returns True on success.
        """
        voice = self.config.voice_name or ""
        model = self.config.model_name or ""

        sentences = split_into_sentences(text_for_tts, self.config.language or "ru")
        if not sentences:
            logger.warning("Chapter %d '%s' produced no sentences; skipping.", chapter_idx, chapter_key)
            return False

        logger.info(
            "Chapter %d: %d sentences to synthesise (chunked mode).", chapter_idx, len(sentences)
        )

        # Load current DB state for this chapter to detect superseded chunks.
        existing_rows = self.store.get_chunks_for_chapter(self.run_id, chapter_idx)
        existing_by_pos: dict[int, str] = {r["sentence_pos"]: r["sentence_hash"] for r in existing_rows}

        new_hashes = [_sentence_hash(s, voice, model) for s in sentences]
        new_hash_set = set(new_hashes)

        # Detect superseded: hashes that were in the DB but are no longer present.
        old_hash_set = set(existing_by_pos.values())
        stale_hashes = old_hash_set - new_hash_set
        if stale_hashes:
            logger.info(
                "Chapter %d: %d sentence(s) changed or removed; marking as superseded.",
                chapter_idx, len(stale_hashes),
            )
            # Try to link each stale hash to its positional replacement using SequenceMatcher.
            old_hashes_in_order = [existing_by_pos.get(p, "") for p in sorted(existing_by_pos)]
            matcher = difflib.SequenceMatcher(None, old_hashes_in_order, new_hashes, autojunk=False)
            stale_to_replacement: dict[str, Optional[str]] = {h: None for h in stale_hashes}
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "replace":
                    for a, b in zip(
                        old_hashes_in_order[i1:i2], new_hashes[j1:j2]
                    ):
                        if a in stale_to_replacement:
                            stale_to_replacement[a] = b
            for stale_hash, replacement_hash in stale_to_replacement.items():
                self.store.mark_superseded(
                    run_id=self.run_id,
                    chapter_idx=chapter_idx,
                    old_hash=stale_hash,
                    superseded_by_hash=replacement_hash,
                )

        # Register all new chunks in DB (upsert is idempotent).
        for pos, (sentence, s_hash) in enumerate(zip(sentences, new_hashes)):
            chunk_path = self._chunk_path(chapter_key, s_hash)
            already_done = self.store.has_synthesized(self.run_id, chapter_idx, s_hash)
            self.store.upsert_chunk(
                run_id=self.run_id,
                chapter_idx=chapter_idx,
                chapter_key=chapter_key,
                sentence_pos=pos,
                sentence_hash=s_hash,
                sentence_text=sentence,
                audio_path=chunk_path if already_done else None,
                status=STATUS_SYNTHESIZED if already_done else "pending",
            )

        # Synthesise missing chunks.
        synthesised = 0
        skipped = 0
        errors = 0
        for pos, (sentence, s_hash) in enumerate(zip(sentences, new_hashes)):
            chunk_path = self._chunk_path(chapter_key, s_hash)
            if os.path.exists(chunk_path) and self.store.has_synthesized(
                self.run_id, chapter_idx, s_hash
            ):
                skipped += 1
                continue
            try:
                self.tts_provider.text_to_speech(sentence, chunk_path, audio_tags)
                self.store.mark_synthesized(
                    run_id=self.run_id,
                    chapter_idx=chapter_idx,
                    sentence_hash=s_hash,
                    audio_path=chunk_path,
                )
                synthesised += 1
            except Exception as exc:
                logger.error(
                    "Chapter %d sentence %d synthesis failed: %s", chapter_idx, pos, exc
                )
                errors += 1

        logger.info(
            "Chapter %d synthesis done: %d new, %d skipped, %d errors.",
            chapter_idx, synthesised, skipped, errors,
        )

        if errors > 0:
            logger.warning(
                "Chapter %d has %d synthesis errors; output may be incomplete.", chapter_idx, errors
            )

        # Collect synthesised chunk paths in position order.
        chunk_paths = self.store.get_synthesized_audio_paths(self.run_id, chapter_idx)
        if not chunk_paths:
            logger.error("Chapter %d: no audio chunks available for merging.", chapter_idx)
            return False

        # Merge chunks into the chapter output file.
        try:
            _merge_audio_files(chunk_paths, output_file)
            logger.info("Chapter %d merged into %s", chapter_idx, output_file)
            return True
        except Exception as exc:
            logger.error("Chapter %d merge failed: %s", chapter_idx, exc)
            return False

