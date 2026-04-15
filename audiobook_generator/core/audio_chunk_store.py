"""SQLite store for per-sentence audio chunk tracking.

Supports chunked TTS generation with resume: each sentence is addressed by a
content-hash so unchanged sentences are not re-synthesised between runs.

DB location: output_folder/_state/audio_chunks.sqlite3
  (shared across all runs so supersession history is preserved)

Schema
------
audio_chunk_runs   — one row per run (run_id = wav/NNN path suffix like "001")
audio_chunks       — one row per sentence per run; tracks synthesis status,
                     audio file path, and optional link to the hash that
                     superseded it in a later run.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, UTC
from pathlib import Path
from typing import Iterator, List, Optional


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


# Possible values for audio_chunks.status
STATUS_PENDING = "pending"
STATUS_SYNTHESIZED = "synthesized"
STATUS_SUPERSEDED = "superseded"
STATUS_ERROR = "error"


class AudioChunkStore:
    """Tracks per-sentence audio chunks in a WAL-mode SQLite database."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self):
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _initialize(self):
        with closing(self._connect()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS audio_chunk_runs (
                    run_id      TEXT PRIMARY KEY,
                    created_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS audio_chunks (
                    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id               TEXT NOT NULL,
                    chapter_idx          INTEGER NOT NULL,
                    chapter_key          TEXT NOT NULL,
                    sentence_pos         INTEGER NOT NULL,
                    sentence_hash        TEXT NOT NULL,
                    sentence_text        TEXT NOT NULL,
                    audio_path           TEXT,
                    status               TEXT NOT NULL DEFAULT 'pending',
                    superseded_by_hash   TEXT,
                    created_at           TEXT NOT NULL,
                    updated_at           TEXT NOT NULL,
                    UNIQUE (run_id, chapter_idx, sentence_pos),
                    UNIQUE (run_id, chapter_idx, sentence_hash)
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_hash
                    ON audio_chunks (sentence_hash);

                CREATE INDEX IF NOT EXISTS idx_chunks_chapter
                    ON audio_chunks (run_id, chapter_idx);
                """
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Run management
    # ------------------------------------------------------------------

    def ensure_run(self, run_id: str) -> None:
        """Register a run_id if it does not exist yet."""
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO audio_chunk_runs (run_id, created_at) VALUES (?, ?)",
                (run_id, _utc_now()),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Chunk write operations
    # ------------------------------------------------------------------

    def upsert_chunk(
        self,
        *,
        run_id: str,
        chapter_idx: int,
        chapter_key: str,
        sentence_pos: int,
        sentence_hash: str,
        sentence_text: str,
        audio_path: Optional[str] = None,
        status: str = STATUS_PENDING,
        superseded_by_hash: Optional[str] = None,
    ) -> None:
        now = _utc_now()
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO audio_chunks
                    (run_id, chapter_idx, chapter_key, sentence_pos,
                     sentence_hash, sentence_text, audio_path,
                     status, superseded_by_hash, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (run_id, chapter_idx, sentence_pos) DO UPDATE SET
                    sentence_hash      = excluded.sentence_hash,
                    sentence_text      = excluded.sentence_text,
                    audio_path         = COALESCE(excluded.audio_path, audio_path),
                    status             = excluded.status,
                    superseded_by_hash = excluded.superseded_by_hash,
                    updated_at         = excluded.updated_at
                """,
                (
                    run_id, chapter_idx, chapter_key, sentence_pos,
                    sentence_hash, sentence_text, audio_path,
                    status, superseded_by_hash, now, now,
                ),
            )
            conn.commit()

    def mark_synthesized(self, *, run_id: str, chapter_idx: int,
                         sentence_hash: str, audio_path: str) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                UPDATE audio_chunks
                SET status = ?, audio_path = ?, updated_at = ?
                WHERE run_id = ? AND chapter_idx = ? AND sentence_hash = ?
                """,
                (STATUS_SYNTHESIZED, audio_path, _utc_now(),
                 run_id, chapter_idx, sentence_hash),
            )
            conn.commit()

    def mark_superseded(self, *, run_id: str, chapter_idx: int,
                        old_hash: str, superseded_by_hash: Optional[str]) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                UPDATE audio_chunks
                SET status = ?, superseded_by_hash = ?, updated_at = ?
                WHERE run_id = ? AND chapter_idx = ? AND sentence_hash = ?
                """,
                (STATUS_SUPERSEDED, superseded_by_hash, _utc_now(),
                 run_id, chapter_idx, old_hash),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Chunk read operations
    # ------------------------------------------------------------------

    def get_chunks_for_chapter(self, run_id: str, chapter_idx: int) -> List[sqlite3.Row]:
        """Return all chunk rows for this run+chapter, ordered by sentence_pos."""
        with closing(self._connect()) as conn:
            return conn.execute(
                """
                SELECT * FROM audio_chunks
                WHERE run_id = ? AND chapter_idx = ?
                ORDER BY sentence_pos
                """,
                (run_id, chapter_idx),
            ).fetchall()

    def get_synthesized_audio_paths(self, run_id: str, chapter_idx: int) -> List[str]:
        """Return audio file paths for all synthesized chunks in order."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT audio_path FROM audio_chunks
                WHERE run_id = ? AND chapter_idx = ?
                  AND status = ?
                  AND audio_path IS NOT NULL
                ORDER BY sentence_pos
                """,
                (run_id, chapter_idx, STATUS_SYNTHESIZED),
            ).fetchall()
        return [r["audio_path"] for r in rows]

    def get_hash_by_pos(self, run_id: str, chapter_idx: int, sentence_pos: int) -> Optional[str]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT sentence_hash FROM audio_chunks WHERE run_id=? AND chapter_idx=? AND sentence_pos=?",
                (run_id, chapter_idx, sentence_pos),
            ).fetchone()
        return row["sentence_hash"] if row else None

    def has_synthesized(self, run_id: str, chapter_idx: int, sentence_hash: str) -> bool:
        """Return True if this hash already has a synthesized audio file."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT audio_path FROM audio_chunks
                WHERE run_id = ? AND chapter_idx = ? AND sentence_hash = ?
                  AND status = ?
                """,
                (run_id, chapter_idx, sentence_hash, STATUS_SYNTHESIZED),
            ).fetchone()
        return row is not None and row["audio_path"] is not None

