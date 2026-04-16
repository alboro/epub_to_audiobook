import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

from audiobook_generator.core.chunked_audio_generator import _sentence_hash, split_into_sentences


def test_audio_mode_reuses_existing_chunks():
    """Ensure audio mode with chunked_audio reuses existing synthesized chunks (no TTS calls)."""
    from audiobook_generator.config.general_config import GeneralConfig
    from audiobook_generator.core.audiobook_generator import AudiobookGenerator

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        # create previous text run
        (out / "text" / "001").mkdir(parents=True)

        # Prepare chapter text with two sentences (fallback split on double newline)
        chapter_text = "alpha\n\nbeta"
        sentences = split_into_sentences(chapter_text, "ru")
        assert sentences == ["alpha", "beta"]

        # Build hashes using default voice/model values we'll set in config
        # We'll create a config with specific voice/model to compute same hashes
        args = type("A", (), {})()
        args.output_folder = tmp
        args.tts = "openai"
        args.language = "ru-RU"
        args.voice_name = "reference"
        args.model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        args.worker_count = 1
        args.chapter_start = 1
        args.chapter_end = -1
        args.mode = "audio"
        args.normalize = False
        args.chunked_audio = True
        args.no_prompt = True
        # ensure normalize-related fields exist
        args.normalize_log_changes = False

        config = GeneralConfig(args)
        gen = AudiobookGenerator(config)

        # Prepare DB: output/_state/audio_chunks.sqlite3
        state_dir = out / "_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        db_path = state_dir / "audio_chunks.sqlite3"
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS audio_chunk_runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audio_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                chapter_idx INTEGER NOT NULL,
                chapter_key TEXT NOT NULL,
                sentence_pos INTEGER NOT NULL,
                sentence_hash TEXT NOT NULL,
                sentence_text TEXT NOT NULL,
                audio_path TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                superseded_by_hash TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        # register run_id '001'
        conn.execute("INSERT OR IGNORE INTO audio_chunk_runs (run_id, created_at) VALUES (?, datetime('now'))", ("001",))

        # Determine hashes
        hashes = [_sentence_hash(s, config.voice_name or "", config.model_name or "") for s in sentences]

        # Create chunk files under wav/001/chunks/0001_Test
        chapter_key = "0001_Test"
        chunk_dir = out / "wav" / "001" / "chunks" / chapter_key
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for pos, (s_hash, s_text) in enumerate(zip(hashes, sentences)):
            chunk_path = chunk_dir / f"{s_hash}.wav"
            chunk_path.write_bytes(b"dummy")
            conn.execute(
                "INSERT INTO audio_chunks (run_id, chapter_idx, chapter_key, sentence_pos, sentence_hash, sentence_text, audio_path, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,datetime('now'),datetime('now'))",
                ("001", 1, chapter_key, pos, s_hash, s_text, str(chunk_path), 'synthesized'),
            )
        conn.commit()
        conn.close()

        # Patch make_safe_filename to produce our chapter_key for simplicity
        with patch("audiobook_generator.core.audiobook_generator.make_safe_filename", return_value=chapter_key):

            # Dummy TTS that would raise if called (we expect 0 calls)
            class DummyTTS:
                def __init__(self):
                    self.calls = 0

                def get_break_string(self):
                    return "\n\n"

                def estimate_cost(self, total_chars):
                    return 0.0

                def get_output_file_extension(self):
                    return "wav"

                def text_to_speech(self, text, out_path, tags):
                    self.calls += 1
                    raise RuntimeError("TTS should not be called when chunks already present")

            # Patch get_book_parser to return our single chapter
            class DummyParser:
                def get_chapters(self, break_str):
                    return [("Test", chapter_text)]
                def get_book_title(self):
                    return "Book"
                def get_book_author(self):
                    return "Author"
                def get_book_cover(self):
                    return None

            dummy_tts = DummyTTS()

            from unittest.mock import patch as upatch
            with upatch("audiobook_generator.core.audiobook_generator.get_book_parser", return_value=DummyParser()), \
                 upatch("audiobook_generator.core.audiobook_generator.get_tts_provider", return_value=dummy_tts):
                # Run generator in audio mode
                gen.run()

            # Ensure tts was not called (chunks reused)
            assert dummy_tts.calls == 0


if __name__ == "__main__":
    test_audio_mode_reuses_existing_chunks()

