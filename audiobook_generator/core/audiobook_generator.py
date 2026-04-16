import logging
import multiprocessing
import os
import shutil
from pathlib import Path

from audiobook_generator.book_parsers.base_book_parser import get_book_parser
from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.core.m4b_packager import package_m4b
from audiobook_generator.normalizers.base_normalizer import get_normalizer
from audiobook_generator.core.pipeline_runner import NormalizationPipelineRunner
from audiobook_generator.tts_providers.base_tts_provider import get_tts_provider
from audiobook_generator.utils.log_handler import setup_logging
from audiobook_generator.utils.filename_sanitizer import make_safe_filename

logger = logging.getLogger(__name__)

# Known audio extensions to scan when packaging without a TTS provider.
_AUDIO_EXTENSIONS = [".mp3", ".wav", ".ogg", ".opus", ".aac", ".m4a", ".flac"]


def confirm_conversion():
    logger.info("Do you want to continue? (y/n)")
    answer = input()
    if answer.lower() != "y":
        logger.info("Aborted.")
        exit(0)


def get_total_chars(chapters):
    total_characters = 0
    for title, text in chapters:
        total_characters += len(text)
    return total_characters


class AudiobookGenerator:
    def __init__(self, config: GeneralConfig):
        self.config = config

    def __str__(self) -> str:
        return f"{self.config}"

    def _chapter_text_path(self, base_dir, idx, title):
        safe_txt_name = make_safe_filename(
            title=title,
            idx=idx,
            output_dir=base_dir,
            ext=".txt",
            collision_check=False,
        )
        return os.path.join(base_dir, safe_txt_name)

    def _write_chapter_text(self, base_dir, idx, title, text):
        os.makedirs(base_dir, exist_ok=True)
        text_file = self._chapter_text_path(base_dir, idx, title)
        with open(text_file, "w", encoding="utf-8", newline="\n") as file_handle:
            file_handle.write(text)
        return text_file

    # ------------------------------------------------------------------
    # Run-index helpers (sequential NNN folders under text/ and wav/)
    # ------------------------------------------------------------------

    def _run_subdir(self, kind: str) -> "Path":
        """Return the Path for text/ or wav/ subdir under output_folder."""
        from pathlib import Path
        return Path(self.config.output_folder) / kind

    def _next_run_index(self, kind: str) -> str:
        """Return the next sequential run index string, e.g. '001'."""
        base = self._run_subdir(kind)
        base.mkdir(parents=True, exist_ok=True)
        existing = sorted(
            p.name for p in base.iterdir() if p.is_dir() and p.name.isdigit()
        )
        return f"{int(existing[-1]) + 1:03d}" if existing else "001"

    def _latest_run_index(self, kind: str) -> str | None:
        """Return the highest existing run index string, or None if none exist."""
        base = self._run_subdir(kind)
        if not base.exists():
            return None
        existing = sorted(
            p.name for p in base.iterdir() if p.is_dir() and p.name.isdigit()
        )
        return existing[-1] if existing else None

    def _text_run_dir(self) -> str:
        """Absolute path of the text run folder for the current run."""
        from pathlib import Path
        return str(
            Path(self.config.output_folder) / "text" / (self.config.current_run_index or "001")
        )

    def _wav_run_dir(self) -> str:
        """Absolute path of the wav run folder for the current run."""
        from pathlib import Path
        return str(
            Path(self.config.output_folder) / "wav" / (self.config.current_run_index or "001")
        )

    def _save_ini_snapshot(self, run_dir: str) -> None:
        """Write a config snapshot INI into the root output folder and the run folder."""
        from pathlib import Path
        from audiobook_generator.config.ini_config_manager import save_ini
        book_stem = Path(self.config.input_file or "book").stem
        # Root snapshot (always updated to reflect the latest run)
        root_ini = Path(self.config.output_folder) / f"{book_stem}.ini"
        save_ini(root_ini, self.config)
        # Run-specific snapshot
        if run_dir:
            run_ini = Path(run_dir) / f"{book_stem}.ini"
            save_ini(run_ini, self.config)
            logger.debug("Run config snapshot: %s", run_ini)

    def _copy_input_book(self):
        if not self.config.input_file or not self.config.output_folder:
            return None

        source_path = Path(self.config.input_file).expanduser().resolve()
        if not source_path.is_file():
            return None

        source_dir = Path(self.config.output_folder) / "_source"
        source_dir.mkdir(parents=True, exist_ok=True)
        target_path = source_dir / source_path.name

        try:
            if source_path.samefile(target_path):
                return str(target_path)
        except FileNotFoundError:
            pass

        if target_path.exists():
            source_stat = source_path.stat()
            target_stat = target_path.stat()
            if (
                source_stat.st_size == target_stat.st_size
                and int(source_stat.st_mtime) == int(target_stat.st_mtime)
            ):
                logger.info("Source book already copied to %s", target_path)
                return str(target_path)

        shutil.copy2(source_path, target_path)
        logger.info("Copied source book to %s", target_path)
        return str(target_path)

    def _chapter_artifact_dir(self, idx, title):
        # Artifacts live under the text run folder (they describe the normalization
        # that produced the text for this run).
        text_run = self._text_run_dir() if self.config.current_run_index else os.path.join(
            self.config.output_folder, "_chapter_artifacts"
        )
        artifacts_root = os.path.join(text_run, "_chapter_artifacts")
        safe_name = make_safe_filename(
            title=title,
            idx=idx,
            output_dir=artifacts_root,
            ext=".txt",
            collision_check=False,
        )
        if safe_name.lower().endswith(".txt"):
            safe_name = safe_name[:-4]
        return os.path.join(artifacts_root, safe_name)

    def _write_chapter_artifact(self, artifact_dir, filename, text):
        os.makedirs(artifact_dir, exist_ok=True)
        path = os.path.join(artifact_dir, filename)
        with open(path, "w", encoding="utf-8", newline="\n") as file_handle:
            file_handle.write(text)
        return path

    def _save_chapter_artifacts(
        self,
        *,
        idx,
        title,
        raw_text,
        source_text,
        prepared_text_path,
        normalizer_trace,
        final_text,
        final_label,
    ):
        artifact_dir = self._chapter_artifact_dir(idx, title)
        source_kind = "prepared_text" if prepared_text_path else "raw_epub"
        normalize_steps = self.config.normalize_steps or self.config.normalize_provider or "disabled"
        manifest_lines = [
            f"chapter_index: {idx}",
            f"chapter_title: {title}",
            f"source_kind: {source_kind}",
            f"prepared_text_path: {prepared_text_path or ''}",
            f"language: {self.config.language}",
            f"tts_provider: {self.config.tts}",
            f"tts_model: {self.config.model_name}",
            f"voice_name: {self.config.voice_name}",
            f"normalize_enabled: {bool(self.config.normalize)}",
            f"normalize_steps: {normalize_steps}",
            f"raw_chars: {len(raw_text)}",
            f"source_chars: {len(source_text)}",
            f"final_chars: {len(final_text)}",
            f"final_label: {final_label}",
        ]
        self._write_chapter_artifact(artifact_dir, "00_manifest.txt", "\n".join(manifest_lines) + "\n")
        self._write_chapter_artifact(artifact_dir, "01_raw_parser_text.txt", raw_text)
        self._write_chapter_artifact(artifact_dir, "02_tts_source_text.txt", source_text)

        for step_index, (step_name, step_text) in enumerate(normalizer_trace, start=1):
            safe_step_name = step_name.replace(" ", "_").replace("/", "_")
            filename = f"{step_index * 10:02d}_step_{step_index}_{safe_step_name}.txt"
            self._write_chapter_artifact(artifact_dir, filename, step_text)

        self._write_chapter_artifact(artifact_dir, f"99_{final_label}.txt", final_text)
        logger.info("Saved chapter %s text artifacts to %s", idx, artifact_dir)
        return artifact_dir

    def _normalize_with_trace(self, normalizer, text, title, artifact_dir):
        if not normalizer:
            return text, []
        runner = NormalizationPipelineRunner(config=self.config, artifact_dir=artifact_dir)
        return runner.run(normalizer, text, title)

    def _load_prepared_text(self, idx, title):
        if not self.config.prepared_text_folder:
            return None, None

        text_file = self._chapter_text_path(self.config.prepared_text_folder, idx, title)
        if not os.path.exists(text_file):
            raise FileNotFoundError(
                f"Reviewed text file not found for chapter {idx}: {text_file}"
            )

        with open(text_file, "r", encoding="utf-8") as file_handle:
            text = file_handle.read().strip()

        if not text:
            raise ValueError(f"Reviewed text file is empty for chapter {idx}: {text_file}")

        return text, text_file

    def _find_audio_file(self, output_dir, idx, title):
        """Find an existing audio file for a chapter by trying known extensions."""
        for ext in _AUDIO_EXTENSIONS:
            safe_name = make_safe_filename(
                title=title,
                idx=idx,
                output_dir=output_dir,
                ext=ext,
                collision_check=False,
            )
            path = os.path.join(output_dir, safe_name)
            if os.path.exists(path):
                return path
        return None

    def _detect_audio_folder(self) -> str:
        """Return the folder that contains chapter audio files for packaging.

        Priority:
        1. ``config.audio_folder`` — explicit CLI override.
        2. Latest ``wav/NNN/`` subdirectory inside ``output_folder``.
        3. ``output_folder`` root itself.
        """
        if getattr(self.config, "audio_folder", None):
            return self.config.audio_folder

        wav_base = Path(self.config.output_folder) / "wav"
        if wav_base.is_dir():
            run_dirs = sorted(
                p for p in wav_base.iterdir()
                if p.is_dir() and p.name.isdigit()
            )
            if run_dirs:
                found = str(run_dirs[-1])
                logger.info("Auto-detected audio folder: %s", found)
                return found

        return self.config.output_folder

    @staticmethod
    def _scan_audio_files(audio_folder: str):
        """Return (file_path, chapter_title) pairs for all audio files in *audio_folder*,
        sorted by filename. Derives chapter titles from filenames."""
        result = []
        ext_set = set(_AUDIO_EXTENSIONS)
        for entry in sorted(os.scandir(audio_folder), key=lambda e: e.name):
            if not entry.is_file():
                continue
            _, ext = os.path.splitext(entry.name)
            if ext.lower() not in ext_set:
                continue
            # Derive a human-readable title: strip leading NNNN_, strip extension, replace _ with space
            stem = os.path.splitext(entry.name)[0]
            import re as _re
            title = _re.sub(r"^\d+_", "", stem).replace("_", " ")
            result.append((entry.path, title))
        return result

    def _merge_chunks_into_chapters(self, audio_folder: str) -> list[tuple[str, str]]:
        """If no chapter audio files exist, scan for chunked audio and merge by chapter.

        Returns list of (chapter_file_path, chapter_title) for merged chapters.
        """
        chunks_dir = os.path.join(audio_folder, "chunks")
        if not os.path.isdir(chunks_dir):
            return []

        logger.info("No chapter audio files found; scanning for audio chunks in %s", chunks_dir)

        merged = []
        for chapter_dir in sorted(os.scandir(chunks_dir), key=lambda e: e.name):
            if not chapter_dir.is_dir():
                continue

            chapter_key = chapter_dir.name
            chunk_files = []
            for chunk_entry in sorted(os.scandir(chapter_dir.path), key=lambda e: e.name):
                if chunk_entry.is_file():
                    _, ext = os.path.splitext(chunk_entry.name)
                    if ext.lower() in set(_AUDIO_EXTENSIONS):
                        chunk_files.append(chunk_entry.path)

            if not chunk_files:
                continue

            # Merge chunks into a chapter file
            chapter_file = os.path.join(audio_folder, f"{chapter_key}.wav")  # Assume wav for merged
            try:
                from audiobook_generator.core.chunked_audio_generator import _merge_audio_files
                _merge_audio_files(chunk_files, chapter_file)
                # Derive title from chapter_key: replace _ with space, strip numbers if any
                title = chapter_key.replace("_", " ")
                import re as _re
                title = _re.sub(r"^\d+ ", "", title)
                merged.append((chapter_file, title))
                logger.info("Merged %d chunks into chapter: %s", len(chunk_files), chapter_file)
            except Exception as exc:
                logger.warning("Failed to merge chunks for %s: %s", chapter_key, exc)

        return merged

    def _run_package_only(self):
        """Package existing chapter audio files into m4b without running TTS."""
        os.makedirs(self.config.output_folder, exist_ok=True)

        audio_folder = self._detect_audio_folder()
        logger.info("Package mode: scanning audio files in %s", audio_folder)

        if not os.path.isdir(audio_folder):
            logger.error("Audio folder does not exist: %s", audio_folder)
            return

        scanned = self._scan_audio_files(audio_folder)
        if not scanned:
            # Try to merge chunks into chapters
            scanned = self._merge_chunks_into_chapters(audio_folder)
            if not scanned:
                logger.error(
                    "No audio files or chunks found in %s. Run 'audio' mode first, "
                    "or pass --audio_folder to specify the folder explicitly.",
                    audio_folder,
                )
                return

        chapter_files = [p for p, _ in scanned]
        chapter_titles = [t for _, t in scanned]
        logger.info("Found %d audio file(s) to package.", len(chapter_files))

        # Book metadata — only read epub/fb2 if the input file is available.
        book_title = book_author = book_cover = None
        if self.config.input_file and os.path.isfile(self.config.input_file):
            try:
                book_parser = get_book_parser(self.config)
                book_title = book_parser.get_book_title()
                book_author = book_parser.get_book_author()
                book_cover = book_parser.get_book_cover()
            except Exception as exc:
                logger.warning("Could not read book metadata: %s", exc)

        m4b_path = package_m4b(
            chapter_files=chapter_files,
            chapter_titles=chapter_titles,
            book_title=book_title or Path(self.config.input_file or "book").stem,
            book_author=book_author or "",
            output_dir=self.config.output_folder,
            ffmpeg_path=self.config.ffmpeg_path,
            output_filename=self.config.m4b_filename,
            bitrate=self.config.m4b_bitrate,
            cover=book_cover,
        )
        logger.info("✅ Packaged m4b audiobook: %s", m4b_path)

    def process_chapter(self, idx, title, text, book_parser):
        """Process a single chapter: write text (if needed) and convert to audio."""
        try:
            logger.info(f"Processing chapter {idx}: {title}")
            tts_provider = get_tts_provider(self.config)
            normalizer = get_normalizer(self.config) if self.config.normalize else None
            prepared_text, prepared_text_path = self._load_prepared_text(idx, title)
            source_text = prepared_text if prepared_text is not None else text

            if prepared_text_path:
                logger.info("Using reviewed text for chapter %s from %s", idx, prepared_text_path)

            # Determine output dirs for this run
            text_out_dir = self._text_run_dir() if self.config.current_run_index else self.config.output_folder
            wav_out_dir = self._wav_run_dir() if self.config.current_run_index else self.config.output_folder

            # Save chapter text if required
            if self.config.output_text:
                self._write_chapter_text(text_out_dir, idx, title, source_text)

            # Generate audio file (safe, length-limited, cross-platform)
            audio_ext = "." + tts_provider.get_output_file_extension()
            safe_audio_name = make_safe_filename(
                title=title,
                idx=idx,
                output_dir=wav_out_dir,
                ext=audio_ext,
                collision_check=False,
            )
            output_file = os.path.join(wav_out_dir, safe_audio_name)

            audio_tags = AudioTags(
                title, book_parser.get_book_author(), book_parser.get_book_title(), idx
            )
            artifact_dir = self._chapter_artifact_dir(idx, title)
            text_for_tts, tts_trace = self._normalize_with_trace(
                normalizer,
                source_text,
                title,
                artifact_dir,
            )
            final_label = "tts_input"
            if self.config.prepare_text:
                final_label = "prepared_text"
            elif self.config.preview:
                final_label = "preview_tts_input"
            self._save_chapter_artifacts(
                idx=idx,
                title=title,
                raw_text=text,
                source_text=source_text,
                prepared_text_path=prepared_text_path,
                normalizer_trace=tts_trace,
                final_text=text_for_tts,
                final_label=final_label,
            )

            if self.config.prepare_text:
                text_file = self._write_chapter_text(text_out_dir, idx, title, text_for_tts)
                logger.info("Prepared chapter %s text for review: %s", idx, text_file)
                return True

            if self.config.preview:
                text_file = self._write_chapter_text(text_out_dir, idx, title, text_for_tts)
                logger.info("Preview stopped before TTS for chapter %s; final text saved to %s", idx, text_file)
                return True

            os.makedirs(wav_out_dir, exist_ok=True)

            # --- Chunked synthesis (sentence-level resume) ---
            if self.config.chunked_audio:
                from audiobook_generator.core.audio_chunk_store import AudioChunkStore
                from audiobook_generator.core.chunked_audio_generator import ChunkedAudioGenerator
                from audiobook_generator.utils.filename_sanitizer import make_safe_filename as _msf
                run_id = self.config.current_run_index or "000"
                chunk_store_path = os.path.join(self.config.output_folder, "_state", "audio_chunks.sqlite3")
                chunk_store = AudioChunkStore(chunk_store_path)
                chunk_store.ensure_run(run_id)
                chunks_base = os.path.join(wav_out_dir, "chunks")
                # Chapter key = safe directory name
                chapter_key = _msf(title=title, idx=idx, output_dir=chunks_base, ext="", collision_check=False).rstrip(".")
                chunked = ChunkedAudioGenerator(
                    config=self.config,
                    chunk_store=chunk_store,
                    tts_provider=tts_provider,
                    run_id=run_id,
                    chunks_base_dir=chunks_base,
                )
                success = chunked.process_chapter(
                    chapter_idx=idx,
                    chapter_key=chapter_key,
                    text_for_tts=text_for_tts,
                    output_file=output_file,
                    audio_tags=audio_tags,
                )
                if success:
                    logger.info("✅ Converted chapter %d (chunked): %s, output: %s", idx, title, output_file)
                return success

            # --- Standard synthesis (whole chapter in one TTS call) ---
            tts_provider.text_to_speech(text_for_tts, output_file, audio_tags)

            logger.info(f"✅ Converted chapter {idx}: {title}, output file: {output_file}")

            return True
        except Exception as e:
            logger.exception(f"Error processing chapter {idx}, error: {e}")
            return False

    def process_chapter_wrapper(self, args):
        """Wrapper for process_chapter to handle unpacking args for imap."""
        idx, title, text, book_parser = args
        return idx, self.process_chapter(idx, title, text, book_parser)

    def run(self):
        try:
            logger.info("Starting audiobook generation...")

            # Map --mode to internal flags.  Legacy code (mode=None) keeps its existing flags.
            mode = getattr(self.config, 'mode', None)
            if mode == 'prepare':
                self.config.prepare_text = True
                self.config.package_m4b = False
                # Auto-enable normalization when steps or provider are configured.
                if not self.config.normalize and (self.config.normalize_steps or self.config.normalize_provider):
                    self.config.normalize = True
                logger.info("Mode: prepare — parsing + normalizing, writing review .txt files.")
            elif mode == 'audio':
                self.config.prepare_text = False
                self.config.package_m4b = False
                logger.info("Mode: audio — synthesizing audio from text.")
            elif mode == 'package':
                self._run_package_only()
                return
            elif mode == 'all':
                self.config.prepare_text = False
                self.config.package_m4b = True
                # Auto-enable normalization when steps or provider are configured.
                if not self.config.normalize and (self.config.normalize_steps or self.config.normalize_provider):
                    self.config.normalize = True
                logger.info("Mode: all — normalize + synthesize + package.")

            os.makedirs(self.config.output_folder, exist_ok=True)

            # Determine run index for the structured text/ and wav/ layout.
            # Only applies when mode is explicitly set (not legacy mode=None).
            if mode in ('prepare', 'all'):
                run_index = self._next_run_index("text")
            elif mode == 'audio':
                # Use the same index as the latest text run.
                latest_text = self._latest_run_index("text")
                run_index = latest_text if latest_text else "001"
                # Auto-detect prepared_text_folder if not explicitly provided.
                if not self.config.prepared_text_folder and latest_text:
                    auto_text_dir = str(self._run_subdir("text") / latest_text)
                    if os.path.isdir(auto_text_dir):
                        self.config.prepared_text_folder = auto_text_dir
                        logger.info("Auto-detected text source for audio mode: %s", auto_text_dir)
            else:
                run_index = None  # legacy path, no structured layout

            self.config.current_run_index = run_index

            # Set per-run normalization state path so resume works per run.
            if run_index:
                from pathlib import Path
                state_dir = Path(self.config.output_folder) / "text" / run_index / "_state"
                self.config.normalization_state_path = str(
                    state_dir / "normalization_progress.sqlite3"
                )

            # Save INI config snapshot for this run.
            if mode and run_index:
                run_dir = self._text_run_dir() if mode in ('prepare', 'all') else self._wav_run_dir()
                self._save_ini_snapshot(run_dir)

            book_parser = get_book_parser(self.config)
            tts_provider = get_tts_provider(self.config)

            self._copy_input_book()
            if self.config.prepared_text_folder and not os.path.isdir(self.config.prepared_text_folder):
                raise FileNotFoundError(
                    f"Prepared text folder not found: {self.config.prepared_text_folder}"
                )
            if self.config.prepared_text_folder and self.config.normalize:
                logger.warning(
                    "Both --prepared_text_folder and --normalize are enabled. Reviewed text will be normalized again before TTS."
                )
            if self.config.prepare_text:
                logger.info("Prepare-text mode enabled. Chapters will be exported for review and TTS will be skipped.")
            if self.config.prepare_text and self.config.package_m4b:
                logger.warning("Ignoring --package_m4b because --prepare_text is enabled.")
            chapters = book_parser.get_chapters(tts_provider.get_break_string())
            # Filter out empty or very short chapters
            chapters = [(title, text) for title, text in chapters if text.strip()]

            logger.info(f"Chapters count: {len(chapters)}.")

            # Check chapter start and end args
            if self.config.chapter_start < 1 or self.config.chapter_start > len(chapters):
                raise ValueError(
                    f"Chapter start index {self.config.chapter_start} is out of range. Check your input."
                )
            if self.config.chapter_end < -1 or self.config.chapter_end > len(chapters):
                raise ValueError(
                    f"Chapter end index {self.config.chapter_end} is out of range. Check your input."
                )
            if self.config.chapter_end == -1:
                self.config.chapter_end = len(chapters)
            if self.config.chapter_start > self.config.chapter_end:
                raise ValueError(
                    f"Chapter start index {self.config.chapter_start} is larger than chapter end index {self.config.chapter_end}. Check your input."
                )

            logger.info(
                f"Converting chapters from {self.config.chapter_start} to {self.config.chapter_end}."
            )

            # Initialize total_characters to 0
            total_characters = get_total_chars(
                chapters[self.config.chapter_start - 1 : self.config.chapter_end]
            )
            logger.info(f"Total characters in selected book chapters: {total_characters}")
            if not self.config.prepare_text:
                rough_price = tts_provider.estimate_cost(total_characters)
                logger.info(f"Estimate book voiceover would cost you roughly: ${rough_price:.2f}\n")

            # Prompt user to continue if not in preview mode
            if self.config.prepare_text:
                logger.info("Skipping prompt in prepare-text mode")
            elif self.config.no_prompt:
                logger.info("Skipping prompt as passed parameter no_prompt")
            elif self.config.preview:
                logger.info("Skipping prompt as in preview mode")
            else:
                confirm_conversion()

            # Prepare chapters for processing
            chapters_to_process = chapters[self.config.chapter_start - 1 : self.config.chapter_end]
            tasks = [
                (idx, title, text, book_parser)
                for idx, (title, text) in enumerate(
                    chapters_to_process, start=self.config.chapter_start
                )
            ]

            # Build expected audio file paths (used for m4b packaging).
            wav_dir = self._wav_run_dir() if run_index else self.config.output_folder
            chapter_output_records = []
            chapter_audio_ext = "." + tts_provider.get_output_file_extension()
            for idx, (title, _text) in enumerate(
                chapters_to_process, start=self.config.chapter_start
            ):
                chapter_output_records.append(
                    (
                        idx,
                        title,
                        os.path.join(
                            wav_dir,
                            make_safe_filename(
                                title=title,
                                idx=idx,
                                output_dir=wav_dir,
                                ext=chapter_audio_ext,
                                collision_check=False,
                            ),
                        ),
                    )
                )

            # Track failed chapters
            failed_chapters = []

            # Use multiprocessing to process chapters in parallel
            with multiprocessing.Pool(
                processes=self.config.worker_count,
                initializer=setup_logging,
                initargs=(self.config.log, self.config.log_file, True)
            ) as pool:
                # Process chapters and collect results
                results = list(pool.imap_unordered(self.process_chapter_wrapper, tasks))

                # Check for failed chapters
                for idx, success in results:
                    if not success:
                        chapter_title = chapters_to_process[idx - self.config.chapter_start][0]
                        failed_chapters.append((idx, chapter_title))

            if failed_chapters:
                logger.warning("The following chapters failed to convert:")
                for idx, title in failed_chapters:
                    logger.warning(f"  - Chapter {idx}: {title}")
                logger.info(f"Conversion completed with {len(failed_chapters)} failed chapters. Check your output directory: {self.config.output_folder} and log file: {self.config.log_file} for more details.")
            elif self.config.prepare_text:
                text_run_label = f"text/{run_index}" if run_index else ""
                logger.info(
                    "All chapters prepared for review successfully. "
                    "Check: %s",
                    os.path.join(self.config.output_folder, text_run_label) if text_run_label else self.config.output_folder,
                )
            else:
                wav_label = f"wav/{run_index}" if run_index else ""
                logger.info(
                    "All chapters converted successfully. Check: %s",
                    os.path.join(self.config.output_folder, wav_label) if wav_label else self.config.output_folder,
                )

            if (
                self.config.package_m4b
                and not self.config.preview
                and not self.config.prepare_text
                and not failed_chapters
            ):
                chapter_files = [path for _, _, path in chapter_output_records if os.path.exists(path)]
                chapter_titles = [title for _, title, path in chapter_output_records if os.path.exists(path)]
                if len(chapter_files) != len(chapter_output_records):
                    logger.warning("Skipping m4b packaging because not all chapter files were produced.")
                else:
                    m4b_path = package_m4b(
                        chapter_files=chapter_files,
                        chapter_titles=chapter_titles,
                        book_title=book_parser.get_book_title(),
                        book_author=book_parser.get_book_author(),
                        output_dir=self.config.output_folder,
                        ffmpeg_path=self.config.ffmpeg_path,
                        output_filename=self.config.m4b_filename,
                        bitrate=self.config.m4b_bitrate,
                        cover=book_parser.get_book_cover(),
                    )
                    logger.info("Packaged m4b audiobook: %s", m4b_path)

        except KeyboardInterrupt:
            logger.info("Audiobook generation process interrupted by user (Ctrl+C).")
        except Exception as e:
            logger.exception(f"Error during audiobook generation: {e}")
        finally:
            logger.debug("AudiobookGenerator.run() method finished.")
