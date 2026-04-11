from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_NORMALIZE_STEPS = "simple_symbols,numbers_ru,llm,simple_symbols,tts_safe_split"
DEFAULT_NORMALIZE_MODEL = "gpt-5.4"
DEFAULT_VOICE_NAME = "reference_long"
DEFAULT_TTS_BASE_URL = "http://127.0.0.1:8020"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simplified Russian XTTS recipe for epub_to_audiobook.",
    )
    parser.add_argument("book_path", help="Path to the EPUB book.")
    parser.add_argument("--language", default="ru-RU", help="Book language. Default: ru-RU.")
    parser.add_argument("--output-dir", help="Optional output directory.")
    parser.add_argument(
        "--voice-name",
        default=DEFAULT_VOICE_NAME,
        help=f"Voice name/prefix. Default: {DEFAULT_VOICE_NAME}.",
    )
    parser.add_argument(
        "--tts-base-url",
        default=DEFAULT_TTS_BASE_URL,
        help=f"Polling TTS server base URL. Default: {DEFAULT_TTS_BASE_URL}.",
    )
    parser.add_argument(
        "--normalize-steps",
        default=DEFAULT_NORMALIZE_STEPS,
        help=f"Normalizer chain. Default: {DEFAULT_NORMALIZE_STEPS}.",
    )
    parser.add_argument(
        "--normalize-model",
        default=DEFAULT_NORMALIZE_MODEL,
        help=f"Normalizer model. Default: {DEFAULT_NORMALIZE_MODEL}.",
    )
    parser.add_argument(
        "--normalize-base-url",
        default=os.getenv("NORMALIZER_OPENAI_BASE_URL"),
        help="Normalizer base URL. Defaults to NORMALIZER_OPENAI_BASE_URL.",
    )
    parser.add_argument("--normalize-system-prompt-file", help="Optional custom system prompt file.")
    parser.add_argument(
        "--normalize-max-chars",
        type=int,
        default=4000,
        help="Max chars per LLM normalization request. Default: 4000.",
    )
    parser.add_argument(
        "--normalize-tts-safe-max-chars",
        type=int,
        default=160,
        help="Deterministic safe split max chars. Default: 160.",
    )
    parser.add_argument("--worker-count", type=int, default=1, help="Worker count. Default: 1.")
    parser.add_argument("--poll-interval", type=int, default=5, help="Polling interval in seconds. Default: 5.")
    parser.add_argument("--poll-timeout", type=int, default=14400, help="Polling timeout in seconds. Default: 14400.")
    parser.add_argument("--ffmpeg-path", help="Optional ffmpeg path. Defaults to auto-detection.")
    parser.add_argument("--prepare-text", action="store_true", help="Prepare reviewed text instead of generating audio.")
    parser.add_argument("--package-m4b", action="store_true", help="Force m4b packaging. Enabled by default unless --prepare-text is set.")
    return parser


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_python(project_root: Path) -> Path:
    if os.name == "nt":
        candidate = project_root / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = project_root / ".venv" / "bin" / "python"
    if not candidate.is_file():
        raise FileNotFoundError(f"Virtual environment not found: {candidate}")
    return candidate


def resolve_output_dir(project_root: Path, book_path: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    safe_name = "".join(char if (char.isalnum() or char in "._ -") else "_" for char in book_path.stem).strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (project_root / "out" / f"{safe_name}_{timestamp}").resolve()


def resolve_ffmpeg(explicit: str | None) -> str:
    if explicit:
        return explicit

    discovered = shutil.which("ffmpeg")
    if discovered:
        return discovered

    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            candidates = [
                Path(local_app_data)
                / "Microsoft"
                / "WinGet"
                / "Packages"
                / "BtbN.FFmpeg.LGPL.Shared.8.1_Microsoft.Winget.Source_8wekyb3d8bbwe"
                / "ffmpeg-n8.1-7-ga3475e2554-win64-lgpl-shared-8.1"
                / "bin"
                / "ffmpeg.exe",
                Path("C:/ffmpeg/bin/ffmpeg.exe"),
                Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe"),
            ]
            for candidate in candidates:
                if candidate.is_file():
                    return str(candidate)

    return "ffmpeg"


def main() -> int:
    args = build_parser().parse_args()
    project_root = resolve_project_root()
    python_exe = resolve_python(project_root)
    book_path = Path(args.book_path).expanduser().resolve()
    if not book_path.is_file():
        raise FileNotFoundError(f"Book not found: {book_path}")

    output_dir = resolve_output_dir(project_root, book_path, args.output_dir)
    ffmpeg_path = resolve_ffmpeg(args.ffmpeg_path)

    command = [
        str(python_exe),
        "main.py",
        str(book_path),
        str(output_dir),
        "--tts",
        "openai",
        "--language",
        args.language,
        "--worker_count",
        str(args.worker_count),
        "--no_prompt",
        "--model_name",
        "tts_models/multilingual/multi-dataset/xtts_v2",
        "--voice_name",
        args.voice_name,
        "--output_format",
        "wav",
        "--openai_base_url",
        args.tts_base_url,
        "--openai_enable_polling",
        "--openai_submit_url",
        "/v1/tts/jobs",
        "--openai_status_url_template",
        "/v1/tts/jobs/{job_id}",
        "--openai_download_url_template",
        "/v1/tts/jobs/{job_id}/audio",
        "--openai_job_id_path",
        "id",
        "--openai_job_status_path",
        "status",
        "--openai_poll_interval",
        str(args.poll_interval),
        "--openai_poll_timeout",
        str(args.poll_timeout),
        "--openai_max_chars",
        "0",
        "--normalize",
        "--normalize_steps",
        args.normalize_steps,
        "--normalize_model",
        args.normalize_model,
        "--normalize_max_chars",
        str(args.normalize_max_chars),
        "--normalize_tts_safe_max_chars",
        str(args.normalize_tts_safe_max_chars),
        "--ffmpeg_path",
        ffmpeg_path,
    ]

    if args.normalize_base_url:
        command += ["--normalize_base_url", args.normalize_base_url]
    if args.normalize_system_prompt_file:
        command += ["--normalize_system_prompt_file", args.normalize_system_prompt_file]

    if args.prepare_text:
        command.append("--prepare_text")
    else:
        command.append("--package_m4b")

    print(f"Book: {book_path}")
    print(f"Language: {args.language}")
    print(f"Output: {output_dir}")
    print(f"Voice: {args.voice_name}")
    print(f"TTS server: {args.tts_base_url}")
    print(f"Normalizer steps: {args.normalize_steps}")
    if args.normalize_base_url:
        print(f"Normalizer base URL: {args.normalize_base_url}")
    if args.normalize_system_prompt_file:
        print(f"Prompt file: {args.normalize_system_prompt_file}")
    print(f"ffmpeg: {ffmpeg_path}")
    print("Mode: prepare_text" if args.prepare_text else "Mode: full audiobook")

    return subprocess.call(command, cwd=project_root, env=os.environ.copy())


if __name__ == "__main__":
    raise SystemExit(main())
