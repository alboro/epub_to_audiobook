#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_FILE="${1:-examples/The_Life_and_Adventures_of_Robinson_Crusoe.epub}"
OUTPUT_DIR="${2:-/tmp/epub_to_audiobook_preview}"
VOICE_NAME="${VOICE_NAME:-en-US-AriaNeural}"

cd "${REPO_DIR}"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Virtual environment not found. Run: bash scripts/setup_macos.sh"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

exec .venv/bin/python main.py \
  "${INPUT_FILE}" \
  "${OUTPUT_DIR}" \
  --preview \
  --tts edge \
  --voice_name "${VOICE_NAME}" \
  --no_prompt \
  "${@:3}"
