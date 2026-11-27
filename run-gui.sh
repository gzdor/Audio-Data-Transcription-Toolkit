#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo "  Audio Data Transcription Toolkit (.wav, .mp3, .mp4, etc.)"
echo "========================================================================"

PYTHON_BIN="python3"
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
fi

echo "[*] Using Python runtime: $PYTHON_BIN"
if ! "$PYTHON_BIN" -c "import faster_whisper, docx, av, numpy, lxml" >/dev/null 2>&1; then
    echo "[*] Installing required packages from requirements.txt..."
    "$PYTHON_BIN" -m pip install --disable-pip-version-check -r "$SCRIPT_DIR/requirements.txt"
fi

echo "[OK] Launching Audio Data Transcription Toolkit GUI..."
exec "$PYTHON_BIN" "$SCRIPT_DIR/gui.py" "$@"

