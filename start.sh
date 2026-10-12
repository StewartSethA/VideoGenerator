#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if ! command -v uv >/dev/null 2>&1; then
    echo "Bootstrapping uv in $ROOT/.uv-bootstrap"
    python3 -m venv "$ROOT/.uv-bootstrap"
    "$ROOT/.uv-bootstrap/bin/python" -m pip install uv==0.13.0
    PATH="$ROOT/.uv-bootstrap/bin:$PATH"
    export PATH
fi
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1 ||
   { ! command -v espeak >/dev/null 2>&1 && ! command -v espeak-ng >/dev/null 2>&1; }; then
    if command -v apt-get >/dev/null 2>&1; then
        if [ "$(id -u)" = 0 ]; then
            apt-get update
            apt-get install -y ffmpeg espeak-ng
        else
            sudo apt-get update
            sudo apt-get install -y ffmpeg espeak-ng
        fi
    else
        echo "Install FFmpeg and eSpeak/eSpeak NG first (macOS: brew install ffmpeg espeak)." >&2
        exit 1
    fi
fi
if [ "$#" = 0 ]; then set -- serve; fi
exec uv run --project "$ROOT" videogenerator "$@"
