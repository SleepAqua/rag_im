#!/usr/bin/env bash
# Always run this project with the interpreter that has its dependencies.
# Bare `python` on this machine can resolve to a different venv (e.g. a tool
# venv without beautifulsoup4), so the path is pinned here.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-/home/louis/.venvs/main/bin/python}"
if [ ! -x "$PYTHON" ]; then
  echo "Interpreter not found: $PYTHON" >&2
  echo "Set PYTHON=/path/to/python and retry." >&2
  exit 1
fi
case "${1:-}" in
  test)      shift; exec "$PYTHON" -m unittest discover -s tests -v "$@" ;;
  seed-demo) shift; exec "$PYTHON" scripts/seed_demo.py "$@" ;;
  "")        exec "$PYTHON" -m rag_im --help ;;
  *)         exec "$PYTHON" -m rag_im "$@" ;;
esac
