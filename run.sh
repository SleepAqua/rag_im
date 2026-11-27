#!/usr/bin/env bash
# Convenience wrapper for a checkout that has its dependencies installed.
# Override the interpreter with PYTHON=/path/to/python if needed.
set -euo pipefail
cd "$(dirname "$0")"
# Machine-local defaults (provider, model, endpoint). Never committed.
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi
if [ -z "${PYTHON:-}" ] && [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Interpreter not found: $PYTHON" >&2
  echo "Set PYTHON=/path/to/python, or install the package with: python -m pip install -e ." >&2
  exit 1
fi
case "${1:-}" in
  test)      shift; exec "$PYTHON" -m unittest discover -s tests -v "$@" ;;
  seed-demo) shift; exec "$PYTHON" scripts/seed_demo.py "$@" ;;
  "")        exec "$PYTHON" -m rag_im --help ;;
  *)         exec "$PYTHON" -m rag_im "$@" ;;
esac
