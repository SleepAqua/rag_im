"""Entry point for ``python -m rag_im`` (usually: ``./run.sh``)."""

try:
    from .cli import main
except ModuleNotFoundError as exc:  # wrong interpreter, not a broken project
    raise SystemExit(
        f"Missing dependency {exc.name!r}. Run with the pinned interpreter:\n"
        "    ./run.sh <command>            (e.g. ./run.sh fetch, ./run.sh test)\n"
        "    PYTHON=/path/to/python ./run.sh <command>   to override"
    ) from None

if __name__ == "__main__":
    raise SystemExit(main())
