"""Seed the local database from fixture pages, then run a diff -- fully offline.

    /home/louis/.venvs/main/bin/python scripts/seed_demo.py [--reset]

Useful for demos and for eyeballing the report format without hitting the
network. Real use is `python -m rag_im fetch`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_im.config import ensure_dirs, get_paths  # noqa: E402
from rag_im.diff import diff_source  # noqa: E402
from rag_im.store import Store  # noqa: E402
from tests.support import add_au_pair, add_nl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="delete the existing demo DB first")
    args = parser.parse_args()

    paths = get_paths()
    ensure_dirs(paths)
    if args.reset and paths.db_path.exists():
        paths.db_path.unlink()

    with Store(paths.db_path) as store:
        add_au_pair(store)
        add_nl(store)
        print(f"DB: {paths.db_path}")
        print(f"sources={store.stats()['sources']} snapshots={store.stats()['snapshots']}")
        cs = diff_source(store, "au_189", paths)
        if cs is None or cs.report_path is None:
            print("no diff (need two snapshots)")
            return 1
        print(f"report: {cs.report_path}")
        print()
        print(cs.report_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
