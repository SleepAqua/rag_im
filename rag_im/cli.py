"""Command line interface: ``python -m rag_im fetch|diff|search|ask|status``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config
from .ask import answer_question
from .diff import diff_source
from .fetch import fetch_all
from .retrieval import search as run_search
from .store import Store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag_im",
        description="Monitor government immigration-policy pages: snapshot, diff, search, ask.",
    )
    parser.add_argument("--data-dir", help="data directory (default: ./data or $RAG_IM_DATA)")
    parser.add_argument("--db", help="sqlite path (default: <data>/policy.sqlite3)")
    parser.add_argument("--sources", help="sources.yaml path (default: ./sources.yaml)")
    parser.add_argument("--reports", help="report directory (default: ./reports)")

    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download sources and store new snapshots")
    fetch.add_argument("--source", action="append", dest="only", metavar="ID",
                       help="limit to a source id (repeatable)")
    fetch.add_argument("--force", action="store_true", help="store even if the hash is unchanged")

    diff = sub.add_parser("diff", help="compare the two most recent snapshots")
    diff.add_argument("--source", action="append", dest="only", metavar="ID")
    diff.add_argument("--no-write", action="store_true", help="print only, do not write a report")

    search = sub.add_parser("search", help="rank passages for a query")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--source", action="append", dest="only", metavar="ID")

    ask = sub.add_parser("ask", help="answer a question from stored passages")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--extractive", action="store_true", help="skip the LLM even if configured")

    sub.add_parser("status", help="show stored sources, snapshots and last change")
    return parser


def _paths(args) -> config.Paths:
    base = config.get_paths(data_dir=args.data_dir, db_path=args.db)
    if args.sources:
        base = config.Paths(
            data_dir=base.data_dir,
            raw_dir=base.raw_dir,
            reports_dir=base.reports_dir,
            db_path=base.db_path,
            sources_path=Path(args.sources).expanduser().resolve(),
        )
    if args.reports:
        base = config.Paths(
            data_dir=base.data_dir,
            raw_dir=base.raw_dir,
            reports_dir=Path(args.reports).expanduser().resolve(),
            db_path=base.db_path,
            sources_path=base.sources_path,
        )
    return base


def cmd_fetch(args, paths: config.Paths, store: Store) -> int:
    try:
        sources = config.load_sources(paths.sources_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    config.ensure_dirs(paths)
    summaries = fetch_all(sources, store, paths, only=args.only, force=args.force)
    if not summaries:
        print("No matching sources.")
        return 1
    failures = 0
    for summary in summaries:
        if summary.status == "error":
            failures += 1
            print(f"  ✗ {summary.source_id}: {summary.message}")
        elif summary.status == "unchanged":
            print(f"  = {summary.source_id}: unchanged ({summary.message})")
        else:
            extra = f" -> {summary.report}" if summary.report else ""
            print(f"  + {summary.source_id}: snapshot #{summary.snapshot_id}, {summary.message}{extra}")
    print(f"\n{len(summaries) - failures} ok, {failures} failed. DB: {paths.db_path}")
    return 0 if failures == 0 else 1


def cmd_diff(args, paths: config.Paths, store: Store) -> int:
    source_ids = args.only or [s["id"] for s in store.list_sources()]
    if not source_ids:
        print("No sources stored yet. Run `python -m rag_im fetch` first.")
        return 1
    quiet = args.no_write
    found = 0
    for source_id in source_ids:
        cs = diff_source(store, source_id, paths, write=not quiet)
        if cs is None:
            count = store.snapshot_count(source_id)
            print(f"{source_id}: need two snapshots to diff (have {count}).")
            continue
        found += 1
        counts = cs.counts
        print(
            f"{source_id}: {cs.old.fetched_at} -> {cs.new.fetched_at} | "
            f"+{counts['added']} -{counts['removed']} ~{counts['modified']} >{counts['moved']}"
        )
        for change in cs.changes:
            print(f"    {change.kind:<8} {change.title}")
        if cs.report_path:
            print(f"    report: {cs.report_path}")
    return 0 if found else 1


def cmd_search(args, paths: config.Paths, store: Store) -> int:
    hits = run_search(
        args.query,
        store,
        top_k=args.top_k,
        source_ids=args.only,
        settings=config.llm_settings(),
    )
    if not hits:
        print("No matching passages. Fetch sources first, or try other words.")
        return 1
    for idx, hit in enumerate(hits, start=1):
        p = hit.passage
        print(f"{idx}. score {hit.score:6.3f}  {p.country} · {p.source_name} · {p.heading}")
        print(f"   {' '.join(p.text.split())[:240]}…")
        print(f"   {p.url}")
    return 0


def cmd_ask(args, paths: config.Paths, store: Store) -> int:
    answer = answer_question(
        args.question,
        store,
        top_k=args.top_k,
        settings=config.llm_settings(),
        use_llm=not args.extractive,
    )
    print(f"Q: {args.question}\n")
    print(answer.text)
    if answer.mode in ("extractive", "llm"):
        print("\nCitations:")
        for idx, hit in enumerate(answer.passages, start=1):
            p = hit.passage
            print(f"  [{idx}] {p.citation} — {p.url}")
    print(f"\n(mode: {answer.mode})")
    return 0 if answer.mode != "none" else 1


def cmd_status(args, paths: config.Paths, store: Store) -> int:
    stats = store.stats()
    print(f"DB: {paths.db_path}")
    print(
        f"sources={stats['sources']} snapshots={stats['snapshots']} "
        f"sections={stats['sections']} changes={stats['changes']}"
    )
    sources = store.list_sources()
    if not sources:
        print("\nNothing stored yet. Run `python -m rag_im fetch`.")
        return 0
    print(f"\n{'ID':<14}{'CC':<5}{'SNAPS':>6}  {'LAST FETCH':<21}LAST CHANGE")
    for source in sources:
        latest = store.latest_snapshot(source["id"])
        last_change = store.last_change(source["id"])
        change_text = f"{last_change.kind}: {last_change.heading}" if last_change else "-"
        print(
            f"{source['id']:<14}{source['country']:<5}"
            f"{store.snapshot_count(source['id']):>6}  "
            f"{(latest.fetched_at if latest else '-'):<21}{change_text}"
        )
    return 0


COMMANDS = {
    "fetch": cmd_fetch,
    "diff": cmd_diff,
    "search": cmd_search,
    "ask": cmd_ask,
    "status": cmd_status,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = _paths(args)
    config.ensure_dirs(paths)
    with Store(paths.db_path) as store:
        return COMMANDS[args.command](args, paths, store)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
