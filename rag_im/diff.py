"""Compare the two most recent snapshots of a source and write a report.

Classification:
  added     -- anchor exists only in the new snapshot
  removed   -- anchor exists only in the old snapshot
  modified  -- same anchor, different (whitespace-insensitive) section hash
  moved     -- removed and added sections share a hash prefix, i.e. the text is
               the same but it now lives under a different heading/anchor

Modified sections get a word-level difflib diff so a report shows exactly which
number or phrase changed, not a wall of reflowed text.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .config import Paths
from .sections import Section, normalise
from .store import Snapshot, Store

HASH_PREFIX = 8
SNIPPET_CHARS = 420


@dataclass
class Change:
    kind: str
    anchor: str
    heading: str
    old_text: str | None = None
    new_text: str | None = None
    old_hash: str | None = None
    new_hash: str | None = None
    detail: str = ""

    @property
    def title(self) -> str:
        return self.heading or "(preamble)"


@dataclass
class ChangeSet:
    source_id: str
    source_name: str
    country: str
    url: str
    old: Snapshot
    new: Snapshot
    changes: list[Change] = field(default_factory=list)
    report_path: Path | None = None

    def of_kind(self, kind: str) -> list[Change]:
        return [c for c in self.changes if c.kind == kind]

    @property
    def counts(self) -> dict[str, int]:
        out = {"added": 0, "removed": 0, "modified": 0, "moved": 0}
        for change in self.changes:
            out[change.kind] = out.get(change.kind, 0) + 1
        return out


def word_diff(old: str, new: str) -> str:
    """Inline word-level diff: ``[-gone-]`` / ``{+new+}``."""
    a, b = normalise(old).split(), normalise(new).split()
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(" ".join(a[i1:i2]))
        elif tag == "delete":
            parts.append("[-" + " ".join(a[i1:i2]) + "-]")
        elif tag == "insert":
            parts.append("{+" + " ".join(b[j1:j2]) + "+}")
        else:
            parts.append("[-" + " ".join(a[i1:i2]) + "-]{+" + " ".join(b[j1:j2]) + "+}")
    return " ".join(p for p in parts if p)


def compare_sections(old_sections: list[Section], new_sections: list[Section]) -> list[Change]:
    old_by_anchor = {s.anchor: s for s in old_sections}
    new_by_anchor = {s.anchor: s for s in new_sections}

    changes: list[Change] = []

    # Modified: same anchor, different hash.
    for anchor, old_sec in old_by_anchor.items():
        new_sec = new_by_anchor.get(anchor)
        if new_sec is None or new_sec.hash == old_sec.hash:
            continue
        changes.append(
            Change(
                kind="modified",
                anchor=anchor,
                heading=new_sec.heading or old_sec.heading,
                old_text=old_sec.body,
                new_text=new_sec.body,
                old_hash=old_sec.hash,
                new_hash=new_sec.hash,
                detail=word_diff(old_sec.body, new_sec.body),
            )
        )

    removed = [s for a, s in old_by_anchor.items() if a not in new_by_anchor]
    added = [s for a, s in new_by_anchor.items() if a not in old_by_anchor]

    # Moved: same content (hash prefix) but a different anchor. Consume pairs.
    moved_added: set[str] = set()
    moved_removed: set[str] = set()
    for old_sec in removed:
        for new_sec in added:
            if new_sec.anchor in moved_added:
                continue
            if old_sec.hash[:HASH_PREFIX] == new_sec.hash[:HASH_PREFIX]:
                changes.append(
                    Change(
                        kind="moved",
                        anchor=new_sec.anchor,
                        heading=new_sec.heading or old_sec.heading,
                        old_text=old_sec.body,
                        new_text=new_sec.body,
                        old_hash=old_sec.hash,
                        new_hash=new_sec.hash,
                        detail=f"moved: '{old_sec.heading}' -> '{new_sec.heading}'",
                    )
                )
                moved_added.add(new_sec.anchor)
                moved_removed.add(old_sec.anchor)
                break

    for sec in removed:
        if sec.anchor in moved_removed:
            continue
        changes.append(
            Change(
                kind="removed",
                anchor=sec.anchor,
                heading=sec.heading,
                old_text=sec.body,
                old_hash=sec.hash,
                detail="section no longer present",
            )
        )
    for sec in added:
        if sec.anchor in moved_added:
            continue
        changes.append(
            Change(
                kind="added",
                anchor=sec.anchor,
                heading=sec.heading,
                new_text=sec.body,
                new_hash=sec.hash,
                detail="new section",
            )
        )

    order = {"added": 0, "removed": 1, "modified": 2, "moved": 3}
    changes.sort(key=lambda c: (order.get(c.kind, 9), c.anchor))
    return changes


def _snippet(text: str | None, limit: int = SNIPPET_CHARS) -> str:
    if not text:
        return "_(empty)_"
    parts = []
    for line in text.splitlines():
        flat = normalise(line)
        if not flat:
            continue
        parts.append("• " + flat[2:] if flat.startswith("- ") else flat)
    flat = " ".join(parts)
    if not flat:
        return "_(empty)_"
    if len(flat) <= limit:
        return flat
    return flat[:limit].rstrip() + " …"


def render_report(cs: ChangeSet, generated_at: str | None = None) -> str:
    generated = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    counts = cs.counts
    lines = [
        f"# Policy change report — {cs.country} · {cs.source_name}",
        "",
        f"- Source: `{cs.source_id}`",
        f"- URL: {cs.url}",
        f"- Previous snapshot: {cs.old.fetched_at} (`{cs.old.short_sha}`)",
        f"- Current snapshot: {cs.new.fetched_at} (`{cs.new.short_sha}`)",
        f"- Generated: {generated}",
        (
            "- Summary: "
            f"{counts['added']} added, {counts['removed']} removed, "
            f"{counts['modified']} modified, {counts['moved']} moved"
        ),
        "",
    ]

    if not cs.changes:
        lines += ["No section-level changes detected.", ""]
        return "\n".join(lines)

    def section_for(kind: str, title: str) -> None:
        block = cs.of_kind(kind)
        if not block:
            return
        lines.extend([f"## {title}", ""])
        for change in block:
            lines.append(f"### {change.title}")
            lines.append("")
            if kind == "modified":
                lines.append("```diff")
                lines.append("- " + _snippet(change.old_text))
                lines.append("+ " + _snippet(change.new_text))
                lines.append("```")
                lines.append("")
                lines.append(f"Word-level: `{change.detail}`")
                lines.append("")
            elif kind == "added":
                lines.append(f"Before: _(not present)_")
                lines.append("")
                lines.append(f"After: {_snippet(change.new_text)}")
                lines.append("")
            elif kind == "removed":
                lines.append(f"Before: {_snippet(change.old_text)}")
                lines.append("")
                lines.append("After: _(section removed)_")
                lines.append("")
            else:  # moved
                lines.append(f"Note: {change.detail}")
                lines.append("")
                lines.append(f"Content (unchanged): {_snippet(change.new_text)}")
                lines.append("")

    section_for("added", f"Added sections ({counts['added']})")
    section_for("removed", f"Removed sections ({counts['removed']})")
    section_for("modified", f"Modified sections ({counts['modified']})")
    section_for("moved", f"Moved sections ({counts['moved']})")
    return "\n".join(lines).rstrip() + "\n"


def write_report(cs: ChangeSet, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = re.sub(r"[^0-9TZ]", "", cs.new.fetched_at) or "undated"
    path = reports_dir / f"{cs.source_id}-{stamp}.md"
    if path.exists():
        path = reports_dir / f"{cs.source_id}-{stamp}-{cs.new.short_sha}.md"
    path.write_text(render_report(cs), encoding="utf-8")
    return path


def diff_source(
    store: Store,
    source_id: str,
    paths: Paths | None = None,
    *,
    write: bool = True,
) -> ChangeSet | None:
    """Diff the two most recent snapshots. ``None`` when <2 snapshots exist."""
    paths = paths or config.get_paths()
    old, new = store.latest_two_snapshots(source_id)
    if old is None or new is None:
        return None
    source = store.get_source(source_id) or {}
    cs = ChangeSet(
        source_id=source_id,
        source_name=source.get("name", source_id),
        country=source.get("country", ""),
        url=new.url,
        old=old,
        new=new,
        changes=compare_sections(store.sections_for(old.id), store.sections_for(new.id)),
    )
    store.record_changes(source_id, old.id, new.id, cs.changes)
    if write:
        cs.report_path = write_report(cs, paths.reports_dir)
    return cs
