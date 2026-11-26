"""SQLite persistence: sources, snapshots, sections, changes.

All writes are idempotent: re-fetching an unchanged page (same content sha)
returns the existing snapshot instead of creating a duplicate, and re-running a
diff over the same snapshot pair updates rather than duplicates change rows.
"""

from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .sections import Section

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sources (
    id          TEXT PRIMARY KEY,
    country     TEXT NOT NULL,
    name        TEXT NOT NULL,
    url         TEXT NOT NULL,
    selector    TEXT,
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE TABLE IF NOT EXISTS snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id       TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    fetched_at      TEXT NOT NULL,
    url             TEXT NOT NULL,
    raw_path        TEXT,
    content_sha256  TEXT NOT NULL,
    text_sha256     TEXT NOT NULL,
    text            TEXT NOT NULL,
    UNIQUE (source_id, content_sha256)
);

CREATE INDEX IF NOT EXISTS idx_snapshots_source_time
    ON snapshots(source_id, fetched_at DESC, id DESC);

CREATE TABLE IF NOT EXISTS sections (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id  INTEGER NOT NULL REFERENCES snapshots(id) ON DELETE CASCADE,
    ordinal      INTEGER NOT NULL,
    anchor       TEXT NOT NULL,
    heading      TEXT NOT NULL,
    level        INTEGER NOT NULL,
    body         TEXT NOT NULL,
    section_hash TEXT NOT NULL,
    UNIQUE (snapshot_id, anchor)
);

CREATE INDEX IF NOT EXISTS idx_sections_snapshot ON sections(snapshot_id, ordinal);

CREATE TABLE IF NOT EXISTS changes (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id         TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    from_snapshot_id  INTEGER NOT NULL REFERENCES snapshots(id) ON DELETE CASCADE,
    to_snapshot_id    INTEGER NOT NULL REFERENCES snapshots(id) ON DELETE CASCADE,
    kind              TEXT NOT NULL CHECK (kind IN ('added','removed','modified','moved')),
    anchor            TEXT NOT NULL,
    heading           TEXT NOT NULL,
    old_hash          TEXT,
    new_hash          TEXT,
    detail            TEXT,
    detected_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    UNIQUE (source_id, from_snapshot_id, to_snapshot_id, kind, anchor)
);

CREATE INDEX IF NOT EXISTS idx_changes_source ON changes(source_id, detected_at DESC);

-- Dense vectors are expensive and deterministic for a given model + text, so
-- they are cached by content hash and reused across queries and processes.
CREATE TABLE IF NOT EXISTS embeddings (
    model       TEXT NOT NULL,
    text_sha    TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    PRIMARY KEY (model, text_sha)
);

-- One row per source per `fetch` invocation. This is the observability table:
-- success rate, latency, bytes and change counts over time.
CREATE TABLE IF NOT EXISTS fetch_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    source_id    TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    started_at   TEXT NOT NULL,
    finished_at  TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('new','unchanged','error')),
    latency_ms   INTEGER NOT NULL,
    bytes        INTEGER NOT NULL DEFAULT 0,
    sections     INTEGER NOT NULL DEFAULT 0,
    changes      INTEGER NOT NULL DEFAULT 0,
    message      TEXT
);

CREATE INDEX IF NOT EXISTS idx_fetch_runs_source
    ON fetch_runs(source_id, finished_at DESC);
CREATE INDEX IF NOT EXISTS idx_fetch_runs_run
    ON fetch_runs(run_id);
"""


@dataclass(frozen=True)
class Snapshot:
    id: int
    source_id: str
    fetched_at: str
    url: str
    raw_path: str | None
    content_sha256: str
    text_sha256: str
    text: str

    @property
    def short_sha(self) -> str:
        return self.content_sha256[:12]


@dataclass(frozen=True)
class ChangeRow:
    kind: str
    anchor: str
    heading: str
    old_hash: str | None
    new_hash: str | None
    detail: str | None
    detected_at: str


class Store:
    """Thin, explicit wrapper around sqlite3. Use as a context manager."""

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        try:
            self.conn.commit()
        finally:
            self.conn.close()

    # ------------------------------------------------------------------ sources
    def upsert_source(
        self,
        source_id: str,
        country: str,
        name: str,
        url: str,
        selector: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO sources (id, country, name, url, selector)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                country=excluded.country,
                name=excluded.name,
                url=excluded.url,
                selector=excluded.selector,
                updated_at=strftime('%Y-%m-%dT%H:%M:%SZ','now')
            """,
            (source_id, country, name, url, selector),
        )
        self.conn.commit()

    def get_source(self, source_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
        return dict(row) if row else None

    def list_sources(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM sources ORDER BY country, id").fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------------------- snapshots
    def add_snapshot(
        self,
        source_id: str,
        *,
        fetched_at: str,
        url: str,
        content_sha256: str,
        text: str,
        raw_path: str | None,
        sections: Iterable[Section],
    ) -> tuple[int, bool]:
        """Insert a snapshot + its sections. Returns (id, created).

        Same content sha for the same source => no new row, no new sections.
        """
        if self.get_source(source_id) is None:
            raise ValueError(f"unknown source_id {source_id!r}; call upsert_source first")

        text_sha = _sha(text)
        existing = self.conn.execute(
            "SELECT id FROM snapshots WHERE source_id=? AND content_sha256=?",
            (source_id, content_sha256),
        ).fetchone()
        if existing:
            return int(existing["id"]), False

        cur = self.conn.execute(
            """
            INSERT INTO snapshots
                (source_id, fetched_at, url, raw_path, content_sha256, text_sha256, text)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (source_id, fetched_at, url, raw_path, content_sha256, text_sha, text),
        )
        snapshot_id = cur.lastrowid
        assert snapshot_id is not None
        snapshot_id = int(snapshot_id)
        for ordinal, sec in enumerate(sections):
            self.conn.execute(
                """
                INSERT INTO sections
                    (snapshot_id, ordinal, anchor, heading, level, body, section_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_id, anchor) DO UPDATE SET
                    ordinal=excluded.ordinal,
                    heading=excluded.heading,
                    level=excluded.level,
                    body=excluded.body,
                    section_hash=excluded.section_hash
                """,
                (
                    snapshot_id,
                    sec.ordinal if sec.ordinal else ordinal,
                    sec.anchor,
                    sec.heading,
                    sec.level,
                    sec.body,
                    sec.hash,
                ),
            )
        self.conn.commit()
        return snapshot_id, True

    def get_snapshot(self, snapshot_id: int) -> Snapshot | None:
        row = self.conn.execute("SELECT * FROM snapshots WHERE id=?", (snapshot_id,)).fetchone()
        return _row_to_snapshot(row) if row else None

    def latest_snapshot(self, source_id: str) -> Snapshot | None:
        row = self.conn.execute(
            """
            SELECT * FROM snapshots WHERE source_id=?
            ORDER BY fetched_at DESC, id DESC LIMIT 1
            """,
            (source_id,),
        ).fetchone()
        return _row_to_snapshot(row) if row else None

    def latest_two_snapshots(self, source_id: str) -> tuple[Snapshot | None, Snapshot | None]:
        rows = self.conn.execute(
            """
            SELECT * FROM snapshots WHERE source_id=?
            ORDER BY fetched_at DESC, id DESC LIMIT 2
            """,
            (source_id,),
        ).fetchall()
        if not rows:
            return None, None
        if len(rows) == 1:
            return None, _row_to_snapshot(rows[0])
        return _row_to_snapshot(rows[1]), _row_to_snapshot(rows[0])

    def latest_sha(self, source_id: str) -> str | None:
        snap = self.latest_snapshot(source_id)
        return snap.content_sha256 if snap else None

    def snapshot_count(self, source_id: str | None = None) -> int:
        if source_id:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM snapshots WHERE source_id=?", (source_id,)
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) AS n FROM snapshots").fetchone()
        return int(row["n"])

    # ----------------------------------------------------------------- sections
    def sections_for(self, snapshot_id: int) -> list[Section]:
        rows = self.conn.execute(
            "SELECT * FROM sections WHERE snapshot_id=? ORDER BY ordinal, id", (snapshot_id,)
        ).fetchall()
        return [
            Section(
                heading=r["heading"],
                level=int(r["level"]),
                body=r["body"],
                hash=r["section_hash"],
                anchor=r["anchor"],
                ordinal=int(r["ordinal"]),
            )
            for r in rows
        ]

    def latest_sections_by_source(self) -> list[tuple[dict[str, Any], Snapshot, list[Section]]]:
        out = []
        for src in self.list_sources():
            snap = self.latest_snapshot(src["id"])
            if snap is None:
                continue
            out.append((src, snap, self.sections_for(snap.id)))
        return out

    # ------------------------------------------------------------------ changes
    def record_changes(
        self,
        source_id: str,
        from_snapshot_id: int,
        to_snapshot_id: int,
        changes: Iterable[Any],
    ) -> int:
        """Upsert change rows (objects need kind/anchor/heading/old_hash/new_hash/detail)."""
        count = 0
        for ch in changes:
            self.conn.execute(
                """
                INSERT INTO changes
                    (source_id, from_snapshot_id, to_snapshot_id, kind, anchor, heading,
                     old_hash, new_hash, detail)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, from_snapshot_id, to_snapshot_id, kind, anchor)
                DO UPDATE SET
                    heading=excluded.heading,
                    old_hash=excluded.old_hash,
                    new_hash=excluded.new_hash,
                    detail=excluded.detail
                """,
                (
                    source_id,
                    from_snapshot_id,
                    to_snapshot_id,
                    ch.kind,
                    ch.anchor,
                    ch.heading,
                    getattr(ch, "old_hash", None),
                    getattr(ch, "new_hash", None),
                    getattr(ch, "detail", None),
                ),
            )
            count += 1
        self.conn.commit()
        return count

    def changes_for(self, source_id: str, limit: int = 20) -> list[ChangeRow]:
        rows = self.conn.execute(
            """
            SELECT kind, anchor, heading, old_hash, new_hash, detail, detected_at
            FROM changes WHERE source_id=?
            ORDER BY detected_at DESC, id DESC LIMIT ?
            """,
            (source_id, limit),
        ).fetchall()
        return [
            ChangeRow(
                kind=r["kind"],
                anchor=r["anchor"],
                heading=r["heading"],
                old_hash=r["old_hash"],
                new_hash=r["new_hash"],
                detail=r["detail"],
                detected_at=r["detected_at"],
            )
            for r in rows
        ]

    def last_change(self, source_id: str) -> ChangeRow | None:
        rows = self.changes_for(source_id, limit=1)
        return rows[0] if rows else None

    # --------------------------------------------------------------- embeddings
    def get_embeddings(self, model: str, text_hashes: Iterable[str]) -> dict[str, list[float]]:
        """Return cached vectors for the given content hashes, missing keys omitted."""
        keys = list(dict.fromkeys(text_hashes))
        out: dict[str, list[float]] = {}
        # Stay under SQLite's bound-variable limit on large corpora.
        for start in range(0, len(keys), 500):
            batch = keys[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = self.conn.execute(
                f"SELECT text_sha, vector, dim FROM embeddings "
                f"WHERE model=? AND text_sha IN ({placeholders})",
                (model, *batch),
            ).fetchall()
            for row in rows:
                vector = _blob_to_vec(row["vector"], int(row["dim"]))
                if vector:
                    out[row["text_sha"]] = vector
        return out

    def put_embeddings(self, model: str, vectors: dict[str, list[float]]) -> int:
        """Upsert vectors keyed by content hash. Returns the number written."""
        rows = [
            (model, text_sha, len(vec), sqlite3.Binary(_vec_to_blob(vec)))
            for text_sha, vec in vectors.items()
            if vec
        ]
        if not rows:
            return 0
        self.conn.executemany(
            """
            INSERT INTO embeddings (model, text_sha, dim, vector)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(model, text_sha) DO UPDATE SET
                dim=excluded.dim,
                vector=excluded.vector,
                created_at=strftime('%Y-%m-%dT%H:%M:%SZ','now')
            """,
            rows,
        )
        self.conn.commit()
        return len(rows)

    def embedding_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) n FROM embeddings").fetchone()["n"])

    def clear_embeddings(self, model: str | None = None) -> int:
        if model:
            cur = self.conn.execute("DELETE FROM embeddings WHERE model=?", (model,))
        else:
            cur = self.conn.execute("DELETE FROM embeddings")
        self.conn.commit()
        return cur.rowcount

    # --------------------------------------------------------------- fetch runs
    def record_fetch_run(
        self,
        *,
        run_id: str,
        source_id: str,
        started_at: str,
        finished_at: str,
        status: str,
        latency_ms: int,
        bytes: int = 0,
        sections: int = 0,
        changes: int = 0,
        message: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO fetch_runs
                (run_id, source_id, started_at, finished_at, status,
                 latency_ms, bytes, sections, changes, message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                source_id,
                started_at,
                finished_at,
                status,
                latency_ms,
                max(0, bytes),
                max(0, sections),
                max(0, changes),
                message,
            ),
        )
        self.conn.commit()

    def fetch_run_stats(self) -> dict[str, Any]:
        """Aggregate fetch history: success rate, latency percentiles, freshness."""
        rows = self.conn.execute(
            "SELECT * FROM fetch_runs ORDER BY finished_at DESC, id DESC"
        ).fetchall()
        latencies = [int(r["latency_ms"]) for r in rows]
        per_source: dict[str, dict[str, Any]] = {}
        for row in rows:
            sid = row["source_id"]
            entry = per_source.setdefault(
                sid,
                {
                    "source_id": sid,
                    "runs": 0,
                    "ok": 0,
                    "errors": 0,
                    "latencies": [],
                    "last_run": row["finished_at"],
                    "last_status": row["status"],
                    "last_error": None,
                    "changes": 0,
                    "bytes": 0,
                },
            )
            entry["runs"] += 1
            entry["latencies"].append(int(row["latency_ms"]))
            entry["changes"] += int(row["changes"] or 0)
            entry["bytes"] += int(row["bytes"] or 0)
            if row["status"] == "error":
                entry["errors"] += 1
                if entry["last_error"] is None:
                    entry["last_error"] = row["message"]
            else:
                entry["ok"] += 1
        sources = []
        for entry in sorted(per_source.values(), key=lambda e: e["source_id"]):
            values = entry.pop("latencies")
            entry["p50_ms"] = round(_percentile(values, 0.50))
            entry["p95_ms"] = round(_percentile(values, 0.95))
            sources.append(entry)
        return {
            "runs": len(rows),
            "ok": sum(1 for r in rows if r["status"] != "error"),
            "errors": sum(1 for r in rows if r["status"] == "error"),
            "p50_ms": round(_percentile(latencies, 0.50)),
            "p95_ms": round(_percentile(latencies, 0.95)),
            "last_run": rows[0]["finished_at"] if rows else None,
            "sources": sources,
        }

    def stats(self) -> dict[str, int]:
        return {
            "sources": int(self.conn.execute("SELECT COUNT(*) n FROM sources").fetchone()["n"]),
            "snapshots": int(self.conn.execute("SELECT COUNT(*) n FROM snapshots").fetchone()["n"]),
            "sections": int(self.conn.execute("SELECT COUNT(*) n FROM sections").fetchone()["n"]),
            "changes": int(self.conn.execute("SELECT COUNT(*) n FROM changes").fetchone()["n"]),
            "embeddings": self.embedding_count(),
            "fetch_runs": int(self.conn.execute("SELECT COUNT(*) n FROM fetch_runs").fetchone()["n"]),
        }


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _vec_to_blob(vec: list[float]) -> bytes:
    """Little-endian float32 blob; explicit byte order keeps the DB portable."""
    return struct.pack(f"<{len(vec)}f", *(float(x) for x in vec))


def _blob_to_vec(blob: bytes, dim: int) -> list[float]:
    if dim <= 0:
        return []
    try:
        return list(struct.unpack(f"<{dim}f", bytes(blob)))
    except struct.error:
        return []


def _percentile(values: list[int], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 1]. No numpy dependency."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def _row_to_snapshot(row: sqlite3.Row) -> Snapshot:
    return Snapshot(
        id=int(row["id"]),
        source_id=row["source_id"],
        fetched_at=row["fetched_at"],
        url=row["url"],
        raw_path=row["raw_path"],
        content_sha256=row["content_sha256"],
        text_sha256=row["text_sha256"],
        text=row["text"],
    )
