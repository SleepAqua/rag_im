import tempfile
import unittest
from pathlib import Path

from rag_im.store import Store
from tests.support import add_au_pair, add_fixture


class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "policy.sqlite3"
        self.store = Store(self.db)
        self.store.upsert_source("au_189", "AU", "Skilled Independent 189", "https://example.gov.au/189")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_source_upsert_is_idempotent(self):
        self.store.upsert_source("au_189", "AU", "Renamed", "https://example.gov.au/189", "#content")
        rows = self.store.list_sources()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "Renamed")
        self.assertEqual(rows[0]["selector"], "#content")

    def test_add_snapshot_same_hash_returns_existing(self):
        first, sections, _ = add_fixture(self.store, "au_189", "au_189_v1.html", "2026-07-01T09:00:00Z")
        second, created = self.store.add_snapshot(
            "au_189",
            fetched_at="2026-07-02T09:00:00Z",
            url="https://example.gov.au/189",
            content_sha256=self.store.get_snapshot(first).content_sha256,
            text="ignored",
            raw_path=None,
            sections=sections,
        )
        self.assertEqual(first, second)
        self.assertFalse(created)
        self.assertEqual(self.store.snapshot_count("au_189"), 1)
        self.assertEqual(len(self.store.sections_for(first)), len(sections))

    def test_sections_roundtrip(self):
        snapshot_id, sections, text = add_fixture(
            self.store, "au_189", "au_189_v1.html", "2026-07-01T09:00:00Z"
        )
        stored = self.store.sections_for(snapshot_id)
        self.assertEqual([s.anchor for s in stored], [s.anchor for s in sections])
        self.assertEqual([s.hash for s in stored], [s.hash for s in sections])
        snap = self.store.get_snapshot(snapshot_id)
        self.assertEqual(snap.text, text)
        self.assertEqual(len(snap.content_sha256), 64)

    def test_latest_two_ordering(self):
        old, new = add_au_pair(self.store)
        first, second = self.store.latest_two_snapshots("au_189")
        self.assertEqual(first.id, old)
        self.assertEqual(second.id, new)
        self.assertEqual(second.fetched_at, "2026-10-31T09:00:00Z")

    def test_latest_two_with_single_snapshot(self):
        add_fixture(self.store, "au_189", "au_189_v1.html", "2026-07-01T09:00:00Z")
        old, new = self.store.latest_two_snapshots("au_189")
        self.assertIsNone(old)
        self.assertIsNotNone(new)

    def test_unknown_source_rejected(self):
        with self.assertRaises(ValueError):
            self.store.add_snapshot(
                "nope",
                fetched_at="2026-01-01T00:00:00Z",
                url="https://example.gov.au",
                content_sha256="0" * 64,
                text="x",
                raw_path=None,
                sections=[],
            )

    def test_change_records_are_idempotent(self):
        class Change:
            kind = "modified"
            anchor = "points-requirement"
            heading = "Points requirement"
            old_hash = "a" * 64
            new_hash = "b" * 64
            detail = "65 -> 67"

        old, new = add_au_pair(self.store)
        self.store.record_changes("au_189", old, new, [Change()])
        self.store.record_changes("au_189", old, new, [Change()])
        self.assertEqual(len(self.store.changes_for("au_189")), 1)
        self.assertEqual(self.store.last_change("au_189").heading, "Points requirement")

    def test_stats(self):
        add_au_pair(self.store)
        stats = self.store.stats()
        self.assertEqual(stats["sources"], 1)
        self.assertEqual(stats["snapshots"], 2)
        self.assertGreater(stats["sections"], 10)
        self.assertEqual(stats["changes"], 0)


if __name__ == "__main__":
    unittest.main()
