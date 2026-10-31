import tempfile
import unittest
from pathlib import Path

from rag_im.diff import compare_sections, diff_source, render_report, word_diff
from rag_im.sections import split_text
from rag_im.store import Store
from tests.support import add_au_pair, add_fixture


class TestChangeClassification(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "policy.sqlite3")
        self.old_id, self.new_id = add_au_pair(self.store)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def _changes(self):
        old = self.store.sections_for(self.old_id)
        new = self.store.sections_for(self.new_id)
        return compare_sections(old, new)

    def test_kinds_and_counts(self):
        changes = self._changes()
        kinds = {}
        for change in changes:
            kinds.setdefault(change.kind, []).append(change.heading)
        self.assertEqual(kinds["added"], ["Changes from 1 July 2026"])
        self.assertEqual(kinds["removed"], ["Visa cost"])
        self.assertEqual(sorted(kinds["modified"]), ["Eligibility", "Points requirement"])
        self.assertEqual(kinds["moved"], ["English language requirements"])

    def test_added_and_removed_bodies(self):
        changes = {c.heading: c for c in self._changes()}
        self.assertIn("minimum age for invited applicants", changes["Changes from 1 July 2026"].new_text)
        self.assertIsNone(changes["Changes from 1 July 2026"].old_text)
        self.assertIn("AUD 4,765", changes["Visa cost"].old_text)
        self.assertIsNone(changes["Visa cost"].new_text)

    def test_modified_bodies(self):
        changes = {c.heading: c for c in self._changes()}
        self.assertIn("under 45 years", changes["Eligibility"].old_text)
        self.assertIn("under 40 years", changes["Eligibility"].new_text)
        self.assertIn("65 points", changes["Points requirement"].old_text)
        self.assertIn("67 points", changes["Points requirement"].new_text)

    def test_moved_section_detected_by_content(self):
        moved = [c for c in self._changes() if c.kind == "moved"]
        self.assertEqual(len(moved), 1)
        self.assertIn("English language", moved[0].detail)
        self.assertIn("English language requirements", moved[0].detail)
        # content is identical, only the anchor/heading changed
        self.assertEqual(moved[0].old_hash, moved[0].new_hash)

    def test_word_diff_marks_both_sides(self):
        diff = word_diff("You must score 65 points", "You must score 67 points")
        self.assertIn("[-65-]", diff)
        self.assertIn("{+67+}", diff)
        self.assertIn("You must score", diff)

    def test_identical_sections_produce_no_changes(self):
        sections = split_text("# A\n\nsame text\n\n## B\n\nother")
        self.assertEqual(compare_sections(sections, sections), [])

    def test_whitespace_only_reflow_is_not_a_change(self):
        old = split_text("# A\n\n65 points needed\n\n## B\n\nunchanged")
        new = split_text("# A\n\n  65\tpoints   needed \n\n## B\n\nunchanged")
        self.assertEqual(compare_sections(old, new), [])


class TestReports(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.store = Store(self.tmp_path / "policy.sqlite3")
        add_au_pair(self.store)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_diff_source_writes_markdown_report(self):
        cs = diff_source(self.store, "au_189", _paths(self.tmp_path), write=True)
        self.assertIsNotNone(cs)
        self.assertIsNotNone(cs.report_path)
        text = cs.report_path.read_text(encoding="utf-8")
        self.assertIn("# Policy change report — AU", text)
        self.assertIn("## Added sections (1)", text)
        self.assertIn("## Removed sections (1)", text)
        self.assertIn("## Modified sections (2)", text)
        self.assertIn("## Moved sections (1)", text)
        self.assertIn("2026-10-31T09:00:00Z", text)
        self.assertIn("Word-level", text)

    def test_report_persists_change_rows(self):
        diff_source(self.store, "au_189", _paths(self.tmp_path))
        rows = self.store.changes_for("au_189")
        self.assertEqual(len(rows), 5)
        self.assertIn("modified", {r.kind for r in rows})

    def test_no_changes_report(self):
        from rag_im.diff import ChangeSet

        cs = diff_source(self.store, "au_189", _paths(self.tmp_path))
        cs.changes = []
        rendered = render_report(cs)
        self.assertIn("No section-level changes detected.", rendered)
        self.assertIn("0 added", rendered)

    def test_diff_needs_two_snapshots(self):
        with Store(self.tmp_path / "single.sqlite3") as store:
            add_fixture(store, "au_189", "au_189_v1.html", "2026-07-01T09:00:00Z")
            self.assertIsNone(diff_source(store, "au_189", _paths(self.tmp_path)))


def _paths(tmp: Path):
    from rag_im.config import Paths

    return Paths(
        data_dir=tmp / "data",
        raw_dir=tmp / "data" / "raw",
        reports_dir=tmp / "reports",
        db_path=tmp / "policy.sqlite3",
        sources_path=tmp / "sources.yaml",
    )


if __name__ == "__main__":
    unittest.main()
