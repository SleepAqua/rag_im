import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from rag_im import cli
from rag_im.store import Store
from tests.support import add_au_pair


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


class TestCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = self.root / "policy.sqlite3"
        self.reports = self.root / "reports"
        self.common = ["--data-dir", str(self.root / "data"), "--db", str(self.db), "--reports", str(self.reports)]

    def tearDown(self):
        self.tmp.cleanup()

    def seed(self):
        with Store(self.db) as store:
            add_au_pair(store)

    def test_status_on_empty_db(self):
        code, out, _ = run(self.common + ["status"])
        self.assertEqual(code, 0)
        self.assertIn("sources=0", out)
        self.assertIn("Nothing stored yet", out)

    def test_status_lists_sources(self):
        self.seed()
        code, out, _ = run(self.common + ["status"])
        self.assertEqual(code, 0)
        self.assertIn("au_189", out)
        self.assertIn("snapshots=2", out)
        self.assertIn("model:", out)

    def test_search_prints_ranked_passages(self):
        self.seed()
        code, out, _ = run(self.common + ["search", "points requirement", "--top-k", "3"])
        self.assertEqual(code, 0)
        self.assertIn("score", out)
        self.assertIn("Points requirement", out)
        self.assertIn("https://immi.homeaffairs.gov.au", out)
        first_line = out.splitlines()[0]
        self.assertTrue(first_line.startswith("1."), first_line)

    def test_search_without_matches_exits_1(self):
        self.seed()
        code, out, _ = run(self.common + ["search", "zzzqqq"])
        self.assertEqual(code, 1)
        self.assertIn("No matching passages", out)

    def test_ask_prints_citations(self):
        self.seed()
        code, out, _ = run(self.common + ["ask", "points requirement", "--extractive"])
        self.assertEqual(code, 0)
        self.assertIn("Citations:", out)
        self.assertIn("(mode: extractive)", out)

    def test_diff_writes_report_and_prints_path(self):
        self.seed()
        code, out, _ = run(self.common + ["diff"])
        self.assertEqual(code, 0)
        self.assertIn("+1 -1 ~2 >1", out)
        self.assertIn("report:", out)
        self.assertEqual(len(list(self.reports.glob("au_189-*.md"))), 1)

    def test_diff_without_snapshots_exits_1(self):
        code, out, _ = run(self.common + ["diff"])
        self.assertEqual(code, 1)
        self.assertIn("No sources stored", out)

    def test_fetch_with_missing_sources_file_exits_2(self):
        code, _, err = run(self.common + ["--sources", str(self.root / "nope.yaml"), "fetch"])
        self.assertEqual(code, 2)
        self.assertIn("sources file not found", err)

    def test_unknown_command_is_an_argparse_error(self):
        with self.assertRaises(SystemExit):
            run(["frobnicate"])


if __name__ == "__main__":
    unittest.main()
