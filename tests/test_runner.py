import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rag_im import cli
from rag_im.fetch import FetchSummary
from rag_im.runner import RunLocked, run_lock


class TestRunLock(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "data" / "run.lock"

    def tearDown(self):
        self.tmp.cleanup()

    def test_second_holder_is_rejected(self):
        with run_lock(self.path):
            with self.assertRaises(RunLocked):
                with run_lock(self.path):
                    pass

    def test_lock_is_released(self):
        with run_lock(self.path):
            pass
        with run_lock(self.path):
            pass
        self.assertTrue(self.path.exists())
        self.assertIn("pid=", self.path.read_text())


class TestRunCommand(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.sources = self.root / "sources.yaml"
        self.sources.write_text(
            "sources:\n"
            "  - id: au_189\n"
            "    country: AU\n"
            "    name: Skilled Independent 189\n"
            "    url: https://example.gov.au/189\n"
            "    selector: main#content\n"
        )
        self.common = [
            "--data-dir", str(self.root / "data"),
            "--db", str(self.root / "data" / "policy.sqlite3"),
            "--reports", str(self.root / "reports"),
            "--sources", str(self.sources),
        ]

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, argv, summaries):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch("rag_im.cli.fetch_all", return_value=summaries):
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = cli.main(self.common + argv)
        return code, out.getvalue(), err.getvalue()

    def test_run_exits_zero_and_writes_the_lock_file(self):
        code, out, _ = self.run_cli(
            ["run"], [FetchSummary("au_189", "new", "7 sections", snapshot_id=1, sections=7)]
        )
        self.assertEqual(code, 0)
        self.assertIn("1 ok, 0 failed", out)
        self.assertTrue((self.root / "data" / "run.lock").exists())

    def test_run_exits_one_on_source_error(self):
        code, out, _ = self.run_cli(
            ["run"], [FetchSummary("au_189", "error", "bot wall")]
        )
        self.assertEqual(code, 1)
        self.assertIn("0 ok, 1 failed", out)

    def test_run_exits_three_when_locked(self):
        lock = self.root / "data" / "run.lock"
        with run_lock(lock):
            code, _, err = self.run_cli(["run"], [])
        self.assertEqual(code, 3)
        self.assertIn("another rag-im run holds", err)

    def test_run_with_no_matching_sources_exits_one(self):
        code, out, err = self.run_cli(["run"], [])
        self.assertEqual(code, 1)
        self.assertIn("No matching sources", out)
        self.assertEqual(err, "")

    def test_run_with_bad_sources_file(self):
        self.sources.unlink()
        code, _, err = self.run_cli(["run"], [])
        self.assertEqual(code, 2)
        self.assertIn("sources file not found", err)


if __name__ == "__main__":
    unittest.main()
