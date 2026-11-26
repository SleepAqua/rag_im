import contextlib
import io
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from rag_im import cli
from rag_im.config import Paths, Source
from rag_im.fetch import fetch_all
from rag_im.store import Store
from tests.support import fixture_text


class FakeResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code


class FakeSession:
    def __init__(self, outcome):
        self.outcome = outcome

    def get(self, url, timeout=None, headers=None):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return FakeResponse(self.outcome)


def make_paths(tmp: Path) -> Paths:
    return Paths(
        data_dir=tmp / "data",
        raw_dir=tmp / "data" / "raw",
        reports_dir=tmp / "reports",
        db_path=tmp / "data" / "policy.sqlite3",
        sources_path=tmp / "sources.yaml",
    )


def make_source() -> Source:
    return Source(
        id="au_189",
        country="AU",
        name="Skilled Independent 189",
        url="https://example.gov.au/189",
        selector="main#content",
    )


class TestFetchRunMetrics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.paths = make_paths(self.root)
        self.paths.data_dir.mkdir(parents=True)
        self.store = Store(self.paths.db_path)
        self.store.upsert_source(
            "au_189", "AU", "Skilled Independent 189", "https://example.gov.au/189"
        )
        self.html = fixture_text("au_189_v1.html")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def run_fetch(self, outcome, run_id="run-1"):
        return fetch_all(
            [make_source()],
            self.store,
            self.paths,
            session=FakeSession(outcome),
            sleep=lambda _: None,
            run_id=run_id,
            now=datetime(2026, 7, 1, tzinfo=timezone.utc),
        )

    def test_successful_run_is_recorded_with_bytes_and_latency(self):
        summaries = self.run_fetch(self.html)
        self.assertEqual(summaries[0].status, "new")
        self.assertGreater(summaries[0].bytes, 0)
        self.assertGreaterEqual(summaries[0].latency_ms, 0)
        self.assertEqual(self.store.stats()["fetch_runs"], 1)

        stats = self.store.fetch_run_stats()
        self.assertEqual(stats["runs"], 1)
        self.assertEqual(stats["ok"], 1)
        self.assertEqual(stats["errors"], 0)
        self.assertEqual(stats["sources"][0]["source_id"], "au_189")
        self.assertEqual(stats["sources"][0]["last_status"], "new")

    def test_unchanged_run_still_records_bytes(self):
        self.run_fetch(self.html, run_id="first")
        summaries = self.run_fetch(self.html, run_id="second")
        self.assertEqual(summaries[0].status, "unchanged")
        self.assertGreater(summaries[0].bytes, 0)
        stats = self.store.fetch_run_stats()
        self.assertEqual(stats["runs"], 2)
        self.assertEqual(stats["sources"][0]["runs"], 2)
        self.assertEqual(stats["sources"][0]["ok"], 2)

    def test_failed_run_records_the_error(self):
        summaries = self.run_fetch(ConnectionError("connection reset"))
        self.assertEqual(summaries[0].status, "error")
        stats = self.store.fetch_run_stats()
        self.assertEqual(stats["errors"], 1)
        self.assertEqual(stats["ok"], 0)
        self.assertIn("giving up", stats["sources"][0]["last_error"])
        self.assertEqual(self.store.snapshot_count(), 0)

    def test_percentiles(self):
        for index, latency in enumerate((10, 20, 30)):
            self.store.record_fetch_run(
                run_id=f"r{index}",
                source_id="au_189",
                started_at="2026-07-01T00:00:00Z",
                finished_at=f"2026-07-01T00:00:0{index}Z",
                status="unchanged",
                latency_ms=latency,
            )
        stats = self.store.fetch_run_stats()
        self.assertEqual(stats["p50_ms"], 20)
        self.assertEqual(stats["p95_ms"], 29)

    def test_status_metrics_prints_history(self):
        self.run_fetch(self.html)
        self.store.close()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(
                [
                    "--data-dir", str(self.paths.data_dir),
                    "--db", str(self.paths.db_path),
                    "--reports", str(self.paths.reports_dir),
                    "status", "--metrics",
                ]
            )
        self.assertEqual(code, 0)
        self.assertIn("Fetch runs: 1 total", out.getvalue())
        self.assertIn("au_189", out.getvalue())
        # Reopen for tearDown's close().
        self.store = Store(self.paths.db_path)


if __name__ == "__main__":
    unittest.main()
