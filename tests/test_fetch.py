import hashlib
import tempfile
import unittest
from pathlib import Path

from rag_im.config import Paths, Source
from rag_im.fetch import FetchError, fetch_all, fetch_source, http_get
from rag_im.store import Store
from tests.support import AU_NAME, AU_URL, fixture_text


class FakeResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code


class FakeSession:
    """Returns queued outcomes; exceptions are raised, strings become bodies."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.timeouts = []

    def get(self, url, timeout=None, headers=None):
        self.calls += 1
        self.timeouts.append(timeout)
        idx = min(self.calls - 1, len(self.outcomes) - 1)
        outcome = self.outcomes[idx]
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, FakeResponse):
            return outcome
        return FakeResponse(outcome)


def make_paths(tmp: Path) -> Paths:
    return Paths(
        data_dir=tmp / "data",
        raw_dir=tmp / "data" / "raw",
        reports_dir=tmp / "reports",
        db_path=tmp / "data" / "policy.sqlite3",
        sources_path=tmp / "sources.yaml",
    )


def make_source() -> Source:
    return Source(id="au_189", country="AU", name=AU_NAME, url=AU_URL, selector="main#content")


class TestHttpGet(unittest.TestCase):
    def test_retries_transient_error_then_succeeds(self):
        html = fixture_text("au_189_v1.html")
        session = FakeSession([ConnectionError("connection reset"), html])
        sleeps: list[float] = []
        body = http_get("https://example.gov.au/189", session=session, sleep=sleeps.append)
        self.assertEqual(body, html)
        self.assertEqual(session.calls, 2)
        self.assertEqual(len(sleeps), 1)

    def test_timeout_budget_is_finite_and_message_is_clear(self):
        session = FakeSession([TimeoutError("read timed out")])
        sleeps: list[float] = []
        with self.assertRaises(FetchError) as ctx:
            http_get("https://example.gov.au/189", session=session, sleep=sleeps.append, retries=2)
        message = str(ctx.exception)
        self.assertIn("giving up on https://example.gov.au/189", message)
        self.assertIn("3 attempt(s)", message)
        self.assertIn("TimeoutError", message)
        self.assertIn("connect=", message)
        self.assertEqual(session.calls, 3)
        self.assertEqual(len(sleeps), 2)

    def test_404_is_not_retried(self):
        session = FakeSession([FakeResponse("not found", status_code=404)])
        with self.assertRaises(FetchError) as ctx:
            http_get("https://example.gov.au/189", session=session, sleep=lambda _: None)
        self.assertIn("HTTP 404", str(ctx.exception))
        self.assertEqual(session.calls, 1)

    def test_503_is_retried(self):
        session = FakeSession([FakeResponse("busy", 503), "<html>ok</html>"])
        body = http_get("https://example.gov.au/x", session=session, sleep=lambda _: None, retries=1)
        self.assertEqual(body, "<html>ok</html>")
        self.assertEqual(session.calls, 2)

    def test_empty_body_rejected(self):
        session = FakeSession(["   "])
        with self.assertRaises(FetchError):
            http_get("https://example.gov.au/x", session=session, sleep=lambda _: None)

    def test_timeout_is_passed_to_requests(self):
        session = FakeSession(["<html></html>"])
        http_get("https://example.gov.au/x", session=session, sleep=lambda _: None)
        connect, read = session.timeouts[0]
        self.assertGreater(connect, 0)
        self.assertGreater(read, 0)


class TestFetchSource(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.paths = make_paths(Path(self.tmp.name))
        self.html = fixture_text("au_189_v1.html")

    def tearDown(self):
        self.tmp.cleanup()

    def test_writes_raw_html_and_sha(self):
        result = fetch_source(make_source(), session=FakeSession([self.html]), paths=self.paths)
        self.assertFalse(result.unchanged)
        self.assertIsNotNone(result.path)
        self.assertTrue(result.path.exists())
        self.assertEqual(result.path.parent, self.paths.raw_dir / "au_189")
        self.assertEqual(result.path.read_text(encoding="utf-8"), self.html)
        self.assertEqual(
            result.content_sha256, hashlib.sha256(self.html.encode("utf-8")).hexdigest()
        )

    def test_unchanged_page_is_a_noop(self):
        first = fetch_source(make_source(), session=FakeSession([self.html]), paths=self.paths)
        second = fetch_source(
            make_source(),
            session=FakeSession([self.html]),
            paths=self.paths,
            previous_sha=first.content_sha256,
        )
        self.assertTrue(second.unchanged)
        self.assertIsNone(second.path)
        self.assertEqual(second.content_sha256, first.content_sha256)
        self.assertEqual(len(list((self.paths.raw_dir / "au_189").glob("*.html"))), 1)

    def test_changed_page_creates_second_file(self):
        first = fetch_source(make_source(), session=FakeSession([self.html]), paths=self.paths)
        changed = fixture_text("au_189_v2.html")
        second = fetch_source(
            make_source(),
            session=FakeSession([changed]),
            paths=self.paths,
            previous_sha=first.content_sha256,
        )
        self.assertFalse(second.unchanged)
        self.assertNotEqual(second.content_sha256, first.content_sha256)


class TestFetchAll(unittest.TestCase):
    def test_pipeline_stores_snapshot_and_reports_change(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        paths = make_paths(Path(tmp.name))
        source = make_source()
        with Store(paths.db_path) as store:
            first = fetch_all(
                [source],
                store,
                paths,
                session=FakeSession([fixture_text("au_189_v1.html")]),
            )
            self.assertEqual(first[0].status, "new")
            self.assertEqual(store.snapshot_count("au_189"), 1)

            second = fetch_all(
                [source],
                store,
                paths,
                session=FakeSession([fixture_text("au_189_v2.html")]),
            )
            self.assertEqual(second[0].status, "new")
            self.assertEqual(store.snapshot_count("au_189"), 2)
            self.assertIsNotNone(second[0].report)
            self.assertTrue(second[0].report.exists())

            third = fetch_all(
                [source],
                store,
                paths,
                session=FakeSession([fixture_text("au_189_v2.html")]),
            )
            self.assertEqual(third[0].status, "unchanged")
            self.assertEqual(store.snapshot_count("au_189"), 2)

    def test_js_shell_is_rejected_not_stored(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        paths = make_paths(Path(tmp.name))
        shell = (
            "<html><body><div id='app'></div>"
            "<noscript>Please enable JavaScript to view this page.</noscript>"
            "</body></html>"
        )
        with Store(paths.db_path) as store:
            summaries = fetch_all([make_source()], store, paths, session=FakeSession([shell]))
            self.assertEqual(summaries[0].status, "error")
            self.assertIn("extracted only", summaries[0].message)
            self.assertIn("JavaScript", summaries[0].message)
            self.assertEqual(store.snapshot_count("au_189"), 0)
            # raw HTML is kept for diagnosis
            self.assertEqual(len(list((paths.raw_dir / "au_189").glob("*.html"))), 1)

    def test_bot_wall_is_reported(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        paths = make_paths(Path(tmp.name))
        wall = "<html><body><script src='/_Incapsula_Resource'></script>Request unsuccessful</body></html>"
        with Store(paths.db_path) as store:
            summaries = fetch_all([make_source()], store, paths, session=FakeSession([wall]))
            self.assertEqual(summaries[0].status, "error")
            self.assertIn("Incapsula", summaries[0].message)

    def test_error_is_isolated_per_source(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        paths = make_paths(Path(tmp.name))
        source = make_source()
        with Store(paths.db_path) as store:
            summaries = fetch_all(
                [source],
                store,
                paths,
                session=FakeSession([TimeoutError("no route to host")]),
            )
            self.assertEqual(summaries[0].status, "error")
            self.assertIn("giving up", summaries[0].message)
            self.assertEqual(store.snapshot_count("au_189"), 0)


if __name__ == "__main__":
    unittest.main()
