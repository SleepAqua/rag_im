import json
import tempfile
import unittest
from pathlib import Path

from rag_im import cli, evaluate
from rag_im.config import LLMSettings
from rag_im.evaluate import EvalQuestion
from rag_im.store import Store
from tests.support import add_au_pair, add_nl


def no_key() -> LLMSettings:
    return LLMSettings(None, "https://example.invalid/v1", "test", "test", 1)


class TestQuestionLoading(unittest.TestCase):
    def test_repo_question_set_loads(self):
        questions = evaluate.load_questions()
        self.assertGreaterEqual(len(questions), 15)
        self.assertEqual(len({q.id for q in questions}), len(questions))
        self.assertTrue(all(q.expected_source and q.expected_anchor for q in questions))

    def test_missing_file_is_readable(self):
        with self.assertRaises(FileNotFoundError):
            evaluate.load_questions("/nonexistent/questions.yaml")

    def test_malformed_question_reports_which_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q.yaml"
            path.write_text(
                "questions:\n"
                "  - id: ok\n"
                "    question: fine?\n"
                "    source: au_189\n"
                "    anchor: overview\n"
                "  - id: broken\n"
                "    question: missing source?\n"
                "    anchor: overview\n"
            )
            with self.assertRaises(ValueError) as ctx:
                evaluate.load_questions(path)
            self.assertIn("question #2", str(ctx.exception))
            self.assertIn("source", str(ctx.exception))

    def test_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q.yaml"
            entry = "  - id: same\n    question: q\n    source: au_189\n    anchor: overview\n"
            path.write_text("questions:\n" + entry + entry)
            with self.assertRaises(ValueError) as ctx:
                evaluate.load_questions(path)
            self.assertIn("duplicate question id", str(ctx.exception))


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "policy.sqlite3")
        add_au_pair(self.store)
        add_nl(self.store)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_fixture_corpus_meets_the_gate(self):
        report = evaluate.evaluate(
            self.store, evaluate.load_questions(), settings=no_key()
        )
        self.assertEqual(report.corpus_size, 11)
        self.assertEqual(report.mode, "bm25")
        self.assertEqual(report.recall_at(5), 1.0)
        self.assertGreaterEqual(report.recall_at(1), 0.9)
        self.assertGreaterEqual(report.mrr, 0.9)
        self.assertEqual(report.misses, [])
        self.assertEqual(report.missing_expected, [])

    def test_missing_expected_section_is_a_miss_with_diagnostics(self):
        question = EvalQuestion(
            id="impossible",
            question="How much is the visa application charge?",
            expected_source="au_189",
            expected_anchor="visa-cost",  # only in v1, not the latest snapshot
        )
        report = evaluate.evaluate(self.store, [question], settings=no_key())
        self.assertIsNone(report.results[0].rank)
        self.assertEqual(report.recall_at(5), 0.0)
        self.assertEqual(report.mrr, 0.0)
        self.assertEqual(len(report.misses), 1)
        self.assertEqual(report.missing_expected, ["au_189#visa-cost"])
        rendered = evaluate.render_report(report)
        self.assertIn("not retrieved at all", rendered)
        self.assertIn("Warning:", rendered)

    def test_json_output_is_machine_readable(self):
        report = evaluate.evaluate(
            self.store, evaluate.load_questions(), settings=no_key()
        )
        payload = json.loads(evaluate.to_json(report))
        self.assertEqual(payload["corpus_size"], 11)
        self.assertEqual(payload["questions"], len(report.results))
        self.assertIn("1", payload["recall"])
        self.assertEqual(payload["recall"]["5"], 1.0)

    def test_empty_store_is_zero_not_a_crash(self):
        empty = Store(Path(self.tmp.name) / "empty.sqlite3")
        try:
            report = evaluate.evaluate(empty, evaluate.load_questions(), settings=no_key())
            self.assertEqual(report.corpus_size, 0)
            self.assertEqual(report.recall_at(5), 0.0)
            self.assertTrue(report.missing_expected)
            self.assertLessEqual(len(report.missing_expected), len(report.results))
        finally:
            empty.close()


class TestEvalCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = self.root / "policy.sqlite3"
        self.common = [
            "--data-dir", str(self.root / "data"),
            "--db", str(self.db),
            "--reports", str(self.root / "reports"),
        ]
        with Store(self.db) as store:
            add_au_pair(store)
            add_nl(store)

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, argv):
        import contextlib
        import io

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(self.common + argv)
        return code, out.getvalue()

    def test_eval_reports_metrics(self):
        code, out = self.run_cli(["eval", "--min-recall", "0.8"])
        self.assertEqual(code, 0)
        self.assertIn("recall@1", out)
        self.assertIn("corpus=11", out)
        self.assertIn("No misses", out)

    def test_eval_fails_when_the_gate_is_not_met(self):
        code, _ = self.run_cli(["eval", "--min-recall", "1.01"])
        self.assertEqual(code, 1)

    def test_eval_missing_question_set_exits_2(self):
        code, _ = self.run_cli(["eval", "--questions", str(self.root / "nope.yaml")])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
