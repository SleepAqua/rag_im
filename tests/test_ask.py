import tempfile
import unittest
from pathlib import Path

from rag_im.ask import answer_question, best_snippet, extractive_answer, llm_answer
from rag_im.config import LLMSettings
from rag_im.retrieval import Retriever, passages_from_store
from rag_im.store import Store
from tests.support import add_au_pair, add_nl


def no_key() -> LLMSettings:
    return LLMSettings(None, "https://example.invalid/v1", "test", "test", 1)


def with_key() -> LLMSettings:
    return LLMSettings("sk-test", "https://example.invalid/v1", "test-model", "test-embed", 1)


class FakeChatResponse:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class FakeChatSession:
    def __init__(self, content):
        self.content = content
        self.payload = None

    def post(self, url, json=None, headers=None, timeout=None):
        self.payload = json
        return FakeChatResponse(self.content)


class TestExtractive(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "policy.sqlite3")
        add_au_pair(self.store)
        add_nl(self.store)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_answer_has_passages_and_numbered_citations(self):
        answer = answer_question(
            "What is the points requirement?", self.store, top_k=3, settings=no_key()
        )
        self.assertEqual(answer.mode, "extractive")
        self.assertIn("[1]", answer.text)
        self.assertIn("Points requirement", answer.text)
        self.assertIn("https://immi.homeaffairs.gov.au", answer.text)
        self.assertEqual(answer.passages[0].passage.heading, "Points requirement")
        self.assertIn("AU", answer.citations[0])

    def test_answer_quotes_stored_text_not_invented_text(self):
        answer = answer_question("points requirement", self.store, top_k=1, settings=no_key())
        # the latest snapshot (v2) says 67; the answer must quote the store
        self.assertIn("67 points", answer.text)
        self.assertIn("immi.homeaffairs.gov.au", answer.text)

    def test_no_hits_message(self):
        answer = answer_question("zzzqqq unknown", self.store, top_k=3, settings=no_key())
        self.assertEqual(answer.mode, "none")
        self.assertIn("No stored passages matched", answer.text)
        self.assertEqual(answer.passages, [])

    def test_best_snippet_prefers_query_sentence(self):
        text = "Unrelated first sentence. The salary threshold is EUR 5,688 per month. Another sentence."
        snippet = best_snippet(text, "salary threshold")
        self.assertIn("EUR 5,688", snippet)

    def test_number_of_citations_matches_passages(self):
        answer = answer_question("visa", self.store, top_k=2, settings=no_key())
        self.assertEqual(len(answer.passages), 2)
        self.assertIn("[2]", answer.text)


class TestLLMPath(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "policy.sqlite3")
        add_au_pair(self.store)
        self.hits = Retriever(passages_from_store(self.store), settings=no_key()).search(
            "points requirement", top_k=3
        )

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_valid_citation_accepted(self):
        session = FakeChatSession("You need 67 points [1].")
        answer = llm_answer("points requirement", self.hits, with_key(), session=session)
        self.assertIsNotNone(answer)
        self.assertEqual(answer.mode, "llm")
        self.assertEqual(answer.model, "test-model")
        self.assertIn("[1]", answer.text)
        # passages must be in the prompt with their numbers
        self.assertIn("[1]", session.payload["messages"][1]["content"])

    def test_fabricated_citation_rejected(self):
        session = FakeChatSession("You need 99 points [7].")
        self.assertIsNone(llm_answer("points requirement", self.hits, with_key(), session=session))

    def test_no_key_never_calls_llm(self):
        session = FakeChatSession("should not be used [1]")
        answer = answer_question(
            "points requirement", self.store, top_k=2, settings=no_key(), session=session
        )
        self.assertEqual(answer.mode, "extractive")

    def test_endpoint_failure_falls_back(self):
        class Broken:
            def post(self, *a, **k):
                raise ConnectionError("endpoint down")

        answer = answer_question(
            "points requirement", self.store, top_k=2, settings=with_key(), session=Broken()
        )
        self.assertEqual(answer.mode, "extractive")

    def test_empty_hits_never_calls_llm(self):
        self.assertIsNone(llm_answer("anything", [], with_key()))


if __name__ == "__main__":
    unittest.main()
