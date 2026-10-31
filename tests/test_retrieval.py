import tempfile
import unittest
from pathlib import Path

from rag_im.config import LLMSettings, Paths
from rag_im.retrieval import (
    BM25,
    Retriever,
    reciprocal_rank_fusion,
    passages_from_store,
    search,
    tokenise,
)
from rag_im.store import Store
from tests.support import add_au_pair, add_nl


def no_key_settings() -> LLMSettings:
    return LLMSettings(
        api_key=None,
        base_url="https://example.invalid/v1",
        chat_model="test",
        embed_model="test",
        timeout=1,
    )


class TestTokeniser(unittest.TestCase):
    def test_lowercases_and_strips_punctuation(self):
        self.assertEqual(tokenise("Points, Requirement! 65-point"), ["points", "requirement", "65", "point"])

    def test_empty(self):
        self.assertEqual(tokenise(""), [])


class TestBM25(unittest.TestCase):
    def test_ranks_relevant_document_first(self):
        docs = [
            "The visa application charge is AUD 4765 for the primary applicant",
            "You must score at least 65 points on the points test",
            "Processing times vary by occupation and application volume",
        ]
        scores = BM25(docs).scores("points requirement")
        self.assertGreater(scores[1], scores[0])
        self.assertGreater(scores[1], scores[2])

    def test_zero_for_unmatched_terms(self):
        scores = BM25(["alpha beta", "gamma delta"]).scores("zebra")
        self.assertEqual(scores, [0.0, 0.0])

    def test_empty_query_and_empty_corpus(self):
        self.assertEqual(BM25(["a b"]).scores(""), [0.0])
        self.assertEqual(BM25([]).scores("anything"), [])

    def test_rare_term_outweighs_common_term(self):
        docs = ["visa visa visa", "visa points"]
        scores = BM25(docs).scores("visa points")
        self.assertGreater(scores[1], scores[0])

    def test_rrf_prefers_document_ranked_by_both(self):
        fused = reciprocal_rank_fusion([[0, 1, 2], [1, 2, 0]])
        self.assertEqual(max(fused, key=fused.get), 1)


class TestSearchOverFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "policy.sqlite3")
        add_au_pair(self.store)
        add_nl(self.store)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_points_requirement_is_top_hit(self):
        hits = search("points requirement", self.store, top_k=3, settings=no_key_settings())
        self.assertTrue(hits)
        self.assertEqual(hits[0].passage.heading, "Points requirement")
        self.assertGreater(hits[0].score, 0)
        self.assertEqual(hits[0].passage.source_id, "au_189")

    def test_ranks_are_descending(self):
        hits = search("salary threshold highly skilled", self.store, top_k=5, settings=no_key_settings())
        scores = [h.score for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(hits[0].passage.heading, "Salary threshold")

    def test_no_key_means_bm25_mode(self):
        passages = passages_from_store(self.store)
        retriever = Retriever(passages, settings=no_key_settings())
        hits = retriever.search("English language evidence", top_k=3)
        self.assertEqual(retriever.mode, "bm25")
        self.assertTrue(all(h.bm25_score is not None for h in hits))
        self.assertTrue(all(h.dense_score is None for h in hits))

    def test_source_filter(self):
        hits = search(
            "salary threshold", self.store, top_k=5, source_ids=["nl_hsm"], settings=no_key_settings()
        )
        self.assertTrue(hits)
        self.assertTrue(all(h.passage.source_id == "nl_hsm" for h in hits))

    def test_unmatched_query_returns_nothing(self):
        self.assertEqual(
            search("gibberishzzz", self.store, top_k=3, settings=no_key_settings()), []
        )

    def test_passages_carry_citations(self):
        hits = search("points", self.store, top_k=1, settings=no_key_settings())
        self.assertIn("AU", hits[0].passage.citation)
        self.assertTrue(hits[0].passage.url.startswith("https://"))


if __name__ == "__main__":
    unittest.main()
