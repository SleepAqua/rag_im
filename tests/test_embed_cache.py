import hashlib
import tempfile
import unittest
from pathlib import Path

from rag_im.config import LLMSettings
from rag_im.retrieval import (
    Retriever,
    embed_passages,
    passage_text,
    passages_from_store,
    search,
)
from rag_im.store import Store
from tests.support import add_au_pair, add_nl


def enabled() -> LLMSettings:
    return LLMSettings("sk-test", "https://example.invalid/v1", "chat", "test-embed", 1)


class FakeEmbedder:
    """Deterministic vectors plus a record of every batch it was asked for."""

    def __init__(self, fail_on_single: bool = False, fail_all: bool = False):
        self.batches: list[list[str]] = []
        self.fail_on_single = fail_on_single
        self.fail_all = fail_all

    def __call__(self, texts, settings):
        self.batches.append(list(texts))
        if self.fail_all:
            return None
        if self.fail_on_single and len(texts) == 1:
            return None
        return [
            [1.0, float(len(text) % 13) + 1.0, float(sum(ord(c) for c in text) % 17) + 1.0]
            for text in texts
        ]


class TestEmbeddingCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "policy.sqlite3")
        add_au_pair(self.store)
        add_nl(self.store)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_documents_are_embedded_once_across_queries(self):
        fake = FakeEmbedder()
        first = search("salary threshold", self.store, top_k=3, settings=enabled(), embedder=fake)
        self.assertTrue(first)
        self.assertEqual([len(batch) for batch in fake.batches], [11, 1])
        self.assertEqual(self.store.embedding_count(), 11)

        second = search("points requirement", self.store, top_k=3, settings=enabled(), embedder=fake)
        self.assertTrue(second)
        # Only the new query is embedded; the 11 document vectors came from SQLite.
        self.assertEqual([len(batch) for batch in fake.batches], [11, 1, 1])
        self.assertEqual(self.store.embedding_count(), 11)

    def test_cache_can_be_bypassed(self):
        fake = FakeEmbedder()
        search("salary", self.store, top_k=2, settings=enabled(), embedder=fake)
        search("salary", self.store, top_k=2, settings=enabled(), embedder=fake, use_cache=False)
        self.assertEqual([len(batch) for batch in fake.batches], [11, 1, 11, 1])

    def test_vectors_round_trip_through_sqlite(self):
        passages = passages_from_store(self.store)[:2]
        vectors = embed_passages(passages, enabled(), self.store, embedder=FakeEmbedder())
        self.assertIsNotNone(vectors)
        keys = [
            hashlib.sha256(passage_text(p).encode("utf-8")).hexdigest()
            for p in passages
        ]
        cached = self.store.get_embeddings("test-embed", keys)
        self.assertEqual(set(cached), set(keys))
        for expected, key in zip(vectors, keys):
            self.assertEqual(len(cached[key]), len(expected))
            for a, b in zip(expected, cached[key]):
                self.assertAlmostEqual(a, b, places=5)
        self.assertEqual(self.store.embedding_count(), 2)

    def test_embedding_failure_degrades_to_bm25(self):
        fake = FakeEmbedder(fail_all=True)
        retriever = Retriever(
            passages_from_store(self.store), settings=enabled(), store=self.store, embedder=fake
        )
        hits = retriever.search("points requirement", top_k=3)
        self.assertEqual(retriever.mode, "bm25")
        self.assertEqual(hits[0].passage.heading, "Points requirement")
        self.assertEqual(self.store.embedding_count(), 0)

    def test_query_embedding_failure_also_degrades(self):
        fake = FakeEmbedder(fail_on_single=True)
        retriever = Retriever(
            passages_from_store(self.store), settings=enabled(), store=self.store, embedder=fake
        )
        hits = retriever.search("points requirement", top_k=3)
        self.assertEqual(retriever.mode, "bm25")
        self.assertTrue(hits)
        # Document vectors were still cached even though the query call failed.
        self.assertEqual(self.store.embedding_count(), 11)

    def test_provider_without_embeddings_skips_embedder(self):
        local = LLMSettings(
            None,
            "http://127.0.0.1:8000/v1",
            "qwen3.5-2b",
            "qwen3.5-2b",
            1,
            provider="local",
            embeddings=False,
        )
        fake = FakeEmbedder()
        retriever = Retriever(
            passages_from_store(self.store), settings=local, store=self.store, embedder=fake
        )
        hits = retriever.search("points requirement", top_k=3)
        self.assertEqual(retriever.mode, "bm25")
        self.assertTrue(hits)
        self.assertEqual(fake.batches, [])
        self.assertEqual(self.store.embedding_count(), 0)

    def test_clear_embeddings(self):
        search("salary", self.store, top_k=1, settings=enabled(), embedder=FakeEmbedder())
        self.assertEqual(self.store.embedding_count(), 11)
        self.assertEqual(self.store.clear_embeddings(), 11)
        self.assertEqual(self.store.embedding_count(), 0)


if __name__ == "__main__":
    unittest.main()
