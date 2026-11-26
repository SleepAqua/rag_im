"""Retrieval: BM25 always, dense embeddings optionally, RRF when both exist.

BM25 is pure Python so search works offline with no key and no model download.
A key only enables an extra ranking that is fused with reciprocal-rank fusion;
losing it changes quality, never correctness.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from . import config
from .config import LLMSettings
from .store import Store

TOKEN_RE = re.compile(r"[a-z0-9]+")
RRF_K = 60


def tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, keep numbers (points thresholds matter)."""
    return TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class Passage:
    source_id: str
    source_name: str
    country: str
    url: str
    heading: str
    text: str
    snapshot_id: int
    fetched_at: str
    anchor: str = ""

    @property
    def citation(self) -> str:
        return f"{self.country} · {self.source_name} — “{self.heading}”"


@dataclass
class ScoredPassage:
    passage: Passage
    score: float
    rank: int = 0
    bm25_score: float | None = None
    dense_score: float | None = None


class BM25:
    """Okapi BM25 with the standard k1=1.5, b=0.75 defaults."""

    def __init__(self, documents: Iterable[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = [tokenise(doc) for doc in documents]
        self.n_docs = len(self.docs)
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.term_freqs = [Counter(d) for d in self.docs]
        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(set(doc))
        self.idf = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def scores(self, query: str) -> list[float]:
        terms = Counter(tokenise(query))
        if not terms or not self.n_docs:
            return [0.0] * self.n_docs
        k1, b = self.k1, self.b
        out: list[float] = []
        for idx, tf in enumerate(self.term_freqs):
            length_norm = 1 - b + b * (self.doc_len[idx] / self.avgdl if self.avgdl else 0)
            total = 0.0
            for term, qcount in terms.items():
                freq = tf.get(term)
                if not freq:
                    continue
                denom = freq + k1 * length_norm
                total += self.idf.get(term, 0.0) * (freq * (k1 + 1) / denom) * qcount
            out.append(total)
        return out


# --------------------------------------------------------------------- passages
def passages_from_store(store: Store, source_ids: list[str] | None = None) -> list[Passage]:
    passages: list[Passage] = []
    for source, snapshot, sections in store.latest_sections_by_source():
        if source_ids and source["id"] not in source_ids:
            continue
        for sec in sections:
            body = sec.body.strip()
            if len(body) < 20:
                continue
            passages.append(
                Passage(
                    source_id=source["id"],
                    source_name=source["name"],
                    country=source["country"],
                    url=source["url"],
                    heading=sec.heading,
                    text=body,
                    snapshot_id=snapshot.id,
                    fetched_at=snapshot.fetched_at,
                    anchor=sec.anchor,
                )
            )
    return passages


def passage_text(passage: Passage) -> str:
    return f"{passage.heading}\n{passage.text}"


# ------------------------------------------------------------------------ dense
EMBED_BATCH = 128


def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _request_embeddings(
    texts: list[str], settings: LLMSettings
) -> list[list[float]] | None:
    """POST to an OpenAI-compatible /embeddings endpoint. ``None`` on failure."""
    import requests

    try:
        response = requests.post(
            f"{settings.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": settings.embed_model, "input": texts},
            timeout=settings.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or []
        if len(data) != len(texts):
            return None
        # The API may return items out of order; index is authoritative.
        items = sorted(data, key=lambda item: item.get("index", 0))
        vectors: list[list[float]] = []
        for item in items:
            vector = [float(x) for x in item["embedding"]]
            if not vector:
                return None
            vectors.append(vector)
        return vectors
    except Exception:
        return None


def embed_texts(
    texts: list[str],
    settings: LLMSettings | None = None,
    *,
    embedder: Any = None,
) -> list[list[float]] | None:
    """Embed arbitrary texts. Any failure degrades to BM25; it is never fatal."""
    settings = settings or config.llm_settings()
    if not settings.enabled or not texts:
        return None
    request = embedder or _request_embeddings
    return request(list(texts), settings)


def embed_passages(
    passages: list[Passage],
    settings: LLMSettings | None = None,
    store: Store | None = None,
    *,
    use_cache: bool = True,
    embedder: Any = None,
) -> list[list[float]] | None:
    """Embed passages, reusing vectors cached by (model, text hash).

    The cache is the difference between paying for the same 11 documents on
    every query and paying once; a query still costs one embedding call.
    """
    settings = settings or config.llm_settings()
    if not settings.enabled or not passages:
        return None
    request = embedder or _request_embeddings
    model = settings.embed_model
    texts = [passage_text(p) for p in passages]
    keys = [_text_key(t) for t in texts]
    known = (
        store.get_embeddings(model, keys)
        if store is not None and use_cache
        else {}
    )

    missing: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key, text in zip(keys, texts):
        if key in known or key in seen:
            continue
        seen.add(key)
        missing.append((key, text))

    for start in range(0, len(missing), EMBED_BATCH):
        batch = missing[start : start + EMBED_BATCH]
        computed = request([text for _, text in batch], settings)
        if computed is None or len(computed) != len(batch):
            return None
        fresh: dict[str, list[float]] = {}
        for (key, _), vector in zip(batch, computed):
            if not vector:
                return None
            known[key] = vector
            fresh[key] = vector
        if store is not None and use_cache:
            store.put_embeddings(model, fresh)

    try:
        return [known[key] for key in keys]
    except KeyError:
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def dense_scores(
    query: str,
    passages: list[Passage],
    settings: LLMSettings | None = None,
    store: Store | None = None,
    *,
    use_cache: bool = True,
    embedder: Any = None,
) -> list[float] | None:
    vectors = embed_passages(
        passages, settings, store, use_cache=use_cache, embedder=embedder
    )
    if vectors is None:
        return None
    query_vectors = embed_texts([query], settings, embedder=embedder)
    if not query_vectors:
        return None
    query_vector = query_vectors[0]
    return [_cosine(query_vector, vector) for vector in vectors]


def reciprocal_rank_fusion(
    rankings: list[list[int]], weights: list[float] | None = None, k: int = RRF_K
) -> dict[int, float]:
    """Fuse ranked id lists. Each input must already be best-first."""
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + weight / (k + rank)
    return fused


# -------------------------------------------------------------------- retriever
@dataclass
class Retriever:
    passages: list[Passage]
    settings: LLMSettings = field(default_factory=config.llm_settings)
    mode: str = "bm25"
    store: Store | None = None
    use_cache: bool = True
    embedder: Any = None
    _bm25: BM25 | None = None

    def __post_init__(self) -> None:
        self._bm25 = BM25([passage_text(p) for p in self.passages])

    def search(self, query: str, top_k: int = 5) -> list[ScoredPassage]:
        if not self.passages:
            return []
        assert self._bm25 is not None
        bm_scores = self._bm25.scores(query)
        bm_order = sorted(range(len(bm_scores)), key=lambda i: bm_scores[i], reverse=True)

        dense = dense_scores(
            query,
            self.passages,
            self.settings,
            self.store,
            use_cache=self.use_cache,
            embedder=self.embedder,
        )
        if dense is None:
            self.mode = "bm25"
            ranked = [
                ScoredPassage(self.passages[i], float(bm_scores[i]), rank=r, bm25_score=bm_scores[i])
                for r, i in enumerate(bm_order, start=1)
                if bm_scores[i] > 0
            ]
            return ranked[:top_k]

        self.mode = "hybrid (bm25 + dense, RRF)"
        dense_order = sorted(range(len(dense)), key=lambda i: dense[i], reverse=True)
        fused = reciprocal_rank_fusion([bm_order, dense_order])
        ordered = sorted(fused, key=lambda i: fused[i], reverse=True)
        out = []
        for rank, idx in enumerate(ordered, start=1):
            if bm_scores[idx] <= 0 and dense[idx] <= 0:
                continue
            out.append(
                ScoredPassage(
                    self.passages[idx],
                    fused[idx],
                    rank=rank,
                    bm25_score=bm_scores[idx],
                    dense_score=dense[idx],
                )
            )
        return out[:top_k]


def search(
    query: str,
    store: Store,
    *,
    top_k: int = 5,
    source_ids: list[str] | None = None,
    settings: LLMSettings | None = None,
    use_cache: bool = True,
    embedder: Any = None,
) -> list[ScoredPassage]:
    passages = passages_from_store(store, source_ids)
    return Retriever(
        passages,
        settings=settings or config.llm_settings(),
        store=store,
        use_cache=use_cache,
        embedder=embedder,
    ).search(query, top_k)
