"""Grounded question answering over stored snapshots.

Two modes, same evidence:

* extractive (default, always available) -- the top passages, verbatim, each
  with its source and heading citation.
* LLM (only when an OpenAI-compatible endpoint is configured) -- the passages
  are numbered and the model is told to cite them as ``[n]``. If it cites a
  number that was not supplied, the answer is rejected and we fall back to the
  extractive response rather than printing a fabricated citation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config
from .config import LLMSettings
from .retrieval import Passage, Retriever, ScoredPassage, passages_from_store, tokenise
from .store import Store

CITATION_RE = re.compile(r"\[(\d{1,2})\]")

SYSTEM_PROMPT = (
    "You answer questions about immigration policy using ONLY the numbered "
    "passages supplied by the user. Cite the passages you use as [1], [2], ... "
    "matching their numbers. If the passages do not answer the question, say so "
    "plainly. Never cite a passage number that was not supplied. Never state a "
    "policy detail that is absent from the passages."
)


@dataclass
class Answer:
    question: str
    text: str
    passages: list[ScoredPassage] = field(default_factory=list)
    mode: str = "extractive"
    model: str | None = None

    @property
    def citations(self) -> list[str]:
        return [p.passage.citation for p in self.passages]


def _clip(text: str, limit: int = 700) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit].rstrip() + " …"


def best_snippet(text: str, question: str, limit: int = 700) -> str:
    """Pick the sentences in a passage that best cover the query terms."""
    terms = set(tokenise(question))
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    scored = []
    for idx, sentence in enumerate(sentences):
        overlap = len(terms & set(tokenise(sentence)))
        scored.append((overlap, -idx, sentence))
    scored.sort(reverse=True)
    chosen = sorted(scored[:3], key=lambda item: -item[1])
    snippet = " ".join(s for _, _, s in chosen if s)
    if not snippet:
        snippet = " ".join(text.split())
    return _clip(snippet, limit)


def build_context(hits: list[ScoredPassage]) -> str:
    chunks = []
    for idx, hit in enumerate(hits, start=1):
        p = hit.passage
        chunks.append(
            f"[{idx}] {p.country} · {p.source_name} — \"{p.heading}\"\n"
            f"URL: {p.url}\n"
            f"{_clip(p.text, 1200)}"
        )
    return "\n\n".join(chunks)


def extractive_answer(question: str, hits: list[ScoredPassage], top_k: int = 5) -> Answer:
    if not hits:
        return Answer(
            question=question,
            text=(
                "No stored passages matched this question. Run `python -m rag_im fetch` "
                "to snapshot the sources, or rephrase the query."
            ),
            mode="none",
        )
    lines = [
        "Extractive answer (no LLM endpoint configured) — the most relevant passages, verbatim:",
        "",
    ]
    for idx, hit in enumerate(hits[:top_k], start=1):
        p = hit.passage
        lines.append(f"[{idx}] {p.country} · {p.source_name} — \"{p.heading}\"")
        lines.append(best_snippet(p.text, question))
        lines.append(f"    source: {p.url}")
        lines.append("")
    return Answer(question=question, text="\n".join(lines).rstrip(), passages=hits[:top_k], mode="extractive")


def llm_answer(
    question: str,
    hits: list[ScoredPassage],
    settings: LLMSettings,
    *,
    session=None,
) -> Answer | None:
    """Chat-completion answer, or ``None`` when the endpoint is unusable."""
    if not settings.enabled or not hits:
        return None
    import requests

    payload = {
        "model": settings.chat_model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {question}\n\nPassages:\n\n{build_context(hits)}",
            },
        ],
    }
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
    }
    try:
        if session is not None:
            response = session.post(
                f"{settings.base_url}/chat/completions", json=payload, headers=headers,
                timeout=settings.timeout,
            )
        else:
            response = requests.post(
                f"{settings.base_url}/chat/completions", json=payload, headers=headers,
                timeout=settings.timeout,
            )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None

    cited = [int(n) for n in CITATION_RE.findall(text)]
    if any(n < 1 or n > len(hits) for n in cited):
        return None  # fabricated citation -> refuse, caller falls back
    return Answer(
        question=question,
        text=text,
        passages=hits[: len(hits)],
        mode="llm",
        model=settings.chat_model,
    )


def answer_question(
    question: str,
    store: Store,
    *,
    top_k: int = 5,
    settings: LLMSettings | None = None,
    session=None,
    use_llm: bool = True,
    use_cache: bool = True,
) -> Answer:
    settings = settings or config.llm_settings()
    passages = passages_from_store(store)
    hits = Retriever(passages, settings=settings, store=store, use_cache=use_cache).search(
        question, top_k=top_k
    )
    if use_llm and settings.enabled:
        llm = llm_answer(question, hits, settings, session=session)
        if llm is not None:
            return llm
    return extractive_answer(question, hits, top_k=top_k)
