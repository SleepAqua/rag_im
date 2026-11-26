"""Retrieval evaluation: score ``rag_im search`` against labelled questions.

This is a regression gate, not a benchmark. The fixture corpus is small (11
sections), so recall@5 is easy; recall@1 and MRR are the numbers that move when
retrieval regresses. Every question names the exact source and section anchor a
correct answer must come from, so a passage that merely mentions the words does
not count.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import config
from .config import LLMSettings
from .retrieval import Retriever, passages_from_store
from .store import Store

DEFAULT_QUESTIONS = config.REPO_ROOT / "eval" / "questions.yaml"


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    expected_source: str
    expected_anchor: str
    notes: str = ""


@dataclass
class QuestionResult:
    question: EvalQuestion
    rank: int | None
    top: list[tuple[str, str]] = field(default_factory=list)

    @property
    def hit_at_1(self) -> bool:
        return self.rank == 1

    @property
    def hit_at_3(self) -> bool:
        return self.rank is not None and self.rank <= 3

    @property
    def hit_at_5(self) -> bool:
        return self.rank is not None and self.rank <= 5

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.rank if self.rank else 0.0


@dataclass
class EvalReport:
    results: list[QuestionResult]
    top_k: int
    corpus_size: int
    mode: str
    missing_expected: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.results)

    def recall_at(self, k: int) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.rank is not None and r.rank <= k) / self.n

    @property
    def mrr(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.reciprocal_rank for r in self.results) / self.n

    @property
    def misses(self) -> list[QuestionResult]:
        return [r for r in self.results if r.rank is None or r.rank > self.top_k]

    def as_dict(self) -> dict:
        return {
            "top_k": self.top_k,
            "corpus_size": self.corpus_size,
            "mode": self.mode,
            "questions": self.n,
            "missing_expected": self.missing_expected,
            "recall": {str(k): round(self.recall_at(k), 4) for k in (1, 3, 5)},
            "mrr": round(self.mrr, 4),
            "results": [
                {
                    "id": r.question.id,
                    "rank": r.rank,
                    "reciprocal_rank": round(r.reciprocal_rank, 4),
                    "expected": f"{r.question.expected_source}#{r.question.expected_anchor}",
                    "top": [f"{s}#{a}" for s, a in r.top],
                }
                for r in self.results
            ],
        }


def default_questions_path() -> Path:
    override = os.environ.get("RAG_IM_EVAL_QUESTIONS")
    return Path(override).expanduser().resolve() if override else DEFAULT_QUESTIONS


def load_questions(path: str | Path | None = None) -> list[EvalQuestion]:
    """Read a YAML question set. Raises ValueError with the offending entry."""
    p = Path(path or default_questions_path())
    if not p.exists():
        raise FileNotFoundError(f"evaluation set not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or []
    entries = payload.get("questions") if isinstance(payload, dict) else payload
    if not entries:
        raise ValueError(f"no questions defined in {p}")

    questions: list[EvalQuestion] = []
    seen: set[str] = set()
    for idx, raw in enumerate(entries, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"question #{idx} in {p} is not a mapping: {raw!r}")
        missing = [key for key in ("id", "question", "source", "anchor") if not raw.get(key)]
        if missing:
            raise ValueError(
                f"question #{idx} in {p} missing required field(s): {', '.join(missing)}"
            )
        qid = str(raw["id"])
        if qid in seen:
            raise ValueError(f"duplicate question id in {p}: {qid}")
        seen.add(qid)
        questions.append(
            EvalQuestion(
                id=qid,
                question=str(raw["question"]),
                expected_source=str(raw["source"]),
                expected_anchor=str(raw["anchor"]),
                notes=str(raw.get("notes") or ""),
            )
        )
    return questions


def evaluate(
    store: Store,
    questions: list[EvalQuestion],
    *,
    top_k: int = 5,
    settings: LLMSettings | None = None,
) -> EvalReport:
    """Rank every question over the latest snapshot of every stored source."""
    passages = passages_from_store(store)
    retriever = Retriever(passages, settings=settings or config.llm_settings())
    available = {(p.source_id, p.anchor) for p in passages}
    missing_expected = list(
        dict.fromkeys(
            f"{q.expected_source}#{q.expected_anchor}"
            for q in questions
            if (q.expected_source, q.expected_anchor) not in available
        )
    )
    results: list[QuestionResult] = []
    for question in questions:
        # Rank over the whole corpus so a miss has a true rank for diagnostics,
        # not just "absent from the top 5".
        hits = retriever.search(question.question, top_k=max(len(passages), 1))
        rank = None
        for position, hit in enumerate(hits, start=1):
            p = hit.passage
            if p.source_id == question.expected_source and p.anchor == question.expected_anchor:
                rank = position
                break
        top = [(h.passage.source_id, h.passage.anchor) for h in hits[:top_k]]
        results.append(QuestionResult(question, rank, top))
    return EvalReport(
        results,
        top_k,
        len(passages),
        retriever.mode,
        missing_expected=missing_expected,
    )


def render_report(report: EvalReport) -> str:
    lines = [
        (
            f"Retrieval eval — {report.n} questions · "
            f"corpus={report.corpus_size} passages · "
            f"top_k={report.top_k} · mode={report.mode}"
        ),
        (
            f"recall@1 {report.recall_at(1):.3f}   "
            f"recall@3 {report.recall_at(3):.3f}   "
            f"recall@{report.top_k} {report.recall_at(report.top_k):.3f}   "
            f"MRR {report.mrr:.3f}"
        ),
    ]
    if report.missing_expected:
        lines += [
            "",
            (
                f"Warning: {len(report.missing_expected)} expected section(s) are "
                "absent from this corpus."
            ),
            "The question set targets a different dataset; the scores above are not meaningful.",
        ]
        for pair in report.missing_expected[:5]:
            lines.append(f"  missing {pair}")
        if len(report.missing_expected) > 5:
            lines.append(f"  ... and {len(report.missing_expected) - 5} more")

    if not report.misses:
        lines += ["", f"No misses in the top {report.top_k}."]
        return "\n".join(lines)

    lines += ["", f"Misses ({len(report.misses)}):"]
    for result in report.misses[:10]:
        q = result.question
        expected = f"{q.expected_source}#{q.expected_anchor}"
        if result.rank is None:
            lines.append(f"  {q.id}: {q.question!r}")
            lines.append(f"    expected {expected} — not retrieved at all")
        else:
            lines.append(f"  {q.id}: {q.question!r}")
            lines.append(f"    expected {expected} — found at rank {result.rank}")
        actual = ", ".join(f"{s}#{a}" for s, a in result.top[:3]) or "(nothing)"
        lines.append(f"    top 3: {actual}")
    if len(report.misses) > 10:
        lines.append(f"  ... and {len(report.misses) - 10} more")
    return "\n".join(lines)


def to_json(report: EvalReport) -> str:
    return json.dumps(report.as_dict(), indent=2, sort_keys=True)
