"""Paths, sources and endpoint settings.

Everything here can be overridden with environment variables so that tests and
CI never touch the real ``data/`` directory:

    RAG_IM_DATA       data directory (default: <repo>/data)
    RAG_IM_DB         sqlite file (default: <data>/policy.sqlite3)
    RAG_IM_SOURCES    sources.yaml (default: <repo>/sources.yaml)
    RAG_IM_REPORTS    report directory (default: <repo>/reports)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

USER_AGENT = (
    "rag-im/0.2 (immigration policy monitor; "
    "+https://github.com/louis/rag_im) python-requests"
)

CONNECT_TIMEOUT = float(os.environ.get("RAG_IM_CONNECT_TIMEOUT", "10"))
READ_TIMEOUT = float(os.environ.get("RAG_IM_READ_TIMEOUT", "30"))
MAX_RETRIES = int(os.environ.get("RAG_IM_RETRIES", "3"))
BACKOFF_BASE = float(os.environ.get("RAG_IM_BACKOFF", "1.5"))


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    raw_dir: Path
    reports_dir: Path
    db_path: Path
    sources_path: Path


def get_paths(data_dir: str | Path | None = None, db_path: str | Path | None = None) -> Paths:
    """Resolve all filesystem locations. Explicit args beat env vars."""
    data = Path(
        data_dir or os.environ.get("RAG_IM_DATA") or (REPO_ROOT / "data")
    ).expanduser().resolve()
    reports = Path(
        os.environ.get("RAG_IM_REPORTS") or (REPO_ROOT / "reports")
    ).expanduser().resolve()
    sources = Path(
        os.environ.get("RAG_IM_SOURCES") or (REPO_ROOT / "sources.yaml")
    ).expanduser().resolve()
    db = Path(db_path or os.environ.get("RAG_IM_DB") or (data / "policy.sqlite3")).expanduser().resolve()
    return Paths(
        data_dir=data,
        raw_dir=data / "raw",
        reports_dir=reports,
        db_path=db,
        sources_path=sources,
    )


def ensure_dirs(paths: Paths) -> None:
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    paths.db_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Source:
    id: str
    country: str
    name: str
    url: str
    selector: str | None = None
    notes: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Source":
        missing = [k for k in ("id", "country", "name", "url") if not raw.get(k)]
        if missing:
            raise ValueError(f"source entry missing required field(s): {', '.join(missing)}")
        known = {"id", "country", "name", "url", "selector", "notes"}
        return cls(
            id=str(raw["id"]),
            country=str(raw["country"]).upper(),
            name=str(raw["name"]),
            url=str(raw["url"]),
            selector=raw.get("selector") or None,
            notes=raw.get("notes") or None,
            extra={k: v for k, v in raw.items() if k not in known},
        )


def load_sources(path: str | Path | None = None) -> list[Source]:
    """Read sources.yaml. Raises a readable error on malformed files."""
    p = Path(path or get_paths().sources_path)
    if not p.exists():
        raise FileNotFoundError(f"sources file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    entries = data.get("sources") if isinstance(data, dict) else data
    if not entries:
        raise ValueError(f"no sources defined in {p}")
    sources: list[Source] = []
    seen: set[str] = set()
    for raw in entries:
        if not isinstance(raw, dict):
            raise ValueError(f"invalid source entry in {p}: {raw!r}")
        src = Source.from_dict(raw)
        if src.id in seen:
            raise ValueError(f"duplicate source id in {p}: {src.id}")
        seen.add(src.id)
        sources.append(src)
    return sources


@dataclass(frozen=True)
class LLMSettings:
    """Any OpenAI-compatible endpoint. No key means BM25-only, never an error."""

    api_key: str | None
    base_url: str
    chat_model: str
    embed_model: str
    timeout: float

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    @property
    def label(self) -> str:
        return f"{self.base_url.rstrip('/')} ({self.chat_model})" if self.enabled else "no endpoint configured"


def llm_settings() -> LLMSettings:
    key = os.environ.get("RAG_IM_API_KEY") or os.environ.get("OPENAI_API_KEY") or None
    base = (
        os.environ.get("RAG_IM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )
    return LLMSettings(
        api_key=key,
        base_url=base.rstrip("/"),
        chat_model=os.environ.get("RAG_IM_CHAT_MODEL", "gpt-4o-mini"),
        embed_model=os.environ.get("RAG_IM_EMBED_MODEL", "text-embedding-3-small"),
        timeout=float(os.environ.get("RAG_IM_LLM_TIMEOUT", "60")),
    )
