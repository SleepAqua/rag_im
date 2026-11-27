"""Paths, sources and endpoint settings.

Everything here can be overridden with environment variables so that tests and
CI never touch the real ``data/`` directory:

    RAG_IM_DATA       data directory (default: <repo>/data)
    RAG_IM_DB         sqlite file (default: <data>/policy.sqlite3)
    RAG_IM_SOURCES    sources.yaml (default: <repo>/sources.yaml)
    RAG_IM_REPORTS    report directory (default: <repo>/reports)
    RAG_IM_PROVIDER   auto|local|hermes|opencode|cloud|none (default: auto)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

USER_AGENT = (
    "rag-im/0.4 (immigration policy monitor; "
    "+https://github.com/SleepAqua/rag_im) python-requests"
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


PROVIDERS = ("auto", "local", "hermes", "opencode", "cloud", "none")


@dataclass(frozen=True)
class LLMSettings:
    """An OpenAI-compatible endpoint plus its profile.

    ``provider`` decides whether a key is required: local llama.cpp, the Hermes
    proxy and OpenCode Zen all accept requests without one. ``embeddings`` is
    separate because a chat server is not necessarily an embedding server.
    """

    api_key: str | None
    base_url: str
    chat_model: str
    embed_model: str
    timeout: float
    provider: str = "auto"
    embeddings: bool = True

    @property
    def enabled(self) -> bool:
        if self.provider == "none":
            return False
        if self.provider in {"local", "hermes", "opencode"}:
            return True
        return bool(self.api_key)

    @property
    def dense(self) -> bool:
        return self.enabled and self.embeddings

    @property
    def label(self) -> str:
        if not self.enabled:
            return "no endpoint configured"
        return f"{self.provider} · {self.chat_model} @ {self.base_url.rstrip('/')}"


def _env(name: str, *fallbacks: str) -> str | None:
    for key in (name, *fallbacks):
        value = os.environ.get(key)
        if value is not None and value.strip():
            return value.strip()
    return None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def llm_settings(provider: str | None = None) -> LLMSettings:
    """Resolve a provider profile from an explicit argument or the environment.

    Profiles:
      auto      cloud if an API key is present, otherwise BM25-only (default)
      local     llama.cpp / Ollama / LM Studio, default Qwen on :8000
      hermes    ``hermes proxy start`` OpenAI-compatible proxy on :8645
      opencode  OpenCode Zen OpenAI-compatible endpoint
      cloud     any OpenAI-compatible endpoint that needs a key
      none      disable the model entirely
    """
    name = (provider or os.environ.get("RAG_IM_PROVIDER") or "auto").strip().lower()
    if name == "openai":
        name = "cloud"
    if name not in PROVIDERS:
        raise ValueError(
            f"unknown provider {name!r}; choose one of: {', '.join(PROVIDERS)}"
        )
    timeout = float(os.environ.get("RAG_IM_LLM_TIMEOUT", "60"))

    if name == "none":
        return LLMSettings(None, "", "", "", timeout, provider="none", embeddings=False)

    if name == "local":
        base = _env("RAG_IM_LOCAL_BASE_URL", "RAG_IM_BASE_URL") or "http://127.0.0.1:8000/v1"
        chat = _env("RAG_IM_LOCAL_MODEL", "RAG_IM_CHAT_MODEL") or "qwen3.5-2b"
        return LLMSettings(
            api_key=_env("RAG_IM_LOCAL_API_KEY"),
            base_url=base.rstrip("/"),
            chat_model=chat,
            embed_model=_env("RAG_IM_LOCAL_EMBED_MODEL", "RAG_IM_EMBED_MODEL") or chat,
            timeout=timeout,
            provider="local",
            embeddings=_env_bool("RAG_IM_LOCAL_EMBEDDINGS", False),
        )

    if name == "hermes":
        return LLMSettings(
            # The proxy accepts any bearer token and attaches the real credential.
            api_key=_env("RAG_IM_HERMES_API_KEY") or "local",
            base_url=(_env("RAG_IM_HERMES_BASE_URL") or "http://127.0.0.1:8645/v1").rstrip("/"),
            chat_model=_env("RAG_IM_HERMES_MODEL") or "upstage/solar-mini4:free",
            embed_model=_env("RAG_IM_HERMES_EMBED_MODEL") or "qwen/qwen3-embedding-8b",
            timeout=timeout,
            provider="hermes",
            embeddings=_env_bool("RAG_IM_HERMES_EMBEDDINGS", False),
        )

    if name == "opencode":
        return LLMSettings(
            api_key=_env("RAG_IM_OPENCODE_API_KEY"),
            base_url=(_env("RAG_IM_OPENCODE_BASE_URL") or "https://opencode.ai/zen/v1").rstrip("/"),
            chat_model=_env("RAG_IM_OPENCODE_MODEL") or "deepseek-v4.1-flash-free",
            embed_model=_env("RAG_IM_OPENCODE_EMBED_MODEL") or "text-embedding-3-small",
            timeout=timeout,
            provider="opencode",
            embeddings=_env_bool("RAG_IM_OPENCODE_EMBEDDINGS", False),
        )

    # cloud / auto
    key = _env("RAG_IM_API_KEY", "OPENAI_API_KEY")
    base = _env("RAG_IM_BASE_URL", "OPENAI_BASE_URL") or "https://api.openai.com/v1"
    chat = _env("RAG_IM_CHAT_MODEL") or "gpt-4o-mini"
    embed = _env("RAG_IM_EMBED_MODEL") or "text-embedding-3-small"
    if name == "auto" and not key:
        return LLMSettings(
            None, base.rstrip("/"), chat, embed, timeout,
            provider="none", embeddings=False,
        )
    return LLMSettings(
        key,
        base.rstrip("/"),
        chat,
        embed,
        timeout,
        provider="cloud",
        embeddings=_env_bool("RAG_IM_EMBEDDINGS", True),
    )
