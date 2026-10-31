"""Shared helpers for the unittest suite. Offline only: fixtures, no network."""

from __future__ import annotations

import hashlib
from pathlib import Path

from rag_im.extract import html_to_text
from rag_im.sections import split_text
from rag_im.store import Store

FIXTURES = Path(__file__).resolve().parent / "fixtures"

AU_URL = "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/skilled-independent-189"
AU_NAME = "Skilled Independent visa (subclass 189)"
NL_URL = "https://ind.nl/en/residence-permits/work/highly-skilled-migrant"
NL_NAME = "Highly Skilled Migrant residence permit"


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def add_fixture(
    store: Store,
    source_id: str,
    filename: str,
    fetched_at: str,
    *,
    country: str = "AU",
    name: str = AU_NAME,
    url: str = AU_URL,
    selector: str | None = "main#content",
):
    """Extract + split a fixture page and persist it as a snapshot."""
    store.upsert_source(source_id, country, name, url, selector)
    html = fixture_text(filename)
    text = html_to_text(html, selector=selector)
    sections = split_text(text)
    sha = hashlib.sha256(html.encode("utf-8")).hexdigest()
    snapshot_id, created = store.add_snapshot(
        source_id,
        fetched_at=fetched_at,
        url=url,
        content_sha256=sha,
        text=text,
        raw_path=None,
        sections=sections,
    )
    return snapshot_id, sections, text


def add_au_pair(store: Store, source_id: str = "au_189") -> tuple[int, int]:
    first, _, _ = add_fixture(store, source_id, "au_189_v1.html", "2026-07-01T09:00:00Z")
    second, _, _ = add_fixture(store, source_id, "au_189_v2.html", "2026-10-31T09:00:00Z")
    return first, second


def add_nl(store: Store, source_id: str = "nl_hsm") -> int:
    snapshot_id, _, _ = add_fixture(
        store,
        source_id,
        "nl_hsm_v1.html",
        "2026-09-15T12:00:00Z",
        country="NL",
        name=NL_NAME,
        url=NL_URL,
        selector="main",
    )
    return snapshot_id
