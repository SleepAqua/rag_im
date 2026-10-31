"""HTTP fetching with retries, timeouts and change-aware raw storage.

Every fetch is bounded: requests are given an explicit (connect, read) timeout
and a finite retry budget, so a black-holed network raises ``FetchError`` with
a readable message instead of hanging.

``fetch_source`` accepts an injected session and sleep function so the whole
path is testable offline against fixture HTML.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import config
from .config import Paths, Source
from .extract import html_to_text
from .sections import split_text
from .store import Store

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
MIN_TEXT_CHARS = 200


class FetchError(RuntimeError):
    """Raised when a page cannot be retrieved. Message is user-facing."""


@dataclass(frozen=True)
class FetchResult:
    source_id: str
    url: str
    fetched_at: str
    content_sha256: str
    path: Path | None
    unchanged: bool


def _default_session() -> Any:
    import requests

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": config.USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en;q=0.9",
        }
    )
    return session


def _utc_stamp(now: datetime | None = None) -> tuple[datetime, str]:
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment, moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def http_get(
    url: str,
    *,
    session: Any | None = None,
    sleep: Callable[[float], None] = time.sleep,
    retries: int | None = None,
) -> str:
    """GET a URL, retrying transient failures with exponential backoff."""
    sess = session or _default_session()
    budget = config.MAX_RETRIES if retries is None else retries
    timeout = (config.CONNECT_TIMEOUT, config.READ_TIMEOUT)
    last_error = "unknown error"

    for attempt in range(budget + 1):
        try:
            response = sess.get(url, timeout=timeout, headers={"User-Agent": config.USER_AGENT})
            status = int(getattr(response, "status_code", 200))
            if status in RETRYABLE_STATUS:
                last_error = f"HTTP {status} from {url}"
                raise _Transient(last_error)
            if status >= 400:
                raise FetchError(f"HTTP {status} from {url} (not retryable)")
            body = response.text
            if not body or not body.strip():
                raise FetchError(f"empty response body from {url}")
            return body
        except FetchError:
            raise
        except _Transient as exc:
            last_error = str(exc)
        except Exception as exc:  # requests.Timeout, ConnectionError, DNS, ...
            name = type(exc).__name__
            last_error = f"{name} while fetching {url}: {exc}".strip()

        if attempt < budget:
            sleep(config.BACKOFF_BASE ** attempt)

    raise FetchError(
        f"giving up on {url} after {budget + 1} attempt(s); last error: {last_error}. "
        f"Timeouts were connect={config.CONNECT_TIMEOUT}s read={config.READ_TIMEOUT}s."
    )


class _Transient(Exception):
    pass


def _diagnose(raw_html: str) -> str:
    """Explain why a 200 response produced (almost) no text."""
    lowered = raw_html.lower()
    if "_incapsula_" in lowered or "incapsula_resource" in lowered:
        return "bot protection (Incapsula)"
    if "cloudflare" in lowered and ("challenge" in lowered or "just a moment" in lowered):
        return "bot protection (Cloudflare)"
    if "enable javascript" in lowered or "javascript is disabled" in lowered:
        return "a JavaScript gate"
    if len(raw_html) > 100_000:
        # A huge document with no text is a single-page app shell, not a policy page.
        return "likely a JavaScript-rendered page (content not in the HTML)"
    if "captcha" in lowered:
        return "a CAPTCHA wall"
    return "likely JavaScript-rendered or bot-blocked"


def fetch_source(
    source: Source,
    *,
    session: Any | None = None,
    paths: Paths | None = None,
    previous_sha: str | None = None,
    now: datetime | None = None,
    sleep: Callable[[float], None] = time.sleep,
    retries: int | None = None,
) -> FetchResult:
    """Fetch one source and write raw HTML when the content is new.

    ``previous_sha`` is the last stored content hash. When it matches, nothing
    is written and ``unchanged=True`` is returned -- re-running ``fetch`` is a
    no-op for unchanged pages.
    """
    paths = paths or config.get_paths()
    moment, stamp = _utc_stamp(now)
    html = http_get(source.url, session=session, sleep=sleep, retries=retries)
    digest = hashlib.sha256(html.encode("utf-8")).hexdigest()

    if previous_sha and digest == previous_sha:
        return FetchResult(
            source_id=source.id,
            url=source.url,
            fetched_at=moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            content_sha256=digest,
            path=None,
            unchanged=True,
        )

    directory = paths.raw_dir / source.id
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{stamp}.html"
    if target.exists():  # two fetches within the same second
        target = directory / f"{stamp}-{digest[:8]}.html"
    target.write_text(html, encoding="utf-8")

    return FetchResult(
        source_id=source.id,
        url=source.url,
        fetched_at=moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        content_sha256=digest,
        path=target,
        unchanged=False,
    )


@dataclass
class FetchSummary:
    source_id: str
    status: str  # new | unchanged | error
    message: str = ""
    snapshot_id: int | None = None
    sections: int = 0
    report: Path | None = None


def fetch_all(
    sources: list[Source],
    store: Store,
    paths: Paths,
    *,
    only: list[str] | None = None,
    force: bool = False,
    session: Any | None = None,
    now: datetime | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[FetchSummary]:
    """Fetch, extract, split and persist every source. Errors are per-source."""
    from .diff import diff_source

    summaries: list[FetchSummary] = []
    for source in sources:
        if only and source.id not in only:
            continue
        store.upsert_source(
            source.id, source.country, source.name, source.url, source.selector
        )
        previous = None if force else store.latest_sha(source.id)
        try:
            result = fetch_source(
                source,
                session=session,
                paths=paths,
                previous_sha=previous,
                now=now,
                sleep=sleep,
            )
        except FetchError as exc:
            summaries.append(FetchSummary(source.id, "error", str(exc)))
            continue

        if result.unchanged:
            summaries.append(FetchSummary(source.id, "unchanged", "content hash matched"))
            continue

        assert result.path is not None  # unchanged was handled above
        raw_html = result.path.read_text(encoding="utf-8")
        text = html_to_text(raw_html, selector=source.selector)

        # A 200 response is not necessarily a page: JS-rendered single-page apps
        # and bot walls return shells. Refuse to store an empty snapshot, which
        # would otherwise masquerade as a policy change on the next diff.
        if len(" ".join(text.split())) < MIN_TEXT_CHARS:
            hint = _diagnose(raw_html)
            summaries.append(
                FetchSummary(
                    source.id,
                    "error",
                    f"extracted only {len(text)} chars from {len(raw_html)} bytes of HTML; "
                    f"reason: {hint}; raw HTML kept at {result.path}",
                )
            )
            continue

        sections = split_text(text)
        snapshot_id, created = store.add_snapshot(
            source.id,
            fetched_at=result.fetched_at,
            url=source.url,
            content_sha256=result.content_sha256,
            text=text,
            raw_path=str(result.path),
            sections=sections,
        )
        summary = FetchSummary(
            source.id,
            "new" if created else "unchanged",
            f"{len(sections)} sections",
            snapshot_id=snapshot_id,
            sections=len(sections),
        )
        if created and store.snapshot_count(source.id) >= 2:
            change_set = diff_source(store, source.id, paths)
            if change_set is not None:
                summary.report = change_set.report_path
                summary.message += f", {len(change_set.changes)} change(s)"
        summaries.append(summary)
    return summaries
