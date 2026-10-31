"""Split extracted text into heading-anchored sections and hash them.

The hash is deliberately whitespace-insensitive: a page that is reflowed or
re-indented has not changed policy, and we do not want it in the change report.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .extract import html_to_text

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_WS_RE = re.compile(r"\s+")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def normalise(text: str) -> str:
    """Whitespace-insensitive form used for hashing and comparison."""
    return _WS_RE.sub(" ", text.replace("\xa0", " ")).strip()


def section_hash(heading: str, body: str) -> str:
    """Hash of the section *body* only.

    Deliberately excludes the heading: a section that is re-titled but not
    reworded is a move, not a policy edit, and the diff needs to see the body
    as unchanged. Whitespace is normalised so reflow is not a change either.
    """
    payload = normalise(body)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def slugify(heading: str) -> str:
    slug = _SLUG_RE.sub("-", heading.lower()).strip("-")
    return slug or "section"


@dataclass(frozen=True)
class Section:
    heading: str
    level: int
    body: str
    hash: str
    anchor: str
    ordinal: int = 0

    @property
    def title(self) -> str:
        return self.heading or "(preamble)"

    def __len__(self) -> int:
        return len(self.body)


def split_text(text: str) -> list[Section]:
    """Split markdown-ish text on heading lines.

    Text before the first heading becomes a level-0 ``(preamble)`` section, but
    only when it carries content.
    """
    lines = text.splitlines()
    raw: list[tuple[str, int, list[str]]] = []
    heading, level, buf = "", 0, []
    for line in lines:
        m = HEADING_RE.match(line.strip())
        if m:
            if heading or any(x.strip() for x in buf):
                raw.append((heading, level, buf))
            heading, level, buf = m.group(2).strip(), len(m.group(1)), []
        else:
            buf.append(line)
    if heading or any(x.strip() for x in buf):
        raw.append((heading, level, buf))

    sections: list[Section] = []
    counts: dict[str, int] = {}
    for ordinal, (heading, level, buf) in enumerate(raw):
        body = "\n".join(line for line in buf if line.strip())
        body = body.strip()
        if not heading and not body:
            continue
        base = slugify(heading or "preamble")
        counts[base] = counts.get(base, 0) + 1
        anchor = base if counts[base] == 1 else f"{base}~{counts[base]}"
        sections.append(
            Section(
                heading=heading or "(preamble)",
                level=level,
                body=body,
                hash=section_hash(heading, body),
                anchor=anchor,
                ordinal=ordinal,
            )
        )
    return sections


def split_html(html: str, selector: str | None = None) -> list[Section]:
    return split_text(html_to_text(html, selector=selector))


def render(sections: list[Section]) -> str:
    """Inverse of split_text, mainly for the CLI's ``document`` output."""
    chunks = []
    for sec in sections:
        if sec.level:
            chunks.append("#" * sec.level + " " + sec.heading)
        chunks.append(sec.body)
    return "\n\n".join(c for c in chunks if c)
