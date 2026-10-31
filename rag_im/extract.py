"""HTML -> markdown-ish plain text.

Government visa pages are buried under navigation, cookie banners, sidebars and
tracking scripts. We drop those wholesale, then walk the remaining block
structure so headings survive as ``#`` lines and paragraphs stay separate.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, NavigableString, Tag

NOISE_TAGS = (
    "script",
    "style",
    "noscript",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "svg",
    "iframe",
    "template",
    "button",
)

HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

# Some government sites wrap paragraph copy in heading tags for styling
# (ind.nl puts the pane subtitle inside an <h3>). A "heading" that is a full
# sentence is body text; promoting it would leave its real parent heading with
# an empty body and make the change report unreadable.
MAX_HEADING_CHARS = 120
BLOCK_TAGS = {
    "body",  # without these, a page walked from the document root is treated
    "html",  # as one text blob and every heading is lost
    "p",
    "div",
    "section",
    "article",
    "main",
    "blockquote",
    "pre",
    "li",
    "dd",
    "dt",
    "td",
    "th",
    "tr",
    "table",
    "ul",
    "ol",
    "figure",
    "figcaption",
}

_WS_RE = re.compile(r"[ \t\r\f\v]+")
_NL_RE = re.compile(r"\n{3,}")


def _clean(text: str) -> str:
    """Collapse horizontal whitespace; keep no leading/trailing noise."""
    text = text.replace("\xa0", " ")
    text = _WS_RE.sub(" ", text)
    return " ".join(line.strip() for line in text.splitlines()).strip()


def _walk(node: Tag, out: list[str]) -> None:
    for child in node.children:
        if isinstance(child, NavigableString):
            text = _clean(str(child))
            if text:
                out.append(text)
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name.lower()

        if name in HEADING_TAGS:
            text = _clean(child.get_text(" "))
            if not text:
                continue
            if len(text) > MAX_HEADING_CHARS or (len(text) > 60 and text.endswith(".")):
                out.append(text)
            else:
                out.append("#" * HEADING_TAGS[name] + " " + text)
            continue

        if name == "li":
            text = _clean(child.get_text(" "))
            if text:
                out.append("- " + text)
            continue

        if name in ("br", "hr"):
            continue

        # A heading counts as structure too: a <main> holding only <h2>/<h3>
        # siblings must be descended into, not flattened to one text blob.
        has_block_children = bool(child.find(BLOCK_TAGS | set(HEADING_TAGS), recursive=False))
        if not has_block_children:
            text = _clean(child.get_text(" "))
            if text:
                out.append(text)
        else:
            _walk(child, out)


def html_to_text(html: str, selector: str | None = None) -> str:
    """Convert an HTML document to markdown-ish text.

    ``selector`` is an optional CSS hint (e.g. ``main#content``) used to scope
    extraction to the content area. If it does not match, the whole document is
    used instead of failing -- government sites redesign often.
    """
    if not html or not html.strip():
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(list(NOISE_TAGS)):
        tag.decompose()

    root: Tag | BeautifulSoup = soup
    if selector:
        try:
            node = soup.select_one(selector)
        except Exception:  # invalid selector hint should not kill a fetch
            node = None
        if node is not None:
            root = node

    lines: list[str] = []
    _walk(root, lines)

    # Drop exact consecutive duplicates (nested wrappers often repeat text).
    deduped: list[str] = []
    for line in lines:
        if deduped and deduped[-1] == line:
            continue
        deduped.append(line)

    # Keep bullet runs together as one block; separate blocks by a blank line.
    blocks: list[str] = []
    run: list[str] = []
    for line in deduped:
        if line.startswith("- "):
            run.append(line)
            continue
        if run:
            blocks.append("\n".join(run))
            run = []
        blocks.append(line)
    if run:
        blocks.append("\n".join(run))
    text = "\n\n".join(blocks)
    text = _NL_RE.sub("\n\n", text)
    return text.strip()
