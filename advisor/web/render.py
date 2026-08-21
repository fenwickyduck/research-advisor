"""Rendering the interest profile, which is Markdown, as HTML.

The profile was shown as preformatted text, so its own syntax — ``##``, ``**``
— was left on screen for the reader to mentally strip. This turns the small
subset the profile actually uses into markup.

Deliberately not a Markdown library. The profile is *stored text that a person
or an assistant wrote*, so it is untrusted input rendered back into the page:
the only safe shape is to escape everything first and then re-add a fixed,
closed set of tags. Nothing here can emit an attribute, so nothing here can
emit a URL, a handler or a style.
"""

from __future__ import annotations

import re
from html import escape

from markupsafe import Markup

# Applied to already-escaped text, so the delimiters cannot have come from a tag.
_INLINE = [
    (re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", re.S), r"<strong>\1</strong>"),
    (re.compile(r"(?<![\*\w])\*(?=\S)([^*]+?)(?<=\S)\*(?![\*\w])", re.S), r"<em>\1</em>"),
    (re.compile(r"`([^`]+)`"), r"<code>\1</code>"),
]

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _inline(text: str) -> str:
    for pattern, replacement in _INLINE:
        text = pattern.sub(replacement, text)
    return text


def richtext(source: str | None) -> Markup:
    """Render the profile's Markdown subset: headings, emphasis, code, lists."""
    if not source:
        return Markup("")

    out: list[str] = []
    bullets: list[str] = []

    def flush() -> None:
        if bullets:
            out.append("<ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>")
            bullets.clear()

    # Blank lines separate blocks; a single newline inside one is a line break,
    # which is how the steering sections are written.
    for block in re.split(r"\n\s*\n", escape(source.strip())):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        for line in lines:
            heading = _HEADING.match(line)
            if heading:
                flush()
                # The page already owns the <h1>, so the profile starts at <h2>.
                level = min(len(heading.group(1)) + 1, 6)
                out.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            elif line.startswith(("- ", "* ")):
                bullets.append(_inline(line[2:]))
            else:
                flush()
                out.append(f"<p>{_inline(line)}</p>")

    flush()
    return Markup("".join(out))
