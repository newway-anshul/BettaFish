"""
Markdown template slicing utility.

LLMs need to be called "per chapter", so the Markdown template must be parsed
into a structured chapter queue. This uses lightweight regex and indentation
heuristics, compatible with "# heading" and
"- **1.0 heading** /   - 1.1 subheading" and other formats.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

SECTION_ORDER_STEP = 10


@dataclass
class TemplateSection:
    """
    Template section entity.

    Records title, slug, order, depth, raw title, section number and outline,
    making it easy for subsequent nodes to reference in prompts and maintain anchor consistency.
    """

    title: str
    slug: str
    order: int
    depth: int
    raw_title: str
    number: str = ""
    chapter_id: str = ""
    outline: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """
        Serialize the section entity to a dictionary.

        This structure is widely used in prompt contexts and as inputs for layout/word budget nodes.
        """
        return {
            "title": self.title,
            "slug": self.slug,
            "order": self.order,
            "depth": self.depth,
            "number": self.number,
            "chapterId": self.chapter_id,
            "outline": self.outline,
        }


# Regex patterns intentionally avoid `.*` to maintain matching determinism,
# and mitigate regex DoS risks common with untrusted template text.
heading_pattern = re.compile(
    r"""
    (?P<marker>\#{1,6})       # Markdown heading marker
    [ \t]+                    # Required whitespace
    (?P<title>[^\r\n]+)       # Title text (no newlines)
    """,
    re.VERBOSE,
)
bullet_pattern = re.compile(
    r"""
    (?P<marker>[-*+])         # List bullet symbol
    [ \t]+
    (?P<title>[^\r\n]+)
    """,
    re.VERBOSE,
)
number_pattern = re.compile(
    r"""
    (?P<num>
        (?:0|[1-9]\d*)
        (?:\.(?:0|[1-9]\d*))*
    )
    (?:
        (?:[ \t\u00A0\u3000、:：-]+|\.(?!\d))+
        (?P<label>[^\r\n]*)
    )?
    """,
    re.VERBOSE,
)


def parse_template_sections(template_md: str) -> List[TemplateSection]:
    """
    Split a Markdown template into a chapter list (by major headings).

    Each returned TemplateSection carries slug/order/chapter number,
    facilitating per-chapter calls and anchor generation. Parsing is compatible
    with "# heading", "unnumbered", "list outline" and other formats.

    Args:
        template_md: Full Markdown text of the template.

    Returns:
        list[TemplateSection]: Structured sequence of sections.
    """

    sections: List[TemplateSection] = []
    current: Optional[TemplateSection] = None
    order = SECTION_ORDER_STEP
    used_slugs = set()

    for raw_line in template_md.splitlines():
        if not raw_line.strip():
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()

        meta = _classify_line(stripped, indent)
        if not meta:
            continue

        if meta["is_section"]:
            slug = _ensure_unique_slug(meta["slug"], used_slugs)
            section = TemplateSection(
                title=meta["title"],
                slug=slug,
                order=order,
                depth=meta["depth"],
                raw_title=meta["raw"],
                number=meta["number"],
            )
            sections.append(section)
            current = section
            order += SECTION_ORDER_STEP
            continue

        # Outline entry
        if current:
            current.outline.append(meta["title"])

    for idx, section in enumerate(sections, start=1):
        # Generate a stable chapter_id for each section for subsequent reference
        section.chapter_id = f"S{idx}"

    return sections


def _classify_line(stripped: str, indent: int) -> Optional[dict]:
    """
    Classify a line by indentation and symbol.

    Uses regex to determine if the current line is a section heading, outline,
    or ordinary list item, and derives depth/slug/number etc.

    Args:
        stripped: Original line with leading/trailing whitespace stripped.
        indent: Number of leading spaces, used to distinguish nesting level.

    Returns:
        dict | None: Recognized metadata; returns None if unrecognizable.
    """

    heading_match = heading_pattern.fullmatch(stripped)
    if heading_match:
        level = len(heading_match.group("marker"))
        payload = _strip_markup(heading_match.group("title").strip())
        title_info = _split_number(payload)
        slug = _build_slug(title_info["number"], title_info["title"])
        return {
            "is_section": level <= 2,
            "depth": level,
            "title": title_info["display"],
            "raw": payload,
            "number": title_info["number"],
            "slug": slug,
        }

    bullet_match = bullet_pattern.fullmatch(stripped)
    if bullet_match:
        payload = _strip_markup(bullet_match.group("title").strip())
        title_info = _split_number(payload)
        slug = _build_slug(title_info["number"], title_info["title"])
        is_section = indent <= 1
        depth = 1 if indent <= 1 else 2
        return {
            "is_section": is_section,
            "depth": depth,
            "title": title_info["display"],
            "raw": payload,
            "number": title_info["number"],
            "slug": slug,
        }

    # Also handle "1.1 ..." lines without a leading bullet symbol
    number_match = number_pattern.fullmatch(stripped)
    if number_match and number_match.group("label"):
        payload = stripped
        title = number_match.group("label").strip()
        number = number_match.group("num")
        slug = _build_slug(number, title)
        is_section = indent == 0 and number.count(".") <= 1
        depth = 1 if is_section else 2
        display = f"{number} {title}" if title else number
        return {
            "is_section": is_section,
            "depth": depth,
            "title": display,
            "raw": payload,
            "number": number,
            "slug": slug,
        }

    return None


def _strip_markup(text: str) -> str:
    """Remove wrapping ** or __ emphasis markers to avoid interfering with heading matching."""
    if text.startswith(("**", "__")) and text.endswith(("**", "__")) and len(text) > 4:
        return text[2:-2].strip()
    return text


def _split_number(payload: str) -> dict:
    """
    Split number from title.

    For example, `1.2 Market Trends` is split into number=1.2, label=Market Trends,
    and provides display for title backfill.

    Args:
        payload: Raw title string.

    Returns:
        dict: Contains number/title/display.
    """
    match = number_pattern.fullmatch(payload)
    number = match.group("num") if match else ""
    label = match.group("label") if match else payload
    label = (label or "").strip()
    display = f"{number} {label}".strip() if number else label or payload
    title_core = label or payload
    return {
        "number": number,
        "title": title_core,
        "display": display,
    }


def _build_slug(number: str, title: str) -> str:
    """
    Generate an anchor from number/title, preferring number and slugifying title if missing.

    Args:
        number: Section number.
        title: Title text.

    Returns:
        str: A slug in the form of `section-1-0`.
    """
    if number:
        token = number.replace(".", "-")
    else:
        token = _slugify_text(title)
    token = token or "section"
    return f"section-{token}"


def _slugify_text(text: str) -> str:
    """
    Apply noise reduction and transliteration to arbitrary text to get URL-friendly slug fragments.

    Normalizes case, removes special symbols, and retains CJK characters to ensure readable anchors.
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("·", "-").replace(" ", "-")
    text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-").lower()


def _ensure_unique_slug(slug: str, used: set) -> str:
    """
    If a slug is duplicate, automatically append a sequence number until unique in the used set.

    Uses -2/-3... to ensure identical titles do not produce duplicate anchors.

    Args:
        slug: Initial slug.
        used: Set of already-used slugs.

    Returns:
        str: De-duplicated slug.
    """
    if slug not in used:
        used.add(slug)
        return slug
    base = slug
    idx = 2
    while slug in used:
        slug = f"{base}-{idx}"
        idx += 1
    used.add(slug)
    return slug


__all__ = ["TemplateSection", "parse_template_sections"]
