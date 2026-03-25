"""
Chapter stitcher: responsible for merging multiple chapter JSONs into a complete IR.

DocumentComposer injects missing anchors, unifies ordering, and supplements IR-level metadata.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Set

from ..ir import IR_VERSION


class DocumentComposer:
    """
    Simple stitcher that assembles chapters into Document IR.

    Role:
        - Sort chapters by order and supplement default chapterId;
        - Prevent duplicate anchors, generating globally unique anchors;
        - Inject IR version and generation timestamp.
    """

    def __init__(self):
        """Initialize the stitcher and record used anchors to avoid duplication."""
        self._seen_anchors: Set[str] = set()

    def build_document(
        self,
        report_id: str,
        metadata: Dict[str, object],
        chapters: List[Dict[str, object]],
    ) -> Dict[str, object]:
        """
        Sort all chapters by order, inject unique anchors, and form a complete IR.

        Also merges metadata/themeTokens/assets for direct renderer consumption.

        Args:
            report_id: Report ID for this run.
            metadata: Global metadata (title, topic, toc, etc.).
            chapters: List of chapter payloads.

        Returns:
            dict: Document IR satisfying renderer requirements.
        """
        # Build mapping from chapterId to toc anchor
        toc_anchor_map = self._build_toc_anchor_map(metadata)

        ordered = sorted(chapters, key=lambda c: c.get("order", 0))
        for idx, chapter in enumerate(ordered, start=1):
            chapter.setdefault("chapterId", f"S{idx}")

            # Priority: 1. anchor from toc config  2. chapter's own anchor  3. default anchor
            chapter_id = chapter.get("chapterId")
            anchor = (
                toc_anchor_map.get(chapter_id) or
                chapter.get("anchor") or
                f"section-{idx}"
            )
            chapter["anchor"] = self._ensure_unique_anchor(anchor)
            chapter.setdefault("order", idx * 10)
            if chapter.get("errorPlaceholder"):
                self._ensure_heading_block(chapter)

        document = {
            "version": IR_VERSION,
            "reportId": report_id,
            "metadata": {
                **metadata,
                "generatedAt": metadata.get("generatedAt")
                or datetime.utcnow().isoformat() + "Z",
            },
            "themeTokens": metadata.get("themeTokens", {}),
            "chapters": ordered,
            "assets": metadata.get("assets", {}),
        }
        return document

    def _ensure_unique_anchor(self, anchor: str) -> str:
        """If a duplicate anchor exists, append a sequence number to ensure global uniqueness."""
        base = anchor
        counter = 2
        while anchor in self._seen_anchors:
            anchor = f"{base}-{counter}"
            counter += 1
        self._seen_anchors.add(anchor)
        return anchor

    def _build_toc_anchor_map(self, metadata: Dict[str, object]) -> Dict[str, str]:
        """
        Build a chapterId to anchor mapping from metadata.toc.customEntries.

        Args:
            metadata: Document metadata.

        Returns:
            dict: Mapping of chapterId -> anchor.
        """
        toc_config = metadata.get("toc") or {}
        custom_entries = toc_config.get("customEntries") or []
        anchor_map = {}

        for entry in custom_entries:
            if isinstance(entry, dict):
                chapter_id = entry.get("chapterId")
                anchor = entry.get("anchor")
                if chapter_id and anchor:
                    anchor_map[chapter_id] = anchor

        return anchor_map

    def _ensure_heading_block(self, chapter: Dict[str, object]) -> None:
        """Ensure placeholder chapters still have a heading block usable by the table of contents."""
        blocks = chapter.get("blocks")
        if isinstance(blocks, list):
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "heading":
                    return
        heading = {
            "type": "heading",
            "level": 2,
            "text": chapter.get("title") or "Placeholder Section",
            "anchor": chapter.get("anchor"),
        }
        if isinstance(blocks, list):
            blocks.insert(0, heading)
        else:
            chapter["blocks"] = [heading]


__all__ = ["DocumentComposer"]
