"""
Generate the full report title, table of contents, and theme design from the
template outline and multi-source reports.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from loguru import logger

from ..core import TemplateSection
from ..prompts import (
    SYSTEM_PROMPT_DOCUMENT_LAYOUT,
    build_document_layout_prompt,
)
from ..utils.json_parser import RobustJSONParser, JSONParseError
from .base_node import BaseNode


class DocumentLayoutNode(BaseNode):
    """
    Generate the global title, table of contents, and hero design.

    Combines template slices, report summaries, and forum discussion to define
    the visual and structural direction of the report.
    """

    def __init__(self, llm_client):
        """Store the LLM client and set the node name for BaseNode logging."""
        super().__init__(llm_client, "DocumentLayoutNode")
        # Initialize the robust JSON parser with all repair strategies enabled.
        self.json_parser = RobustJSONParser(
            enable_json_repair=True,
            enable_llm_repair=False,  # Enable LLM-based repair if needed.
            max_repair_attempts=3,
        )

    def run(
        self,
        sections: List[TemplateSection],
        template_markdown: str,
        reports: Dict[str, str],
        forum_logs: str,
        query: str,
        template_overview: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Generate the report title, table-of-contents structure, and theme tokens
        from the template and source material.

        Args:
            sections: Chapter list produced from template slicing.
            template_markdown: Raw template markdown for LLM context.
            reports: Content mapping from the three engines.
            forum_logs: Forum discussion summary.
            query: User query.
            template_overview: Optional prebuilt template overview used to keep
                prompts shorter.

        Returns:
            dict: Design metadata including title, subtitle, toc, hero, and
            theme tokens.
        """
        # Send the raw template, sliced structure, and source reports together
        # so the LLM can infer hierarchy and available content.
        payload = {
            "query": query,
            "template": {
                "raw": template_markdown,
                "sections": [section.to_dict() for section in sections],
            },
            "templateOverview": template_overview
            or {
                "title": sections[0].title if sections else "",
                "chapters": [section.to_dict() for section in sections],
            },
            "reports": reports,
            "forumLogs": forum_logs,
        }

        user_message = build_document_layout_prompt(payload)
        response = self.llm_client.stream_invoke_to_string(
            SYSTEM_PROMPT_DOCUMENT_LAYOUT,
            user_message,
            temperature=0.3,
            top_p=0.9,
        )
        design = self._parse_response(response)
        logger.info("Document title and table-of-contents design generated")
        return design

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """
        Parse the JSON returned by the LLM and raise a friendly error on
        failure.

        The robust JSON parser applies multiple repair strategies:
        1. Strip markdown wrappers and chain-of-thought style content.
        2. Repair local syntax issues such as bracket balance, missing commas,
           or control-character escaping.
        3. Use the json_repair library for deeper fixes.
        4. Optionally use LLM-assisted repair.

        Args:
            raw: Raw LLM response, which may include fenced code blocks or
                thinking text.

        Returns:
            dict: Structured design output.

        Raises:
            ValueError: Raised when the response is empty or JSON parsing fails.
        """
        try:
            result = self.json_parser.parse(
                raw,
                context_name="document layout",
                # The directory field was renamed to tocPlan; validate against
                # the latest schema.
                expected_keys=["title", "tocPlan", "hero"],
            )
            # Validate the types of required fields.
            if not isinstance(result.get("title"), str):
                logger.warning("Document layout is missing a valid title; using a default")
                result.setdefault("title", "Untitled Report")

            # Normalize the tocPlan field.
            toc_plan = result.get("tocPlan", [])
            if not isinstance(toc_plan, list):
                logger.warning("Document layout is missing a valid tocPlan; using an empty list")
                result["tocPlan"] = []
            else:
                # Clean the description fields inside tocPlan.
                result["tocPlan"] = self._clean_toc_plan_descriptions(toc_plan)

            if not isinstance(result.get("hero"), dict):
                logger.warning("Document layout is missing a valid hero object; using an empty dict")
                result.setdefault("hero", {})

            return result
        except JSONParseError as exc:
            # Keep the original exception type for backward compatibility.
            raise ValueError(f"Document layout JSON parsing failed: {exc}") from exc

    def _clean_toc_plan_descriptions(self, toc_plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean the description field in each tocPlan entry and remove likely JSON
        fragments.

        Args:
            toc_plan: Original table-of-contents plan list.

        Returns:
            List[Dict[str, Any]]: Cleaned table-of-contents plan list.
        """
        import re

        def clean_text(text: Any) -> str:
            """Remove JSON-like fragments from a text value."""
            if not text or not isinstance(text, str):
                return ""

            cleaned = text

            # Remove an incomplete JSON object starting with a comma and `{`.
            cleaned = re.sub(r',\s*\{[^}]*$', '', cleaned)

            # Remove an incomplete JSON array starting with a comma and `[`. 
            cleaned = re.sub(r',\s*\[[^\]]*$', '', cleaned)

            # Remove a trailing unmatched `{` segment.
            open_brace_pos = cleaned.rfind('{')
            if open_brace_pos != -1:
                close_brace_pos = cleaned.rfind('}')
                if close_brace_pos < open_brace_pos:
                    cleaned = cleaned[:open_brace_pos].rstrip(', \t\n')

            # Remove a trailing unmatched `[` segment.
            open_bracket_pos = cleaned.rfind('[')
            if open_bracket_pos != -1:
                close_bracket_pos = cleaned.rfind(']')
                if close_bracket_pos < open_bracket_pos:
                    cleaned = cleaned[:open_bracket_pos].rstrip(', \t\n')

            # Remove fragments that look like JSON key-value pairs.
            cleaned = re.sub(r',?\s*"[^"]+"\s*:\s*"[^"]*$', '', cleaned)
            cleaned = re.sub(r',?\s*"[^"]+"\s*:\s*[^,}\]]*$', '', cleaned)

            # Trim trailing commas and whitespace.
            cleaned = cleaned.rstrip(', \t\n')

            return cleaned.strip()

        cleaned_plan = []
        for entry in toc_plan:
            if not isinstance(entry, dict):
                continue

            # Clean the description field.
            if "description" in entry:
                original_desc = entry["description"]
                cleaned_desc = clean_text(original_desc)

                if cleaned_desc != original_desc:
                    logger.warning(
                        f"Removed JSON fragments from the description of TOC entry '{entry.get('display', 'unknown')}':\n"
                        f"  Before: {original_desc[:100]}...\n"
                        f"  After: {cleaned_desc[:100]}..."
                    )
                    entry["description"] = cleaned_desc

            cleaned_plan.append(entry)

        return cleaned_plan


__all__ = ["DocumentLayoutNode"]
