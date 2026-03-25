"""
Chapter word-budget planning node.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from loguru import logger

from ..core import TemplateSection
from ..prompts import (
    SYSTEM_PROMPT_WORD_BUDGET,
    build_word_budget_prompt,
)
from ..utils.json_parser import RobustJSONParser, JSONParseError
from .base_node import BaseNode


class WordBudgetNode(BaseNode):
    """
    Plan the word count and emphasis for each chapter.

    Outputs total word count, global writing guidelines, and per-chapter or
    per-section target, minimum, and maximum word-count constraints.
    """

    def __init__(self, llm_client):
        """Store the LLM client reference for use during run()."""
        super().__init__(llm_client, "WordBudgetNode")
        # Initialize the robust JSON parser with all repair strategies enabled.
        self.json_parser = RobustJSONParser(
            enable_json_repair=True,
            enable_llm_repair=False,  # Enable LLM-based repair if needed.
            max_repair_attempts=3,
        )

    def run(
        self,
        sections: List[TemplateSection],
        design: Dict[str, Any],
        reports: Dict[str, str],
        forum_logs: str,
        query: str,
        template_overview: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Plan chapter-level word counts from the design draft and all source
        material so the LLM writes against explicit length targets.

        Args:
            sections: Template chapter list.
            design: Design draft returned by the layout node, including fields
                such as title, toc, and hero.
            reports: Mapping of reports from the three engines.
            forum_logs: Raw forum log text.
            query: User query.
            template_overview: Optional template overview with chapter metadata.

        Returns:
            dict: Word-budget plan including totalWords, globalGuidelines, and
            chapter-level entries.
        """
        # Include both the chapter skeleton and the layout output so word-budget
        # planning can account for the visual hierarchy.
        payload = {
            "query": query,
            "design": design,
            "sections": [section.to_dict() for section in sections],
            "templateOverview": template_overview
            or {
                "title": sections[0].title if sections else "",
                "chapters": [section.to_dict() for section in sections],
            },
            "reports": reports,
            "forumLogs": forum_logs,
        }
        user = build_word_budget_prompt(payload)
        response = self.llm_client.stream_invoke_to_string(
            SYSTEM_PROMPT_WORD_BUDGET,
            user,
            temperature=0.25,
            top_p=0.85,
        )
        plan = self._parse_response(response)
        logger.info("Chapter word-budget plan generated")
        return plan

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """
        Convert the LLM JSON output into a dictionary and raise a readable error
        when planning fails.

        The robust JSON parser applies multiple repair strategies:
        1. Strip markdown wrappers and thinking text.
        2. Repair local syntax issues such as bracket balance, missing commas,
           and control-character escaping.
        3. Use the json_repair library for advanced repair.
        4. Optionally use LLM-assisted repair.

        Args:
            raw: Raw LLM output, which may include fenced code blocks or
                thinking text.

        Returns:
            dict: Valid word-budget JSON.

        Raises:
            ValueError: Raised when the response is empty or JSON parsing fails.
        """
        try:
            result = self.json_parser.parse(
                raw,
                context_name="word budget planning",
                expected_keys=["totalWords", "globalGuidelines", "chapters"],
            )
            # Validate required field types.
            if not isinstance(result.get("totalWords"), (int, float)):
                logger.warning("Word-budget plan is missing a valid totalWords field; using a default")
                result.setdefault("totalWords", 10000)
            if not isinstance(result.get("globalGuidelines"), list):
                logger.warning("Word-budget plan is missing valid globalGuidelines; using an empty list")
                result.setdefault("globalGuidelines", [])
            if not isinstance(result.get("chapters"), (list, dict)):
                logger.warning("Word-budget plan is missing valid chapters; using an empty list")
                result.setdefault("chapters", [])
            return result
        except JSONParseError as exc:
            # Keep the original exception type for backward compatibility.
            raise ValueError(f"Word-budget JSON parsing failed: {exc}") from exc


__all__ = ["WordBudgetNode"]
