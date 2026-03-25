"""
Chapter-level JSON generation node.

Each chapter is generated independently from Markdown template slices,
streamed into a raw file, then validated and persisted as normalized JSON.
This node is only responsible for producing compliant chapter output.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple, Callable, Optional, Set

from loguru import logger

from ..core import TemplateSection, ChapterStorage
from ..ir import (
    ALLOWED_BLOCK_TYPES,
    ALLOWED_INLINE_MARKS,
    ENGINE_AGENT_TITLES,
    IRValidator,
)
from ..prompts import (
    SYSTEM_PROMPT_CHAPTER_JSON,
    SYSTEM_PROMPT_CHAPTER_JSON_REPAIR,
    SYSTEM_PROMPT_CHAPTER_JSON_RECOVERY,
    build_chapter_repair_prompt,
    build_chapter_recovery_payload,
    build_chapter_user_prompt,
)
from ..utils.json_parser import RobustJSONParser, JSONParseError
from .base_node import BaseNode

try:
    from json_repair import repair_json as _json_repair_fn
except ImportError:  # pragma: no cover - optional dependency
    _json_repair_fn = None


class ChapterJsonParseError(ValueError):
    """Raised when chapter LLM output cannot be parsed as valid JSON, with raw text attached."""

    def __init__(self, message: str, raw_text: Optional[str] = None):
        """
        Construct the exception and attach raw output for easier log diagnosis.

        Args:
            message: Human-readable error description.
            raw_text: Full LLM output that triggered this exception.
        """
        super().__init__(message)
        self.raw_text = raw_text


class ChapterContentError(ValueError):
    """
    Sparse chapter content exception.

    Triggered when LLM output is only a heading or body content is too thin
    to support a complete chapter, forcing retries for report quality.
    """

    def __init__(
        self,
        message: str,
        chapter: Optional[Dict[str, Any]] = None,
        body_characters: int = 0,
        narrative_characters: int = 0,
        non_heading_blocks: int = 0,
    ):
        """Store body features for retry and fallback strategy decisions."""
        super().__init__(message)
        self.chapter_payload: Optional[Dict[str, Any]] = chapter
        self.body_characters: int = int(body_characters or 0)
        self.narrative_characters: int = int(narrative_characters or 0)
        self.non_heading_blocks: int = int(non_heading_blocks or 0)


class ChapterValidationError(ValueError):
    """
    Raised when chapter structure still fails validation after local and LLM repair.

    Used by the agent layer to retry only the failed chapter instead of restarting the whole report.
    """

    def __init__(self, message: str, errors: Optional[List[str]] | None = None):
        super().__init__(message)
        self.errors: List[str] = list(errors or [])


class ChapterGenerationNode(BaseNode):
    """
    Calls LLM per chapter and validates JSON structure.

    Core capabilities:
        - Build chapter-level payload and prompts;
        - Stream writes to raw files and forward deltas;
        - Repair/parse LLM output and validate with IRValidator;
        - Tolerantly fix block structures to keep final JSON renderable.
    """

    _COLON_EQUALS_PATTERN = re.compile(r'(":\s*)=')
    _LINE_BREAK_SENTINEL = "__LINE_BREAK__"
    _INLINE_MARK_ALIASES = {
        "strong": "bold",
        "b": "bold",
        "em": "italic",
        "emphasis": "italic",
        "i": "italic",
        "u": "underline",
        "strike-through": "strike",
        "strikethrough": "strike",
        "s": "strike",
        "codeblock": "code",
        "monospace": "code",
        "hyperlink": "link",
        "url": "link",
        "colour": "color",
        "textcolor": "color",
        "bgcolor": "highlight",
        "background": "highlight",
        "highlightcolor": "highlight",
        "sub": "subscript",
        "sup": "superscript",
    }
    # If a chapter only contains headings or too few characters, treat it as failed and regenerate.
    _MIN_NON_HEADING_BLOCKS = 2
    _MIN_BODY_CHARACTERS = 600
    _MIN_NARRATIVE_CHARACTERS = 300
    _PARAGRAPH_FRAGMENT_MAX_CHARS = 80
    _PARAGRAPH_FRAGMENT_NO_TERMINATOR_MAX_CHARS = 240
    _TERMINATION_PUNCTUATION = set("。！？!?；;……")

    def __init__(
        self,
        llm_client,
        validator: IRValidator,
        storage: ChapterStorage,
        fallback_llm_clients: Optional[List[Tuple[str, Any]]] = None,
        error_log_dir: Optional[str | Path] = None,
    ):
        """
        Store the LLM client, validator, and chapter storage for run orchestration.

        Args:
            llm_client: Client used to invoke the LLM
            validator: IR structure validator
            storage: Storage backend for chapter stream persistence
        """
        super().__init__(llm_client, "ChapterGenerationNode")
        self.validator = validator
        self.storage = storage
        self.fallback_llm_clients: List[Tuple[str, Any]] = fallback_llm_clients or [
            ("report_engine", llm_client)
        ]
        error_dir = Path(error_log_dir or "logs/json_repair_failures")
        error_dir.mkdir(parents=True, exist_ok=True)
        self.error_log_dir = error_dir
        self._failed_block_counter = 0
        self._active_run_id: Optional[str] = None
        self._rescue_attempted_labels: Dict[str, Set[str]] = {}
        self._skipped_placeholder_chapters: Set[str] = set()
        self._archived_failed_json: Dict[str, str] = {}
        # Fallback to a robust JSON parser to recover valid chunks where possible.
        self._robust_parser = RobustJSONParser(
            enable_json_repair=True,
            enable_llm_repair=False,
        )

    def run(
        self,
        section: TemplateSection,
        context: Dict[str, Any],
        run_dir: Path,
        stream_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Invoke LLM for a single chapter, validate/persist chapter JSON, and return structured output.

        Args:
            section: Chapter object from template slicing (title/order/slug).
            context: Shared context built by the agent (topic, length, layout, etc.).
            run_dir: Chapter persistence directory returned by `ChapterStorage.start_session`.
            stream_callback: Optional streaming callback to push LLM deltas to frontend.
            **kwargs: Sampling parameters such as temperature and top_p.

        Returns:
            dict: Chapter JSON that passed IR validation.

        Raises:
            ChapterJsonParseError: Still cannot parse valid JSON after retries.
            ChapterContentError: Body density is too low or only headings exist, triggering retry.
        """
        chapter_meta = {
            "chapterId": section.chapter_id,
            "slug": section.slug,
            "title": section.title,
            "order": section.order,
        }
        chapter_dir = self.storage.begin_chapter(run_dir, chapter_meta)
        run_id = run_dir.name
        self._ensure_run_state(run_id)
        llm_payload = self._build_payload(section, context)
        user_message = build_chapter_user_prompt(llm_payload)

        raw_text = self._stream_llm(
            user_message,
            chapter_dir,
            stream_callback=stream_callback,
            section_meta=chapter_meta,
            **kwargs,
        )
        parse_context: List[str] = []
        placeholder_created = False
        try:
            chapter_json = self._parse_chapter(raw_text)
        except ChapterJsonParseError as parse_error:
            logger.warning(f"Chapter JSON parsing failed for {section.title}; trying cross-engine repair: {parse_error}")
            parse_context.append(str(parse_error))
            self._archive_failed_output(section, raw_text)
            recovered = self._attempt_cross_engine_json_rescue(
                section,
                llm_payload,
                raw_text,
                run_id,
            )
            if recovered:
                chapter_json = recovered
                logger.info(f"Chapter JSON for {section.title} has been repaired via cross-engine recovery")
            else:
                placeholder = self._build_placeholder_chapter(section, raw_text, parse_error)
                if not placeholder:
                    raise
                chapter_json, placeholder_notes = placeholder
                parse_context.extend(placeholder_notes)
                placeholder_created = True

        # Auto-fill key fields before validation.
        chapter_json.setdefault("chapterId", section.chapter_id)
        chapter_json.setdefault("anchor", section.slug)
        chapter_json.setdefault("title", section.title)
        chapter_json.setdefault("order", section.order)
        self._sanitize_chapter_blocks(chapter_json)

        valid, errors = self.validator.validate_chapter(chapter_json)
        if not valid and errors:
            repaired = self._attempt_llm_structural_repair(
                chapter_json,
                errors,
                raw_text=raw_text,
            )
            if repaired:
                chapter_json = repaired
                chapter_json.setdefault("chapterId", section.chapter_id)
                chapter_json.setdefault("anchor", section.slug)
                chapter_json.setdefault("title", section.title)
                chapter_json.setdefault("order", section.order)
                self._sanitize_chapter_blocks(chapter_json)
                valid, errors = self.validator.validate_chapter(chapter_json)
        content_error: ChapterContentError | None = None
        if valid and not placeholder_created:
            try:
                self._ensure_content_density(chapter_json)
            except ChapterContentError as exc:
                content_error = exc

        error_messages: List[str] = parse_context.copy()
        if not valid and errors:
            error_messages.extend(errors)
        if content_error:
            error_messages.append(str(content_error))

        self.storage.persist_chapter(
            run_dir,
            chapter_meta,
            chapter_json,
            errors=None if not error_messages else error_messages,
        )

        if not valid:
            raise ChapterValidationError(
                f"Chapter JSON validation failed for {section.title}: {'; '.join(errors[:5])}",
                errors=errors,
            )
        if content_error:
            raise content_error

        return chapter_json

    # ====== Internal methods ======

    def _build_payload(self, section: TemplateSection, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build the LLM input payload.

        Args:
            section: Current chapter to generate, including title/number/outline.
            context: Global context dict containing topic, multi-engine reports, length plan, etc.

        Returns:
            dict: Prompt-serializable payload combining chapter info and global constraints.
        """
        reports = context.get("reports", {})
        # Chapter length planning (from WordBudgetNode), used to guide length and emphasis.
        chapter_plan_map = context.get("chapter_directives", {})
        chapter_plan = chapter_plan_map.get(section.chapter_id) if chapter_plan_map else {}

        # Determine whether this chapter allows SWOT and PEST blocks from layout.tocPlan.
        allow_swot = self._get_chapter_swot_permission(section.chapter_id, context)
        allow_pest = self._get_chapter_pest_permission(section.chapter_id, context)

        payload = {
            "section": {
                "chapterId": section.chapter_id,
                "title": section.title,
                "slug": section.slug,
                "order": section.order,
                "number": section.number,
                "outline": section.outline,
            },
            "globalContext": {
                "query": context.get("query"),
                "templateName": context.get("template_name"),
                "themeTokens": context.get("theme_tokens", {}),
                "styleDirectives": context.get("style_directives", {}),
                # layout contains title/toc/hero info to keep visual tone consistent across chapters.
                "layout": context.get("layout"),
                "templateOverview": context.get("template_overview", {}),
            },
            "reports": {
                "query_engine": reports.get("query_engine", ""),
                "media_engine": reports.get("media_engine", ""),
                "insight_engine": reports.get("insight_engine", ""),
            },
            "forumLogs": context.get("forum_logs", ""),
            "dataBundles": context.get("data_bundles", []),
            "constraints": {
                "language": "zh-CN",
                "maxTokens": context.get("max_tokens", 4096),
                "allowedBlocks": ALLOWED_BLOCK_TYPES,
                "allowSwot": allow_swot,
                "allowPest": allow_pest,
                "styleHints": {
                    "expectWidgets": True,
                    "forceHeadingAnchors": True,
                    "allowInlineMix": True,
                },
            },
            "chapterPlan": chapter_plan,
            "wordPlan": context.get("word_plan"),
        }
        if chapter_plan:
            constraints = payload["constraints"]
            if chapter_plan.get("targetWords"):
                constraints["wordTarget"] = chapter_plan["targetWords"]
            if chapter_plan.get("minWords"):
                constraints["minWords"] = chapter_plan["minWords"]
            if chapter_plan.get("maxWords"):
                constraints["maxWords"] = chapter_plan["maxWords"]
            if chapter_plan.get("emphasis"):
                constraints["emphasis"] = chapter_plan["emphasis"]
            if chapter_plan.get("sections"):
                constraints["sectionBudgets"] = chapter_plan["sections"]
                payload["globalContext"]["sectionBudgets"] = chapter_plan["sections"]
        return payload

    def _get_chapter_swot_permission(self, chapter_id: str, context: Dict[str, Any]) -> bool:
        """
        Check whether a chapter is allowed to use SWOT blocks from layout.tocPlan.

        At most one chapter in the report may use a SWOT block, marked by
        the allowSwot field during document design.

        Args:
            chapter_id: Current chapter ID.
            context: Global context dictionary.

        Returns:
            bool: True if SWOT blocks are allowed for this chapter, otherwise False.
        """
        layout = context.get("layout")
        if not isinstance(layout, dict):
            return False

        toc_plan = layout.get("tocPlan")
        if not isinstance(toc_plan, list):
            return False

        for entry in toc_plan:
            if not isinstance(entry, dict):
                continue
            if entry.get("chapterId") == chapter_id:
                return bool(entry.get("allowSwot", False))

        return False

    def _get_chapter_pest_permission(self, chapter_id: str, context: Dict[str, Any]) -> bool:
        """
        Check whether a chapter is allowed to use PEST blocks from layout.tocPlan.

        At most one chapter in the report may use a PEST block, marked by
        the allowPest field during document design.

        PEST blocks are used for macro-environment analysis:
        - Political
        - Economic
        - Social
        - Technological

        Args:
            chapter_id: Current chapter ID.
            context: Global context dictionary.

        Returns:
            bool: True if PEST blocks are allowed for this chapter, otherwise False.
        """
        layout = context.get("layout")
        if not isinstance(layout, dict):
            return False

        toc_plan = layout.get("tocPlan")
        if not isinstance(toc_plan, list):
            return False

        for entry in toc_plan:
            if not isinstance(entry, dict):
                continue
            if entry.get("chapterId") == chapter_id:
                return bool(entry.get("allowPest", False))

        return False

    def _stream_llm(
        self,
        user_message: str,
        chapter_dir: Path,
        stream_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        section_meta: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """
        Stream LLM output to raw file in real time and forward deltas via callback.

        Args:
            user_message: Composed user prompt.
            chapter_dir: Local chapter cache directory for storing stream.raw.
            stream_callback: SSE callback used to push stream deltas.
            section_meta: Chapter ID/title metadata included in callback payload.
            **kwargs: Parameters such as temperature and top_p.

        Returns:
            str: Raw text composed by concatenating all deltas.
        """
        chunks: List[str] = []
        with self.storage.capture_stream(chapter_dir) as stream_fp:
            stream = self.llm_client.stream_invoke(
                SYSTEM_PROMPT_CHAPTER_JSON,
                user_message,
                temperature=kwargs.get("temperature", 0.2),
                top_p=kwargs.get("top_p", 0.95),
            )
            for delta in stream:
                stream_fp.write(delta)
                chunks.append(delta)
                if stream_callback:
                    meta = section_meta or {}
                    try:
                        stream_callback(delta, meta)
                    except Exception as callback_error:  # pragma: no cover - log only, do not block main flow
                        logger.warning(f"Chapter streaming callback failed: {callback_error}")
        return "".join(chunks)

    def _attempt_cross_engine_json_rescue(
        self,
        section: TemplateSection,
        generation_payload: Dict[str, Any],
        raw_text: str,
        run_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Try repairing unparseable JSON through Report/Forum/Insight/Media APIs in sequence.

        Returns:
            dict | None: Repaired chapter JSON on success, otherwise None.
        """
        if not self.fallback_llm_clients:
            return None
        if self._chapter_already_skipped(section):
            logger.info(f"[{run_id}] {section.title} is already marked as placeholder; skipping cross-engine repair")
            return None
        section_payload = {
            "chapterId": section.chapter_id,
            "title": section.title,
            "slug": section.slug,
            "order": section.order,
            "number": section.number,
            "outline": section.outline,
        }
        repair_prompt = build_chapter_recovery_payload(
            section_payload,
            generation_payload,
            raw_text,
        )
        attempted_labels = self._rescue_attempted_labels.setdefault(section.chapter_id, set())
        for label, client in self.fallback_llm_clients:
            if label in attempted_labels:
                continue
            attempt_index = len(attempted_labels) + 1
            attempted_labels.add(label)
            logger.info(
                f"[{run_id}] Chapter {section.title} triggered {label} API JSON rescue (attempt {attempt_index})"
            )
            try:
                response = client.invoke(
                    SYSTEM_PROMPT_CHAPTER_JSON_RECOVERY,
                    repair_prompt,
                    temperature=0.0,
                    top_p=0.05,
                )
            except Exception as exc:
                logger.warning(f"{label} JSON repair invocation failed: {exc}")
                continue
            if not response:
                continue
            try:
                repaired = self._parse_chapter(response)
            except Exception as exc:
                logger.warning(f"{label} JSON repair output is still unparseable: {exc}")
                continue
            logger.warning(f"[{run_id}] {label} API repaired chapter JSON")
            self._archived_failed_json.pop(section.chapter_id, None)
            return repaired
        return None

    def _ensure_run_state(self, run_id: str):
        """Keep repair state isolated per report run to avoid cross-run contamination."""
        if self._active_run_id == run_id:
            return
        self._active_run_id = run_id
        self._rescue_attempted_labels = {}
        self._skipped_placeholder_chapters = set()
        self._archived_failed_json = {}

    def _archive_failed_output(self, section: TemplateSection, raw_text: str):
        """Cache current chapter raw failed JSON for later placeholder/manual usage."""
        if not raw_text:
            return
        self._archived_failed_json[section.chapter_id] = raw_text

    def _get_archived_failed_output(self, section: TemplateSection) -> Optional[str]:
        """Get most recent raw failed output for this chapter."""
        return self._archived_failed_json.get(section.chapter_id)

    def _mark_chapter_skipped(self, section: TemplateSection):
        """Mark chapter as placeholder-downgraded to avoid repeated cross-engine repair."""
        self._skipped_placeholder_chapters.add(section.chapter_id)

    def _chapter_already_skipped(self, section: TemplateSection) -> bool:
        """Check whether chapter has already been marked as placeholder."""
        return section.chapter_id in self._skipped_placeholder_chapters

    def _build_placeholder_chapter(
        self,
        section: TemplateSection,
        raw_text: str,
        parse_error: Exception,
    ) -> Optional[Tuple[Dict[str, Any], List[str]]]:
        """
        Build a renderable placeholder chapter when all repairs fail, and record logs for troubleshooting.
        """
        snapshot = self._get_archived_failed_output(section) or raw_text
        log_ref = self._persist_error_payload(section, snapshot, parse_error)
        if not log_ref:
            logger.error(f"Chapter JSON for {section.title} is fully corrupted and failed to write error log")
            return None
        importance = "critical" if self._is_section_critical(section) else "standard"
        message = (
            f"LLM returned block parsing errors. See record {log_ref['entryId']} in {log_ref['relativeFile']} for details."
        )
        heading_block = {
            "type": "heading",
            "level": 2 if importance == "critical" else 3,
            "text": section.title,
            "anchor": section.slug,
        }
        callout_block = {
            "type": "callout",
            "tone": "danger" if importance == "critical" else "warning",
            "title": "LLM Returned Block Parsing Error",
            "blocks": [
                {
                    "type": "paragraph",
                    "inlines": [
                        {
                            "text": message,
                        }
                    ],
                }
            ],
            "meta": {
                "errorLogRef": log_ref,
                "rawJsonPreview": (snapshot or "")[:2000],
                "errorMessage": message,
                "importance": importance,
            },
        }
        placeholder = {
            "chapterId": section.chapter_id,
            "title": section.title,
            "anchor": section.slug,
            "order": section.order,
            "blocks": [heading_block, callout_block],
            "errorPlaceholder": True,
        }
        errors = [
            f"Chapter JSON parsing failed for {section.title}; downgraded to placeholder. Refer to {log_ref['relativeFile']}#{log_ref['entryId']}"
        ]
        self._mark_chapter_skipped(section)
        return placeholder, errors

    def _parse_chapter(self, raw_text: str) -> Dict[str, Any]:
        """
        Clean LLM output and parse JSON.

        Args:
            raw_text: Raw LLM output (may include ``` wrappers or extra explanations).

        Returns:
            dict: Chapter JSON object, containing at least chapterId/title/blocks.

        Raises:
            ChapterJsonParseError: Valid JSON still cannot be parsed after multiple repair strategies.
        """
        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        if not cleaned:
            raise ChapterJsonParseError("LLM returned empty content", raw_text=raw_text)

        candidate_payloads = [cleaned]
        repaired = self._repair_llm_json(cleaned)
        if repaired != cleaned:
            candidate_payloads.append(repaired)

        data: Dict[str, Any] | None = None
        try:
            data = self._parse_with_candidates(candidate_payloads)
        except json.JSONDecodeError:
            repaired_payload = self._attempt_json_repair(cleaned)
            if repaired_payload:
                candidate_payloads.append(repaired_payload)
                try:
                    data = self._parse_with_candidates(candidate_payloads[-1:])
                except json.JSONDecodeError:
                    data = None
            if data is None:
                try:
                    data = self._robust_parser.parse(
                        cleaned,
                        context_name="ChapterJSON",
                        expected_keys=["chapter", "blocks", "chapterId", "title"],
                    )
                except JSONParseError as robust_exc:
                    raise ChapterJsonParseError(
                        f"Chapter JSON parsing failed: {robust_exc}", raw_text=cleaned
                    ) from robust_exc

        if "chapter" in data and isinstance(data["chapter"], dict):
            return data["chapter"]
        if isinstance(data, dict) and all(
            key in data for key in ("chapterId", "title", "blocks")
        ):
            return data
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if "chapter" in item and isinstance(item["chapter"], dict):
                        return item["chapter"]
                    if all(key in item for key in ("chapterId", "title", "blocks")):
                        return item
        raise ChapterJsonParseError("Chapter JSON is missing chapter field or has incomplete structure", raw_text=cleaned)

    def _persist_error_payload(
        self,
        section: TemplateSection,
        raw_text: str,
        parse_error: Exception,
    ) -> Optional[Dict[str, str]]:
        """Persist unparseable JSON payload to disk so HTML can point to a concrete file."""
        try:
            self._failed_block_counter += 1
            entry_id = f"E{self._failed_block_counter:04d}"
            timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            slug = section.slug or "section"
            filename = f"{timestamp}-{slug}-{entry_id}.json"
            file_path = self.error_log_dir / filename
            payload = {
                "chapterId": section.chapter_id,
                "title": section.title,
                "slug": section.slug,
                "order": section.order,
                "rawOutput": raw_text,
                "error": str(parse_error),
                "loggedAt": timestamp,
            }
            file_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            try:
                relative_path = str(file_path.relative_to(Path.cwd()))
            except ValueError:
                relative_path = str(file_path)
            return {
                "file": str(file_path),
                "relativeFile": relative_path,
                "entryId": entry_id,
                "timestamp": timestamp,
            }
        except Exception as exc:
            logger.error(f"Failed to record chapter JSON error log: {exc}")
            return None

    def _is_section_critical(self, section: TemplateSection) -> bool:
        """Decide prompt severity from chapter depth/numbering impact on table of contents."""
        if not section:
            return False
        if section.depth <= 2:
            return True
        number = section.number or ""
        if number and number.count(".") <= 1:
            return True
        return False

    def _repair_llm_json(self, text: str) -> str:
        """
        Handle common LLM mistakes (e.g., invalid JSON caused by ":=").

        Args:
            text: Raw chapter JSON text.

        Returns:
            str: Repaired text, or original text if unchanged.
        """
        repaired = text
        mutated = False

        new_text = self._COLON_EQUALS_PATTERN.sub(r"\1", repaired)
        if new_text != repaired:
            logger.warning("Detected '\":=\"' in chapter JSON; removed redundant '=' automatically")
            repaired = new_text
            mutated = True

        repaired, escaped = self._escape_in_string_controls(repaired)
        if escaped:
            logger.warning("Detected unescaped control characters in chapter JSON strings; converted to escaped sequences")
            mutated = True

        repaired, balanced = self._balance_brackets(repaired)
        if balanced:
            logger.warning("Detected unbalanced chapter JSON brackets; auto-completed/removed abnormal brackets")
            mutated = True

        repaired, commas_fixed = self._fix_missing_commas(repaired)
        if commas_fixed:
            logger.warning("Detected missing commas between chapter JSON objects/arrays; auto-filled")
            mutated = True

        return repaired if mutated else text

    def _escape_in_string_controls(self, text: str) -> Tuple[str, bool]:
        """
        Replace raw newlines/tabs/control characters in string literals with JSON-safe escapes.
        """
        if not text:
            return text, False

        result: List[str] = []
        in_string = False
        escaped = False
        mutated = False
        control_map = {"\n": "\\n", "\r": "\\n", "\t": "\\t"}

        for ch in text:
            if escaped:
                result.append(ch)
                escaped = False
                continue

            if ch == "\\":
                result.append(ch)
                escaped = True
                continue

            if ch == '"':
                result.append(ch)
                in_string = not in_string
                continue

            if in_string and ch in control_map:
                result.append(control_map[ch])
                mutated = True
                continue

            if in_string and ord(ch) < 0x20:
                result.append(f"\\u{ord(ch):04x}")
                mutated = True
                continue

            result.append(ch)

        return "".join(result), mutated

    def _fix_missing_commas(self, text: str) -> Tuple[str, bool]:
        """Automatically insert commas when objects/arrays appear consecutively."""
        if not text:
            return text, False

        chars: List[str] = []
        mutated = False
        in_string = False
        escaped = False
        length = len(text)
        i = 0
        while i < length:
            ch = text[i]
            chars.append(ch)
            if escaped:
                escaped = False
                i += 1
                continue
            if ch == "\\":
                escaped = True
                i += 1
                continue
            if ch == '"':
                in_string = not in_string
                i += 1
                continue
            if not in_string and ch in "}]":
                j = i + 1
                while j < length and text[j] in " \t\r\n":
                    j += 1
                if j < length:
                    next_ch = text[j]
                    if next_ch in "{[":
                        chars.append(",")
                        mutated = True
            i += 1
        return "".join(chars), mutated

    def _balance_brackets(self, text: str) -> Tuple[str, bool]:
        """Try to repair unbalanced brackets caused by extra/missing LLM bracket output."""
        if not text:
            return text, False

        result: List[str] = []
        stack: List[str] = []
        mutated = False
        in_string = False
        escaped = False

        opener_map = {"{": "}", "[": "]"}

        for ch in text:
            if escaped:
                result.append(ch)
                escaped = False
                continue

            if ch == "\\":
                result.append(ch)
                escaped = True
                continue

            if ch == '"':
                result.append(ch)
                in_string = not in_string
                continue

            if in_string:
                result.append(ch)
                continue

            if ch in "{[":
                stack.append(ch)
                result.append(ch)
                continue

            if ch in "}]":
                if stack and ((ch == "}" and stack[-1] == "{") or (ch == "]" and stack[-1] == "[")):
                    stack.pop()
                    result.append(ch)
                else:
                    mutated = True
                continue

            result.append(ch)

        while stack:
            opener = stack.pop()
            result.append(opener_map[opener])
            mutated = True

        return "".join(result), mutated

    def _attempt_json_repair(self, text: str) -> str | None:
        """Use optional json_repair library to further repair complex syntax errors."""
        if not _json_repair_fn:
            return None
        try:
            fixed = _json_repair_fn(text)
        except Exception as exc:  # pragma: no cover - library-level failure
            logger.warning(f"json_repair failed to repair chapter JSON: {exc}")
            return None
        if fixed == text:
            return None
        logger.warning("Chapter JSON syntax was auto-repaired using json_repair")
        return fixed

    def _attempt_llm_structural_repair(
        self,
        chapter: Dict[str, Any],
        validation_errors: List[str],
        raw_text: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Send structurally invalid chapters to LLM fallback repair using the same Report Engine API settings."""
        if not validation_errors:
            return None
        payload = build_chapter_repair_prompt(chapter, validation_errors, raw_text)
        try:
            response = self.llm_client.invoke(
                SYSTEM_PROMPT_CHAPTER_JSON_REPAIR,
                payload,
                temperature=0.0,
                top_p=0.05,
            )
        except Exception as exc:  # pragma: no cover - network/API exceptions are logged only
            logger.error(f"Chapter JSON LLM repair invocation failed: {exc}")
            return None
        if not response:
            return None
        try:
            repaired = self._parse_chapter(response)
        except Exception as exc:
            logger.error(f"Failed to parse chapter JSON after LLM repair: {exc}")
            return None
        logger.warning("Chapter JSON remained invalid after local repair; LLM fallback repair succeeded")
        return repaired

    def _sanitize_chapter_blocks(self, chapter: Dict[str, Any]):
        """
        Fix common structural issues (e.g., overly nested list.items).

        Args:
            chapter: Chapter JSON object; cleaned and normalized in place.
        """

        def walk(blocks: List[Dict[str, Any]] | None):
            """Recursively inspect and repair nested structures so each block is valid."""
            if not isinstance(blocks, list):
                return
            # Filter out invalid non-dict blocks first.
            valid_indices = []
            for idx, block in enumerate(blocks):
                if not isinstance(block, dict):
                    # Try converting string blocks to paragraph.
                    if isinstance(block, str) and block.strip():
                        blocks[idx] = self._as_paragraph_block(block)
                        valid_indices.append(idx)
                        logger.warning("walk: converted string block to paragraph")
                    elif isinstance(block, list):
                        # Try extracting a valid dict from list blocks.
                        for item in block:
                            if isinstance(item, dict):
                                self._ensure_block_type(item)
                                blocks[idx] = item
                                valid_indices.append(idx)
                                logger.warning("walk: extracted dict block from list")
                                break
                        else:
                            logger.warning(f"walk: skipped invalid list block: {block}")
                    else:
                        logger.warning(f"walk: skipped invalid block (type: {type(block).__name__})")
                else:
                    valid_indices.append(idx)

            for idx in valid_indices:
                block = blocks[idx]
                if not isinstance(block, dict):
                    continue
                self._ensure_block_type(block)
                self._sanitize_block_content(block)
                block_type = block.get("type")
                if block_type == "list":
                    # Auto-fix listType to ensure legal values.
                    self._normalize_list_type(block)
                    items = block.get("items")
                    normalized = self._normalize_list_items(items)
                    if normalized:
                        block["items"] = normalized
                    for entry in block.get("items", []):
                        walk(entry)
                elif block_type in {"callout", "blockquote", "engineQuote"}:
                    walk(block.get("blocks"))
                elif block_type == "table":
                    for row in block.get("rows", []):
                        if not isinstance(row, dict):
                            continue
                        cells = row.get("cells") or []
                        for cell in cells:
                            if not isinstance(cell, dict):
                                continue
                            walk(cell.get("blocks"))
                elif block_type == "widget":
                    self._normalize_widget_block(block)
                else:
                    nested = block.get("blocks")
                    if isinstance(nested, list):
                        walk(nested)

        walk(chapter.get("blocks"))

        blocks = chapter.get("blocks")
        if isinstance(blocks, list):
            # Filter non-dict blocks before merge.
            filtered_blocks = [b for b in blocks if isinstance(b, dict)]
            chapter["blocks"] = self._merge_fragment_sequences(filtered_blocks)

    def _ensure_content_density(self, chapter: Dict[str, Any]):
        """
        Validate chapter body density.

        If blocks are missing, no valid non-heading blocks exist, or body characters
        are below threshold, treat as abnormal content and raise ChapterContentError
        to trigger upstream retries.

        Args:
            chapter: Current chapter JSON.

        Raises:
            ChapterContentError: Raised when body block count or character count is below minimum.
        """
        blocks = chapter.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            raise ChapterContentError(
                "Chapter is missing body blocks and cannot produce content",
                chapter=chapter,
                body_characters=0,
                narrative_characters=0,
                non_heading_blocks=0,
            )

        non_heading_blocks = [
            block
            for block in blocks
            if isinstance(block, dict)
            and block.get("type") not in {"heading", "divider", "toc"}
        ]
        valid_block_count = len(non_heading_blocks)
        body_characters = self._count_body_characters(blocks)
        narrative_characters = self._count_narrative_characters(blocks)

        if (
            valid_block_count < self._MIN_NON_HEADING_BLOCKS
            or body_characters < self._MIN_BODY_CHARACTERS
            or narrative_characters < self._MIN_NARRATIVE_CHARACTERS
        ):
            raise ChapterContentError(
                f"{chapter.get('title') or 'This chapter'} has insufficient body content: {valid_block_count} valid blocks, estimated {body_characters} body characters, {narrative_characters} narrative characters",
                chapter=chapter,
                body_characters=body_characters,
                narrative_characters=narrative_characters,
                non_heading_blocks=valid_block_count,
            )

    def _count_body_characters(self, blocks: Any) -> int:
        """
        Recursively count body characters.

        - Ignore non-body types such as heading/divider/widget;
        - Extract nested text from paragraph/list/table/callout structures;
        - Used only for coarse-grained length sanity checks.

        Args:
            blocks: Chapter blocks list or subtree.

        Returns:
            int: Estimated body character count.
        """

        def walk(node: Any) -> int:
            """Recursively walk block tree and return estimated characters, skipping non-body types."""
            if node is None:
                return 0
            if isinstance(node, list):
                return sum(walk(item) for item in node)
            if isinstance(node, str):
                return len(node.strip())
            if not isinstance(node, dict):
                return 0

            block_type = node.get("type")
            if block_type in {"heading", "divider", "toc", "widget"}:
                return 0

            if block_type == "paragraph":
                return self._estimate_paragraph_characters(node)

            if block_type == "list":
                total = 0
                for item in node.get("items", []):
                    total += walk(item)
                return total

            if block_type in {"blockquote", "callout", "engineQuote"}:
                return walk(node.get("blocks"))

            if block_type == "table":
                total = 0
                for row in node.get("rows", []):
                    cells = row.get("cells") or []
                    for cell in cells:
                        total += walk(cell.get("blocks"))
                return total

            nested = node.get("blocks")
            if isinstance(nested, list):
                return walk(nested)

            return len(self._extract_block_text(node).strip())

        return walk(blocks)

    def _count_narrative_characters(self, blocks: Any) -> int:
        """
        Count characters in narrative structures like paragraph/callout/list/blockquote/engineQuote,
        so table/chart-heavy chapters cannot inflate length artificially.
        """

        def walk(node: Any) -> int:
            """Recursively traverse narrative nodes while ignoring charts/toc and other non-body structures."""
            if node is None:
                return 0
            if isinstance(node, list):
                return sum(walk(item) for item in node)
            if isinstance(node, str):
                return len(node.strip())
            if not isinstance(node, dict):
                return 0

            block_type = node.get("type")
            if block_type == "paragraph":
                return self._estimate_paragraph_characters(node)
            if block_type == "list":
                total = 0
                for item in node.get("items", []):
                    total += walk(item)
                return total
            if block_type in {"callout", "blockquote", "engineQuote"}:
                return walk(node.get("blocks"))

            # List items may be anonymous dicts; handle defensively.
            if block_type is None:
                nested = node.get("blocks")
                if isinstance(nested, list):
                    return walk(nested)
            return 0

        return walk(blocks)

    def _estimate_paragraph_characters(self, block: Dict[str, Any]) -> int:
        """Extract paragraph text length for reuse in multiple counters."""
        inlines = block.get("inlines")
        if isinstance(inlines, list):
            total = 0
            for run in inlines:
                if isinstance(run, dict):
                    text = run.get("text")
                    if isinstance(text, str):
                        total += len(text.strip())
            return total
        text_value = block.get("text")
        if isinstance(text_value, str):
            return len(text_value.strip())
        return len(self._extract_block_text(block).strip())

    def _sanitize_block_content(self, block: Dict[str, Any]):
        """Perform type-specific fine-grained fixes, e.g., invalid inline marks in paragraphs."""
        block_type = block.get("type")
        if block_type == "paragraph":
            self._normalize_paragraph_block(block)
        elif block_type == "table":
            self._sanitize_table_block(block)
        elif block_type == "engineQuote":
            self._sanitize_engine_quote_block(block)

    def _sanitize_table_block(self, block: Dict[str, Any]):
        """Ensure table rows/cells are valid and each cell contains at least one block."""
        raw_rows = block.get("rows")
        # First detect nested row-structure issues (single row but nested cells).
        if isinstance(raw_rows, list) and len(raw_rows) == 1:
            first_row = raw_rows[0]
            if isinstance(first_row, dict):
                cells = first_row.get("cells", [])
                # Detect nested structure.
                has_nested = any(
                    isinstance(cell, dict) and "cells" in cell and "blocks" not in cell
                    for cell in cells
                    if isinstance(cell, dict)
                )
                if has_nested:
                    # Repair nested row structure.
                    fixed_rows = self._fix_nested_rows_structure(raw_rows)
                    block["rows"] = fixed_rows
                    return
        # Normal path: use standard normalization.
        rows = self._normalize_table_rows(raw_rows)
        block["rows"] = rows

    def _fix_nested_rows_structure(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fix incorrectly nested table row structures.

        When LLM generates a table with only one row but all data nested in cells,
        this method flattens all cells and reorganizes them into proper multi-row structure.

        Args:
            rows: Original table rows array (expected to have only one row).

        Returns:
            List[Dict]: Repaired multi-row table structure.
        """
        if not rows or len(rows) != 1:
            return self._normalize_table_rows(rows)

        first_row = rows[0]
        original_cells = first_row.get("cells", [])

        # Recursively flatten all nested cells.
        all_cells = self._flatten_all_cells_recursive(original_cells)

        if len(all_cells) <= 1:
            return self._normalize_table_rows(rows)

        # Helper: extract cell text.
        def _get_cell_text(cell: Dict[str, Any]) -> str:
            blocks = cell.get("blocks", [])
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "paragraph":
                    inlines = block.get("inlines", [])
                    for inline in inlines:
                        if isinstance(inline, dict):
                            text = inline.get("text", "")
                            if text:
                                return str(text).strip()
            return ""

        def _is_placeholder_cell(cell: Dict[str, Any]) -> bool:
            """Determine whether a cell is a placeholder."""
            text = _get_cell_text(cell)
            return text in ("--", "-", "—", "——", "", "N/A", "n/a")

        def _is_header_cell(cell: Dict[str, Any]) -> bool:
            """Determine whether a cell looks like a header (usually bold or containing typical header terms)."""
            blocks = cell.get("blocks", [])
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "paragraph":
                    inlines = block.get("inlines", [])
                    for inline in inlines:
                        if isinstance(inline, dict):
                            marks = inline.get("marks", [])
                            if any(isinstance(m, dict) and m.get("type") == "bold" for m in marks):
                                return True
            # Also check common header keywords.
            text = _get_cell_text(cell)
            header_keywords = {
                "time", "date", "name", "type", "status", "quantity", "amount", "ratio", "metric",
                "platform", "channel", "source", "description", "notes", "remark", "index", "id",
                "event", "key", "data", "evidence", "response", "market", "sentiment", "node",
                "dimension", "point", "detail", "tag", "impact", "trend", "weight", "category",
                "information", "content", "style", "preference", "primary", "user", "core", "feature",
                "classification", "scope", "object", "item", "stage", "period", "frequency", "level",
            }
            return any(kw in text for kw in header_keywords) and len(text) <= 20

        # Filter out placeholder cells.
        valid_cells = [c for c in all_cells if not _is_placeholder_cell(c)]

        if len(valid_cells) <= 1:
            return self._normalize_table_rows(rows)

        # Detect header column count: count consecutive header-like cells.
        header_count = 0
        for cell in valid_cells:
            if _is_header_cell(cell):
                header_count += 1
            else:
                break

        # If no header is detected, use heuristic fallback.
        if header_count == 0:
            total = len(valid_cells)
            for possible_cols in [4, 5, 3, 6, 2]:
                if total % possible_cols == 0:
                    header_count = possible_cols
                    break
            else:
                # Try to find the nearest divisible column count.
                for possible_cols in [4, 5, 3, 6, 2]:
                    remainder = total % possible_cols
                    if remainder <= 3:
                        header_count = possible_cols
                        break
                else:
                    # Cannot determine column count; fallback to original data.
                    return self._normalize_table_rows(rows)

        # Compute effective cell count.
        total = len(valid_cells)
        remainder = total % header_count
        if remainder > 0 and remainder <= 3:
            # Truncate trailing extra cells.
            valid_cells = valid_cells[:total - remainder]
        elif remainder > 3:
            # Remainder too large; likely wrong column-count detection.
            return self._normalize_table_rows(rows)

        # Rebuild into multiple rows.
        fixed_rows: List[Dict[str, Any]] = []
        for i in range(0, len(valid_cells), header_count):
            row_cells = valid_cells[i:i + header_count]
            # Mark first row as header.
            if i == 0:
                for cell in row_cells:
                    cell["header"] = True
            fixed_rows.append({"cells": row_cells})

        return fixed_rows if fixed_rows else self._normalize_table_rows(rows)

    def _flatten_all_cells_recursive(self, cells: List[Any]) -> List[Dict[str, Any]]:
        """
        Recursively flatten all nested cell structures.

        Args:
            cells: Cell array that may contain nested structures.

        Returns:
            List[Dict]: Flattened cell array, each cell containing blocks.
        """
        if not cells:
            return []

        flattened: List[Dict[str, Any]] = []

        def _extract_cells(cell_or_list: Any) -> None:
            if not isinstance(cell_or_list, dict):
                if isinstance(cell_or_list, (str, int, float)):
                    flattened.append({"blocks": [self._as_paragraph_block(str(cell_or_list))]})
                return

            # If current object has blocks, it is a valid cell.
            if "blocks" in cell_or_list:
                # Create cell copy and remove nested cells field.
                clean_cell = {
                    k: v for k, v in cell_or_list.items()
                    if k != "cells"
                }
                # Ensure blocks are valid.
                blocks = clean_cell.get("blocks")
                if not isinstance(blocks, list) or not blocks:
                    clean_cell["blocks"] = [self._as_paragraph_block("")]
                flattened.append(clean_cell)

            # If current object has nested cells, recurse.
            nested_cells = cell_or_list.get("cells")
            if isinstance(nested_cells, list):
                for nested_cell in nested_cells:
                    _extract_cells(nested_cell)

        for cell in cells:
            _extract_cells(cell)

        return flattened

    def _sanitize_engine_quote_block(self, block: Dict[str, Any]):
        """engineQuote is for single-agent speech only; only paragraph blocks are allowed and title must match agent name."""
        engine_raw = block.get("engine")
        engine = engine_raw.lower() if isinstance(engine_raw, str) else None
        if engine not in ENGINE_AGENT_TITLES:
            engine = "insight"
        block["engine"] = engine
        block["title"] = ENGINE_AGENT_TITLES[engine]
        allowed_marks = {"bold", "italic"}
        raw_blocks = block.get("blocks")
        candidates = raw_blocks if isinstance(raw_blocks, list) else ([raw_blocks] if raw_blocks else [])
        sanitized_blocks: List[Dict[str, Any]] = []

        for item in candidates:
            if isinstance(item, dict) and item.get("type") == "paragraph":
                para = dict(item)
            else:
                text = self._extract_block_text(item) if isinstance(item, dict) else (item or "")
                para = self._as_paragraph_block(str(text))

            inlines = para.get("inlines")
            if not isinstance(inlines, list) or not inlines:
                inlines = [self._as_inline_run(self._extract_block_text(para))]

            cleaned_inlines: List[Dict[str, Any]] = []
            for run in inlines:
                if isinstance(run, dict):
                    text_val = run.get("text")
                    text_str = text_val if isinstance(text_val, str) else ("" if text_val is None else str(text_val))
                    marks_raw = run.get("marks") if isinstance(run.get("marks"), list) else []
                    marks_filtered: List[Dict[str, Any]] = []
                    for mark in marks_raw:
                        if not isinstance(mark, dict):
                            continue
                        mark_type = mark.get("type")
                        if mark_type in allowed_marks:
                            marks_filtered.append({"type": mark_type})
                    cleaned_inlines.append({"text": text_str, "marks": marks_filtered})
                else:
                    cleaned_inlines.append(self._as_inline_run(str(run)))

            if not cleaned_inlines:
                cleaned_inlines.append(self._as_inline_run(""))
            para["inlines"] = cleaned_inlines
            para["type"] = "paragraph"
            para.pop("blocks", None)
            sanitized_blocks.append(para)

        if not sanitized_blocks:
            sanitized_blocks.append(self._as_paragraph_block(""))
        block["blocks"] = sanitized_blocks

    def _normalize_table_rows(self, rows: Any) -> List[Dict[str, Any]]:
        """Ensure rows is always a list of row objects."""
        if rows is None:
            rows_iterable: List[Any] = []
        elif isinstance(rows, list):
            rows_iterable = rows
        else:
            rows_iterable = [rows]

        normalized_rows: List[Dict[str, Any]] = []
        for row in rows_iterable:
            sanitized_row = self._normalize_table_row(row)
            if sanitized_row:
                normalized_rows.append(sanitized_row)

        if not normalized_rows:
            normalized_rows.append({"cells": [self._build_default_table_cell()]})
        return normalized_rows

    def _normalize_table_row(self, row: Any) -> Dict[str, Any] | None:
        """Normalize different row forms into {'cells': [...]} structure."""
        if row is None:
            return None
        if isinstance(row, dict):
            result = dict(row)
            cells_value = result.get("cells")
        else:
            result = {}
            cells_value = row

        cells = self._normalize_table_cells(cells_value)
        if not cells:
            cells = [self._build_default_table_cell()]
        result["cells"] = cells
        return result

    def _normalize_table_cells(self, cells: Any) -> List[Dict[str, Any]]:
        """Sanitize cells and ensure each cell has non-empty blocks."""
        if cells is None:
            cell_entries: List[Any] = []
        elif isinstance(cells, list):
            cell_entries = cells
        else:
            cell_entries = [cells]

        normalized_cells: List[Dict[str, Any]] = []
        for cell in cell_entries:
            # Detect wrong nested cells structure: has cells but no blocks.
            # Needs flattening into multiple independent cells.
            if isinstance(cell, dict) and "cells" in cell and "blocks" not in cell:
                flattened = self._flatten_all_nested_cells(cell)
                normalized_cells.extend(flattened)
            else:
                sanitized = self._normalize_table_cell(cell)
                if sanitized:
                    normalized_cells.append(sanitized)

        return normalized_cells

    def _flatten_all_nested_cells(self, cell: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Flatten incorrectly nested cells structure and return all flattened cells.

        LLM sometimes generates invalid structures like:
        { "cells": [
            { "blocks": [...] },
            { "cells": [
                { "blocks": [...] },
                { "cells": [...] }
              ]
            }
          ]
        }

        It should be flattened into an independent cells list.
        """
        nested_cells = cell.get("cells")
        if not isinstance(nested_cells, list) or not nested_cells:
            return [{"blocks": [self._as_paragraph_block("")]}]

        result: List[Dict[str, Any]] = []
        for nested in nested_cells:
            if isinstance(nested, dict):
                if "blocks" in nested and "cells" not in nested:
                    # Normal cell, normalize and append directly.
                    sanitized = self._normalize_table_cell(nested)
                    if sanitized:
                        result.append(sanitized)
                elif "cells" in nested and "blocks" not in nested:
                    # Continue recursively flattening nested cells.
                    result.extend(self._flatten_all_nested_cells(nested))
                else:
                    # Other cases: try normalizing.
                    sanitized = self._normalize_table_cell(nested)
                    if sanitized:
                        result.append(sanitized)
            elif isinstance(nested, (str, int, float)):
                result.append({"blocks": [self._as_paragraph_block(str(nested))]})

        return result if result else [{"blocks": [self._as_paragraph_block("")]}]

    def _normalize_table_cell(self, cell: Any) -> Dict[str, Any] | None:
        """Normalize different cell formats into schema-compliant form."""
        if cell is None:
            return {"blocks": [self._as_paragraph_block("")]}

        if isinstance(cell, dict):
            # Detect wrong nested cells structure: has cells but no blocks.
            # Common LLM error: sibling cells are nested into cells array.
            if "cells" in cell and "blocks" not in cell:
                # Flatten nested cells and return first valid cell.
                # Remaining nested cells are handled in _normalize_table_cells.
                return self._flatten_nested_cell(cell)

            normalized = dict(cell)
            blocks = self._coerce_cell_blocks(normalized.get("blocks"), normalized)
        elif isinstance(cell, list):
            normalized = {}
            blocks = self._coerce_cell_blocks(cell, None)
        elif isinstance(cell, (str, int, float)):
            normalized = {}
            blocks = [self._as_paragraph_block(str(cell))]
        else:
            normalized = {}
            blocks = [self._as_paragraph_block(str(cell))]

        normalized["blocks"] = blocks or [self._as_paragraph_block("")]
        return normalized

    def _flatten_nested_cell(self, cell: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten incorrectly nested cell structures.

        LLM sometimes generates invalid structures like:
        { "cells": [ { "blocks": [...] }, { "cells": [...] } ] }

        It should return the first valid cell content.
        """
        nested_cells = cell.get("cells")
        if not isinstance(nested_cells, list) or not nested_cells:
            # No valid nested content, return empty cell.
            return {"blocks": [self._as_paragraph_block("")]}

        # Recursively locate first valid cell containing blocks.
        for nested in nested_cells:
            if isinstance(nested, dict):
                if "blocks" in nested:
                    # Found valid cell, normalize recursively.
                    return self._normalize_table_cell(nested)
                elif "cells" in nested:
                    # Continue recursive flattening.
                    result = self._flatten_nested_cell(nested)
                    if result:
                        return result

        # No valid content found; try extracting text from first nested element.
        first_nested = nested_cells[0]
        if isinstance(first_nested, dict):
            text = self._extract_block_text(first_nested)
            return {"blocks": [self._as_paragraph_block(text or "")]}

        return {"blocks": [self._as_paragraph_block("")]}

    def _coerce_cell_blocks(
        self, blocks: Any, source: Dict[str, Any] | None
    ) -> List[Dict[str, Any]]:
        """Coerce cell.blocks into a valid block array."""
        if isinstance(blocks, list):
            entries = blocks
        elif blocks is None:
            entries = []
        else:
            entries = [blocks]

        normalized_blocks: List[Dict[str, Any]] = []
        for entry in entries:
            if isinstance(entry, dict):
                normalized_blocks.append(entry)
            elif isinstance(entry, list):
                normalized_blocks.extend(self._coerce_cell_blocks(entry, None))
            elif isinstance(entry, (str, int, float)):
                normalized_blocks.append(self._as_paragraph_block(str(entry)))
            elif entry is None:
                continue
            else:
                normalized_blocks.append(self._as_paragraph_block(str(entry)))

        if normalized_blocks:
            return normalized_blocks

        text_hint = ""
        if isinstance(source, dict):
            text_hint = self._extract_block_text(source).strip()
        return [self._as_paragraph_block(text_hint or "--")]

    def _build_default_table_cell(self) -> Dict[str, Any]:
        """Create a minimal renderable empty cell."""
        return {"blocks": [self._as_paragraph_block("--")]}

    def _normalize_paragraph_block(self, block: Dict[str, Any]):
        """Normalize paragraph inlines and remove invalid marks."""
        inlines = block.get("inlines")
        normalized_runs: List[Dict[str, Any]] = []
        if isinstance(inlines, list) and inlines:
            for run in inlines:
                normalized_runs.extend(self._coerce_inline_run(run))
        else:
            normalized_runs = [self._as_inline_run(self._extract_block_text(block))]
        if not normalized_runs:
            normalized_runs = [self._as_inline_run("")]
        block["inlines"] = self._strip_inline_artifacts(normalized_runs)

    def _strip_inline_artifacts(self, inlines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove JSON sentinel artifacts accidentally emitted by LLM to prevent rendering junk like `{\"type\": \"\"}`."""
        cleaned: List[Dict[str, Any]] = []
        for run in inlines or []:
            if not isinstance(run, dict):
                continue
            text = run.get("text")
            if isinstance(text, str):
                stripped = text.strip()
                if stripped.startswith("{") and stripped.endswith("}"):
                    try:
                        payload = json.loads(stripped)
                    except json.JSONDecodeError:
                        payload = None
                    if isinstance(payload, dict) and set(payload.keys()).issubset({"type", "value"}):
                        continue
            cleaned.append(run)
        return cleaned or [self._as_inline_run("")]

    def _merge_fragment_sequences(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge sentence fragments split by LLM to avoid many isolated <p> tags in HTML."""
        if not isinstance(blocks, list):
            return blocks

        merged: List[Dict[str, Any]] = []
        fragment_buffer: List[Dict[str, Any]] = []

        def flush_buffer():
            """Flush current fragment buffer into merged list; combine into one paragraph when needed."""
            nonlocal fragment_buffer
            if not fragment_buffer:
                return
            if len(fragment_buffer) == 1:
                merged.append(fragment_buffer[0])
            else:
                merged.append(self._combine_paragraph_fragments(fragment_buffer))
            fragment_buffer = []

        for block in blocks:
            # Type guard: skip abnormal non-dict blocks to avoid AttributeError.
            if not isinstance(block, dict):
                # Try converting non-dict blocks to paragraph.
                if isinstance(block, str) and block.strip():
                    converted = self._as_paragraph_block(block)
                    logger.warning(f"Detected non-dict block (string), converted to paragraph: {block[:50]}...")
                    merged.append(converted)
                elif isinstance(block, list):
                    # List-type block may be malformed LLM output; try extracting valid content.
                    logger.warning(f"Detected list-type block; attempting to extract valid content: {block}")
                    for item in block:
                        if isinstance(item, dict):
                            self._ensure_block_type(item)
                            merged.append(self._merge_nested_fragments(item))
                        elif isinstance(item, str) and item.strip():
                            merged.append(self._as_paragraph_block(item))
                else:
                    logger.warning(f"Skipped invalid block (type: {type(block).__name__}): {block}")
                continue
            if self._is_paragraph_fragment(block):
                fragment_buffer.append(block)
                continue
            flush_buffer()
            merged.append(self._merge_nested_fragments(block))

        flush_buffer()
        return merged

    def _merge_nested_fragments(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge fragments inside nested structures (callout/blockquote/engineQuote/list/table)."""
        # Type guard: ensure block is dict.
        if not isinstance(block, dict):
            # Try converting non-dict blocks to paragraph.
            if isinstance(block, str) and block.strip():
                logger.warning("_merge_nested_fragments received string input; converted to paragraph")
                return self._as_paragraph_block(block)
            elif isinstance(block, list):
                # Try extracting first valid dict from list.
                for item in block:
                    if isinstance(item, dict):
                        self._ensure_block_type(item)
                        return self._merge_nested_fragments(item)
                logger.warning("_merge_nested_fragments received invalid list; returning empty paragraph")
                return self._as_paragraph_block("")
            else:
                logger.warning(f"_merge_nested_fragments received invalid type ({type(block).__name__}); returning empty paragraph")
                return self._as_paragraph_block("")

        block_type = block.get("type")
        if block_type in {"callout", "blockquote", "engineQuote"}:
            nested = block.get("blocks")
            if isinstance(nested, list):
                block["blocks"] = self._merge_fragment_sequences(nested)
        elif block_type == "list":
            items = block.get("items")
            if isinstance(items, list):
                for entry in items:
                    if isinstance(entry, list):
                        merged_entry = self._merge_fragment_sequences(entry)
                        entry[:] = merged_entry
        elif block_type == "table":
            for row in block.get("rows", []):
                if not isinstance(row, dict):
                    continue
                cells = row.get("cells") or []
                for cell in cells:
                    if not isinstance(cell, dict):
                        continue
                    nested_blocks = cell.get("blocks")
                    if isinstance(nested_blocks, list):
                        cell["blocks"] = self._merge_fragment_sequences(nested_blocks)
        return block

    def _combine_paragraph_fragments(self, fragments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multiple sentence fragments into a single paragraph block."""
        template = dict(fragments[0])
        combined_inlines: List[Dict[str, Any]] = []
        for fragment in fragments:
            runs = fragment.get("inlines")
            if isinstance(runs, list) and runs:
                combined_inlines.extend(runs)
            else:
                fallback_text = self._extract_block_text(fragment)
                combined_inlines.append(self._as_inline_run(fallback_text))
        if not combined_inlines:
            combined_inlines.append(self._as_inline_run(""))
        template["inlines"] = combined_inlines
        return template

    def _is_paragraph_fragment(self, block: Dict[str, Any]) -> bool:
        """Determine whether paragraph is a short fragment incorrectly split by LLM."""
        if not isinstance(block, dict) or block.get("type") != "paragraph":
            return False
        inlines = block.get("inlines")
        text = ""
        has_marks = False
        if isinstance(inlines, list) and inlines:
            parts: List[str] = []
            for run in inlines:
                if not isinstance(run, dict):
                    continue
                parts.append(str(run.get("text") or ""))
                marks = run.get("marks")
                if isinstance(marks, list) and any(marks):
                    has_marks = True
            text = "".join(parts)
        else:
            text = self._extract_block_text(block)
        stripped = (text or "").strip()
        if not stripped:
            return True
        if has_marks:
            return False
        if "\n" in stripped:
            return False

        short_limit = self._PARAGRAPH_FRAGMENT_MAX_CHARS
        long_limit = getattr(
            self,
            "_PARAGRAPH_FRAGMENT_NO_TERMINATOR_MAX_CHARS",
            short_limit * 3,
        )

        if stripped[-1] in self._TERMINATION_PUNCTUATION:
            return len(stripped) <= short_limit

        if len(stripped) > long_limit:
            return False
        return True

    def _coerce_inline_run(self, run: Any) -> List[Dict[str, Any]]:
        """Normalize arbitrary inline forms into valid runs."""
        if isinstance(run, dict):
            normalized_run = dict(run)
            text = normalized_run.get("text")
            if not isinstance(text, str):
                text = "" if text is None else str(text)
            marks = normalized_run.get("marks")
            sanitized_marks, extra_text = self._sanitize_inline_marks(marks)
            normalized_run["marks"] = sanitized_marks
            normalized_run["text"] = (text or "") + extra_text
            return [normalized_run]
        if isinstance(run, str):
            return [self._as_inline_run(run)]
        if isinstance(run, (int, float)):
            return [self._as_inline_run(str(run))]
        if isinstance(run, list):
            normalized: List[Dict[str, Any]] = []
            for item in run:
                normalized.extend(self._coerce_inline_run(item))
            return normalized
        return [self._as_inline_run("" if run is None else str(run))]

    def _sanitize_inline_marks(self, marks: Any) -> Tuple[List[Dict[str, Any]], str]:
        """Filter invalid marks and convert break-like controls to text."""
        text_suffix = ""
        if marks is None:
            return [], text_suffix
        mark_list = marks if isinstance(marks, list) else [marks]
        sanitized: List[Dict[str, Any]] = []
        for mark in mark_list:
            normalized_mark, extra_text = self._normalize_inline_mark(mark)
            if normalized_mark:
                sanitized.append(normalized_mark)
            if extra_text:
                text_suffix += extra_text
        return sanitized, text_suffix

    def _normalize_inline_mark(self, mark: Any) -> Tuple[Dict[str, Any] | None, str]:
        """Apply compatibility mapping for one mark, or convert to text when needed."""
        if not isinstance(mark, dict):
            return None, ""
        canonical_type = self._canonical_inline_mark_type(mark.get("type"))
        if canonical_type == self._LINE_BREAK_SENTINEL:
            return None, "\n"
        if canonical_type in ALLOWED_INLINE_MARKS:
            normalized = dict(mark)
            normalized["type"] = canonical_type
            return normalized, ""
        return None, ""

    def _canonical_inline_mark_type(self, mark_type: Any) -> str | None:
        """Map mark type to schema-supported values."""
        if not isinstance(mark_type, str):
            return None
        normalized = mark_type.strip()
        if not normalized:
            return None
        lowered = normalized.lower()
        if lowered in {"break", "linebreak", "br"}:
            return self._LINE_BREAK_SENTINEL
        return self._INLINE_MARK_ALIASES.get(lowered, lowered)

    def _extract_block_text(self, block: Dict[str, Any]) -> str:
        """Prefer extracting fallback text from text/content-like fields."""
        for key in ("text", "content", "value", "title"):
            value = block.get(key)
            if isinstance(value, str):
                return value
            if value is not None:
                return str(value)
        return ""

    # Valid listType values.
    _ALLOWED_LIST_TYPES = {"ordered", "bullet", "task"}
    # Alias mapping for listType.
    _LIST_TYPE_ALIASES = {
        "unordered": "bullet",
        "ul": "bullet",
        "ol": "ordered",
        "numbered": "ordered",
        "checkbox": "task",
        "check": "task",
        "todo": "task",
    }

    def _normalize_list_type(self, block: Dict[str, Any]):
        """
        Ensure list block listType is legal.

        If listType is missing or invalid, auto-fix to bullet.
        """
        list_type = block.get("listType")
        if list_type in self._ALLOWED_LIST_TYPES:
            return
        # Try alias mapping.
        if isinstance(list_type, str):
            lowered = list_type.strip().lower()
            if lowered in self._LIST_TYPE_ALIASES:
                block["listType"] = self._LIST_TYPE_ALIASES[lowered]
                logger.warning(f"Mapped listType '{list_type}' to '{block['listType']}'")
                return
            if lowered in self._ALLOWED_LIST_TYPES:
                block["listType"] = lowered
                return
        # Unrecognized value; default to bullet.
        logger.warning(f"Detected invalid listType: {list_type}; auto-fixed to bullet")
        block["listType"] = "bullet"

    def _normalize_list_items(self, items: Any) -> List[List[Dict[str, Any]]]:
        """Ensure list block items use [[block, block], ...] structure."""
        if not isinstance(items, list):
            return []
        normalized: List[List[Dict[str, Any]]] = []
        for item in items:
            normalized.extend(self._coerce_list_item(item))
        return [entry for entry in normalized if entry]

    def _coerce_list_item(self, item: Any) -> List[List[Dict[str, Any]]]:
        """Coerce various nested item forms into arrays of blocks."""
        result: List[List[Dict[str, Any]]] = []
        if isinstance(item, dict):
            self._ensure_block_type(item)
            result.append([item])
            return result
        if isinstance(item, list):
            dicts = [elem for elem in item if isinstance(elem, dict)]
            if dicts:
                for elem in dicts:
                    self._ensure_block_type(elem)
                result.append(dicts)
            for elem in item:
                if isinstance(elem, list):
                    result.extend(self._coerce_list_item(elem))
                elif isinstance(elem, dict):
                    continue
                elif isinstance(elem, str):
                    result.append([self._as_paragraph_block(elem)])
                elif isinstance(elem, (int, float)):
                    result.append([self._as_paragraph_block(str(elem))])
        elif isinstance(item, str):
            result.append([self._as_paragraph_block(item)])
        elif isinstance(item, (int, float)):
            result.append([self._as_paragraph_block(str(item))])
        return result

    def _normalize_widget_block(self, block: Dict[str, Any]):
        """Ensure widget has top-level data or dataRef."""
        has_data = block.get("data") is not None or block.get("dataRef") is not None
        if has_data:
            return
        props = block.get("props")
        if isinstance(props, dict) and "data" in props:
            block["data"] = props.pop("data")
            return
        block["data"] = {"labels": [], "datasets": []}

    def _ensure_block_type(self, block: Dict[str, Any]):
        """Downgrade to paragraph if block lacks a valid type."""
        block_type = block.get("type")
        if isinstance(block_type, str) and block_type in ALLOWED_BLOCK_TYPES:
            return
        text = ""
        for key in ("text", "content", "title"):
            value = block.get(key)
            if isinstance(value, str) and value.strip():
                text = value.strip()
                break
        if not text:
            try:
                text = json.dumps(block, ensure_ascii=False)
            except Exception:
                text = str(block)
        block.clear()
        block["type"] = "paragraph"
        block["inlines"] = [self._as_inline_run(text)]

    @staticmethod
    def _as_paragraph_block(text: str) -> Dict[str, Any]:
        """Quickly wrap a string as a paragraph block for uniform handling."""
        return {
            "type": "paragraph",
            "inlines": [ChapterGenerationNode._as_inline_run(text)],
        }

    @staticmethod
    def _as_inline_run(text: str) -> Dict[str, Any]:
        """Construct a basic inline run and ensure marks field exists."""
        return {"text": text or "", "marks": []}

    @staticmethod
    def _parse_with_candidates(payloads: List[str]) -> Dict[str, Any]:
        """Try multiple payloads in order until one parses successfully."""
        last_exc: json.JSONDecodeError | None = None
        for payload in payloads:
            try:
                return json.loads(payload)
            except json.JSONDecodeError as exc:
                last_exc = exc
        assert last_exc is not None
        raise last_exc


__all__ = [
    "ChapterGenerationNode",
    "ChapterJsonParseError",
    "ChapterContentError",
    "ChapterValidationError",
]
