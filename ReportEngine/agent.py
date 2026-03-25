"""
Report Agent main class.

This module chains together template selection, layout design, chapter generation,
IR binding, and HTML rendering into a single orchestration layer for the Report Engine.
Core responsibilities:
1. Manage input data and state, coordinating three analysis engines, forum logs, and templates;
2. Drive the pipeline in node order: template selection → layout generation → word budgeting → chapter writing → binding and rendering;
3. Handle error fallbacks, streaming event dispatch, output manifests, and final artifact persistence.
"""

import json
import os
from copy import deepcopy
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable, Tuple

from loguru import logger

from .core import (
    ChapterStorage,
    DocumentComposer,
    TemplateSection,
    parse_template_sections,
)
from .ir import IRValidator
from .llms import LLMClient
from .nodes import (
    TemplateSelectionNode,
    ChapterGenerationNode,
    ChapterJsonParseError,
    ChapterContentError,
    ChapterValidationError,
    DocumentLayoutNode,
    WordBudgetNode,
)
from .renderers import HTMLRenderer
from .state import ReportState
from .utils.config import settings, Settings


class StageOutputFormatError(ValueError):
    """Controlled exception raised when a stage's output structure does not match expectations."""


class FileCountBaseline:
    """
    File-count baseline manager.

    This utility is used to:
    - Record Markdown counts exported by Insight/Media/Query engines at task start;
    - Quickly determine whether new reports have landed during later polling;
    - Provide the Flask layer with a readiness signal for input availability.
    """
    
    def __init__(self):
        """
        Prefer loading an existing baseline snapshot at initialization.

        If `logs/report_baseline.json` does not exist, an empty snapshot is
        created implicitly so `initialize_baseline` can persist the real
        baseline on first run.
        """
        self.baseline_file = 'logs/report_baseline.json'
        self.baseline_data = self._load_baseline()
    
    def _load_baseline(self) -> Dict[str, int]:
        """
        Load baseline data.

        - Parse JSON directly when the snapshot file exists;
        - Catch all load exceptions and return an empty dict to keep caller logic simple.
        """
        try:
            if os.path.exists(self.baseline_file):
                with open(self.baseline_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.exception(f"Failed to load baseline data: {e}")
        return {}
    
    def _save_baseline(self):
        """
        Persist the current baseline to disk.

        Uses `ensure_ascii=False` and pretty indentation for readability;
        creates the target directory automatically when missing.
        """
        try:
            os.makedirs(os.path.dirname(self.baseline_file), exist_ok=True)
            with open(self.baseline_file, 'w', encoding='utf-8') as f:
                json.dump(self.baseline_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.exception(f"Failed to save baseline data: {e}")
    
    def initialize_baseline(self, directories: Dict[str, str]) -> Dict[str, int]:
        """
        Initialize file-count baselines.

        Walk through each engine directory, count `.md` files, and persist the
        counts as the initial baseline. `check_new_files` compares against this.
        """
        current_counts = {}
        
        for engine, directory in directories.items():
            if os.path.exists(directory):
                md_files = [f for f in os.listdir(directory) if f.endswith('.md')]
                current_counts[engine] = len(md_files)
            else:
                current_counts[engine] = 0
        
        # Save baseline data
        self.baseline_data = current_counts.copy()
        self._save_baseline()
        
        logger.info(f"File-count baseline initialized: {current_counts}")
        return current_counts
    
    def check_new_files(self, directories: Dict[str, str]) -> Dict[str, Any]:
        """
        Check whether new files exist.

        Compare current directory counts against baselines:
        - Count newly added files and determine whether all engines are ready;
        - Return detailed counts and missing lists for Web-layer user prompts.
        """
        current_counts = {}
        new_files_found = {}
        all_have_new = True
        
        for engine, directory in directories.items():
            if os.path.exists(directory):
                md_files = [f for f in os.listdir(directory) if f.endswith('.md')]
                current_counts[engine] = len(md_files)
                baseline_count = self.baseline_data.get(engine, 0)
                
                if current_counts[engine] > baseline_count:
                    new_files_found[engine] = current_counts[engine] - baseline_count
                else:
                    new_files_found[engine] = 0
                    all_have_new = False
            else:
                current_counts[engine] = 0
                new_files_found[engine] = 0
                all_have_new = False
        
        return {
            'ready': all_have_new,
            'baseline_counts': self.baseline_data,
            'current_counts': current_counts,
            'new_files_found': new_files_found,
            'missing_engines': [engine for engine, count in new_files_found.items() if count == 0]
        }
    
    def get_latest_files(self, directories: Dict[str, str]) -> Dict[str, str]:
        """
        Get the latest file in each directory.

        Uses `os.path.getmtime` to find the most recently written Markdown file,
        ensuring generation always uses the newest three-engine reports.
        """
        latest_files = {}
        
        for engine, directory in directories.items():
            if os.path.exists(directory):
                md_files = [f for f in os.listdir(directory) if f.endswith('.md')]
                if md_files:
                    latest_file = max(md_files, key=lambda x: os.path.getmtime(os.path.join(directory, x)))
                    latest_files[engine] = os.path.join(directory, latest_file)
        
        return latest_files


class ReportAgent:
    """
    Main Report Agent class.

    Integrates:
    - The LLM client and four reasoning nodes on top of it;
    - The output pipeline including chapter storage, IR composition, and rendering;
    - State management, logging, I/O validation, and persistence.
    """
    _CONTENT_SPARSE_MIN_ATTEMPTS = 3
    _CONTENT_SPARSE_WARNING_TEXT = "This chapter may be too short because the LLM output was sparse. Re-running the program may help if needed."
    _STRUCTURAL_RETRY_ATTEMPTS = 2
    
    def __init__(self, config: Optional[Settings] = None):
        """
        Initialize Report Agent.
        
        Args:
            config: Configuration object; auto-loaded if omitted.
        
        Steps overview:
            1. Parse config and wire core components like logging/LLM/rendering;
            2. Build four reasoning nodes (template, layout, word budget, chapter);
            3. Initialize file baselines and chapter output directories;
            4. Build a serializable state container for external services.
        """
        # Load configuration
        self.config = config or settings
        
        # Initialize file baseline manager
        self.file_baseline = FileCountBaseline()
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize LLM client
        self.llm_client = self._initialize_llm()
        self.json_rescue_clients = self._initialize_rescue_llms()
        
        # Initialize chapter-level storage/validation/rendering components
        self.chapter_storage = ChapterStorage(self.config.CHAPTER_OUTPUT_DIR)
        self.document_composer = DocumentComposer()
        self.validator = IRValidator()
        self.renderer = HTMLRenderer()
        
        # Initialize nodes
        self._initialize_nodes()
        
        # Initialize file-count baseline
        self._initialize_file_baseline()
        
        # State
        self.state = ReportState()
        
        # Ensure output directories exist
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.config.DOCUMENT_IR_OUTPUT_DIR, exist_ok=True)
        
        logger.info("Report Agent initialized")
        logger.info(f"Using LLM: {self.llm_client.get_model_info()}")
        
    def _setup_logging(self):
        """
                Configure logging.

                - Ensure the log directory exists;
                - Use a dedicated loguru sink for Report Engine logs to avoid mixing with other subsystems;
                - [Fix] Configure real-time writes and disable buffering for live frontend logs;
                - [Fix] Prevent duplicate handler registration.
        """
                # Ensure log directory exists
        log_dir = os.path.dirname(self.config.LOG_FILE)
        os.makedirs(log_dir, exist_ok=True)

        def _exclude_other_engines(record):
            """
            Exclude logs produced by other engines (Insight/Media/Query/Forum),
            while keeping all other logs.

            Prefer path-based matching; fall back to module-name matching when needed.
            """
            excluded_keywords = ("InsightEngine", "MediaEngine", "QueryEngine", "ForumEngine")
            try:
                file_path = record["file"].path
                if any(keyword in file_path for keyword in excluded_keywords):
                    return False
            except Exception:
                pass

            try:
                module_name = record.get("module", "")
                if isinstance(module_name, str):
                    lowered = module_name.lower()
                    if any(keyword.lower() in lowered for keyword in excluded_keywords):
                        return False
            except Exception:
                pass

            return True

        # [Fix] Check whether this file handler already exists to avoid duplicates.
        # loguru deduplicates automatically, but explicit checks are safer.
        log_file_path = str(Path(self.config.LOG_FILE).resolve())

        # Check existing handlers
        handler_exists = False
        for handler_id, handler_config in logger._core.handlers.items():
            if hasattr(handler_config, 'sink'):
                sink = handler_config.sink
                # Check whether this is a file sink with the same path
                if hasattr(sink, '_name') and sink._name == log_file_path:
                    handler_exists = True
                    logger.debug(f"Log handler already exists, skipping add: {log_file_path}")
                    break

        if not handler_exists:
            # [Fix] Create a dedicated logger with real-time write settings.
            # - enqueue=False: disable async queue, write immediately
            # - buffering=1: line buffering, flush each line
            # - level="DEBUG": record all log levels
            # - encoding="utf-8": explicit UTF-8 encoding
            # - mode="a": append mode, preserve history
            handler_id = logger.add(
                self.config.LOG_FILE,
                level="DEBUG",
                enqueue=False,      # Disable async queue, synchronous write
                buffering=1,        # Line-buffered, flush each line
                serialize=False,    # Plain text format, not JSON
                encoding="utf-8",   # Explicit UTF-8 encoding
                mode="a",           # Append mode
                filter=_exclude_other_engines # Exclude logs from the four engines
            )
            logger.debug(f"Added log handler (ID: {handler_id}): {self.config.LOG_FILE}")

        # [Fix] Validate log file writability
        try:
            with open(self.config.LOG_FILE, 'a', encoding='utf-8') as f:
                f.write('')  # Write empty content to verify permissions
                f.flush()    # Flush immediately
        except Exception as e:
            logger.error(f"Log file is not writable: {self.config.LOG_FILE}, error: {e}")
            raise
        
    def _initialize_file_baseline(self):
        """
        Initialize file-count baseline.

        Pass Insight/Media/Query directories to `FileCountBaseline` to generate
        one-time reference values; then detect new reports via incremental deltas.
        """
        directories = {
            'insight': 'insight_engine_streamlit_reports',
            'media': 'media_engine_streamlit_reports',
            'query': 'query_engine_streamlit_reports'
        }
        self.file_baseline.initialize_baseline(directories)
    
    def _initialize_llm(self) -> LLMClient:
        """
        Initialize the LLM client.

        Build a shared `LLMClient` instance from API key/model/base URL in
        configuration, used as a common inference entrypoint by all nodes.
        """
        return LLMClient(
            api_key=self.config.REPORT_ENGINE_API_KEY,
            model_name=self.config.REPORT_ENGINE_MODEL_NAME,
            base_url=self.config.REPORT_ENGINE_BASE_URL,
        )

    def _initialize_rescue_llms(self) -> List[Tuple[str, LLMClient]]:
        """
        Initialize fallback LLM clients for cross-engine chapter repair.

        The order follows "Report -> Forum -> Insight -> Media".
        Missing configs are skipped automatically.
        """
        clients: List[Tuple[str, LLMClient]] = []
        if self.llm_client:
            clients.append(("report_engine", self.llm_client))
        fallback_specs = [
            (
                "forum_engine",
                self.config.FORUM_HOST_API_KEY,
                self.config.FORUM_HOST_MODEL_NAME,
                self.config.FORUM_HOST_BASE_URL,
            ),
            (
                "insight_engine",
                self.config.INSIGHT_ENGINE_API_KEY,
                self.config.INSIGHT_ENGINE_MODEL_NAME,
                self.config.INSIGHT_ENGINE_BASE_URL,
            ),
            (
                "media_engine",
                self.config.MEDIA_ENGINE_API_KEY,
                self.config.MEDIA_ENGINE_MODEL_NAME,
                self.config.MEDIA_ENGINE_BASE_URL,
            ),
        ]
        for label, api_key, model_name, base_url in fallback_specs:
            if not api_key or not model_name:
                continue
            try:
                client = LLMClient(api_key=api_key, model_name=model_name, base_url=base_url)
            except Exception as exc:
                logger.warning(f"Failed to initialize {label} LLM, skipping this repair channel: {exc}")
                continue
            clients.append((label, client))
        return clients
    
    def _initialize_nodes(self):
        """
        Initialize processing nodes.

        Instantiate template selection, document layout, word budgeting, and
        chapter generation nodes in order; chapter generation additionally
        depends on the IR validator and chapter storage.
        """
        self.template_selection_node = TemplateSelectionNode(
            self.llm_client,
            self.config.TEMPLATE_DIR
        )
        self.document_layout_node = DocumentLayoutNode(self.llm_client)
        self.word_budget_node = WordBudgetNode(self.llm_client)
        self.chapter_generation_node = ChapterGenerationNode(
            self.llm_client,
            self.validator,
            self.chapter_storage,
            fallback_llm_clients=self.json_rescue_clients,
            error_log_dir=self.config.JSON_ERROR_LOG_DIR,
        )
    
    def generate_report(self, query: str, reports: List[Any], forum_logs: str = "",
                        custom_template: str = "", save_report: bool = True,
                        stream_handler: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> str:
        """
        Generate a full report (chapter JSON -> IR -> HTML).

        Main stages:
            1. Normalize three-engine reports plus forum logs and emit streaming events;
            2. Template selection -> template slicing -> document layout -> word budgeting;
            3. Generate each chapter with LLM based on word targets, retrying on parse failures;
            4. Compose chapters into Document IR and render via HTML renderer;
            5. Optionally persist HTML/IR/state and return path metadata.

        Args:
            query: Final report topic or question.
            reports: Raw outputs from Query/Media/Insight and similar engines; can be strings or richer objects.
            forum_logs: Forum/collaboration logs for multi-party discussion context.
            custom_template: User-specified Markdown template; auto-selected when empty.
            save_report: Whether to persist HTML, IR, and state to disk after generation.
            stream_handler: Optional streaming event callback for real-time UI updates.

        Returns:
            dict: Contains `html_content` plus HTML/IR/state paths; returns only HTML string when `save_report=False`.

        Raises:
            Exception: Raised when any child node or rendering stage fails.
        """
        start_time = datetime.now()
        report_id = f"report-{uuid4().hex[:8]}"
        self.state.task_id = report_id
        self.state.query = query
        self.state.metadata.query = query
        self.state.mark_processing()

        normalized_reports = self._normalize_reports(reports)

        def emit(event_type: str, payload: Dict[str, Any]):
            """Event dispatcher for Report Engine streaming channels with safe error isolation."""
            if not stream_handler:
                return
            try:
                stream_handler(event_type, payload)
            except Exception as callback_error:  # pragma: no cover - logging only
                logger.warning(f"Streaming event callback failed: {callback_error}")

        logger.info(f"Starting report generation {report_id}: {query}")
        logger.info(f"Input data - report count: {len(reports)}, forum log length: {len(str(forum_logs))}")
        emit('stage', {'stage': 'agent_start', 'report_id': report_id, 'query': query})

        try:
            template_result = self._select_template(query, reports, forum_logs, custom_template)
            template_result = self._ensure_mapping(
                template_result,
                "Template selection result",
                expected_keys=["template_name", "template_content"],
            )
            self.state.metadata.template_used = template_result.get('template_name', '')
            emit('stage', {
                'stage': 'template_selected',
                'template': template_result.get('template_name'),
                'reason': template_result.get('selection_reason')
            })
            emit('progress', {'progress': 10, 'message': 'Template selection completed'})
            sections = self._slice_template(template_result.get('template_content', ''))
            if not sections:
                raise ValueError("No sections could be parsed from the template. Please check template content.")
            emit('stage', {'stage': 'template_sliced', 'section_count': len(sections)})

            template_text = template_result.get('template_content', '')
            template_overview = self._build_template_overview(template_text, sections)
            # Design global title, TOC, and visual theme from template skeleton + three-engine content
            layout_design = self._run_stage_with_retry(
                "Document design",
                lambda: self.document_layout_node.run(
                    sections,
                    template_text,
                    normalized_reports,
                    forum_logs,
                    query,
                    template_overview,
                ),
                # `toc` has been replaced by `tocPlan`; select/validate by latest schema
                expected_keys=["title", "hero", "tocPlan", "tocTitle"],
            )
            emit('stage', {
                'stage': 'layout_designed',
                'title': layout_design.get('title'),
                'toc': layout_design.get('tocTitle')
            })
            emit('progress', {'progress': 15, 'message': 'Document title/TOC design completed'})
            # Use the generated design draft to budget chapter lengths and emphases across the book
            word_plan = self._run_stage_with_retry(
                "Chapter word-budget planning",
                lambda: self.word_budget_node.run(
                    sections,
                    layout_design,
                    normalized_reports,
                    forum_logs,
                    query,
                    template_overview,
                ),
                expected_keys=["chapters", "totalWords", "globalGuidelines"],
                postprocess=self._normalize_word_plan,
            )
            emit('stage', {
                'stage': 'word_plan_ready',
                'chapter_targets': len(word_plan.get('chapters', []))
            })
            emit('progress', {'progress': 20, 'message': 'Chapter word-budget plan generated'})
            # Track per-chapter word targets/emphases for downstream chapter LLM calls
            chapter_targets = {
                entry.get("chapterId"): entry
                for entry in word_plan.get("chapters", [])
                if entry.get("chapterId")
            }

            generation_context = self._build_generation_context(
                query,
                normalized_reports,
                forum_logs,
                template_result,
                layout_design,
                chapter_targets,
                word_plan,
                template_overview,
            )
            # Global metadata required by IR/rendering, including title/theme/TOC/word plan from design
            manifest_meta = {
                "query": query,
                "title": layout_design.get("title") or (f"{query} - Public Opinion Insight Report" if query else template_result.get("template_name")),
                "subtitle": layout_design.get("subtitle"),
                "tagline": layout_design.get("tagline"),
                "templateName": template_result.get("template_name"),
                "selectionReason": template_result.get("selection_reason"),
                "themeTokens": generation_context.get("theme_tokens", {}),
                "toc": {
                    "depth": 3,
                    "autoNumbering": True,
                    "title": layout_design.get("tocTitle") or "Table of Contents",
                },
                "hero": layout_design.get("hero"),
                "layoutNotes": layout_design.get("layoutNotes"),
                "wordPlan": {
                    "totalWords": word_plan.get("totalWords"),
                    "globalGuidelines": word_plan.get("globalGuidelines"),
                },
                "templateOverview": template_overview,
            }
            if layout_design.get("themeTokens"):
                manifest_meta["themeTokens"] = layout_design["themeTokens"]
            if layout_design.get("tocPlan"):
                manifest_meta["toc"]["customEntries"] = layout_design["tocPlan"]
            # Initialize chapter output directory and write manifest for streaming persistence
            run_dir = self.chapter_storage.start_session(report_id, manifest_meta)
            self._persist_planning_artifacts(run_dir, layout_design, word_plan, template_overview)
            emit('stage', {'stage': 'storage_ready', 'run_dir': str(run_dir)})

            chapters = []
            chapter_max_attempts = max(
                self._CONTENT_SPARSE_MIN_ATTEMPTS, self.config.CHAPTER_JSON_MAX_ATTEMPTS
            )
            total_chapters = len(sections)  # Total number of chapters
            completed_chapters = 0  # Number of completed chapters

            for section in sections:
                logger.info(f"Generating chapter: {section.title}")
                emit('chapter_status', {
                    'chapterId': section.chapter_id,
                    'title': section.title,
                    'status': 'running'
                })
                # Chapter streaming callback: forward LLM deltas to SSE for real-time frontend rendering
                def chunk_callback(delta: str, meta: Dict[str, Any], section_ref: TemplateSection = section):
                    """
                    Streaming callback for chapter content.

                    Args:
                        delta: Latest incremental output text from LLM.
                        meta: Chapter metadata returned by node; used as fallback.
                        section_ref: Defaults to current chapter for robust identification when metadata is missing.
                    """
                    emit('chapter_chunk', {
                        'chapterId': meta.get('chapterId') or section_ref.chapter_id,
                        'title': meta.get('title') or section_ref.title,
                        'delta': delta
                    })

                chapter_payload: Dict[str, Any] | None = None
                attempt = 1
                best_sparse_candidate: Dict[str, Any] | None = None
                best_sparse_score = -1
                fallback_used = False
                while attempt <= chapter_max_attempts:
                    try:
                        chapter_payload = self.chapter_generation_node.run(
                            section,
                            generation_context,
                            run_dir,
                            stream_callback=chunk_callback
                        )
                        break
                    except (ChapterJsonParseError, ChapterContentError, ChapterValidationError) as structured_error:
                        if isinstance(structured_error, ChapterContentError):
                            error_kind = "content_sparse"
                            readable_label = "content sparsity"
                        elif isinstance(structured_error, ChapterValidationError):
                            error_kind = "validation"
                            readable_label = "structural validation failed"
                        else:
                            error_kind = "json_parse"
                            readable_label = "JSON parsing failed"
                        if isinstance(structured_error, ChapterContentError):
                            candidate = getattr(structured_error, "chapter_payload", None)
                            candidate_score = getattr(structured_error, "body_characters", 0) or 0
                            if isinstance(candidate, dict) and candidate_score >= 0:
                                if candidate_score > best_sparse_score:
                                    best_sparse_candidate = deepcopy(candidate)
                                    best_sparse_score = candidate_score
                        will_fallback = (
                            isinstance(structured_error, ChapterContentError)
                            and attempt >= chapter_max_attempts
                            and attempt >= self._CONTENT_SPARSE_MIN_ATTEMPTS
                            and best_sparse_candidate is not None
                        )
                        logger.warning(
                            "Chapter {title} {label} (attempt {attempt}/{total}): {error}",
                            title=section.title,
                            label=readable_label,
                            attempt=attempt,
                            total=chapter_max_attempts,
                            error=structured_error,
                        )
                        status_value = 'retrying' if attempt < chapter_max_attempts or will_fallback else 'error'
                        status_payload = {
                            'chapterId': section.chapter_id,
                            'title': section.title,
                            'status': status_value,
                            'attempt': attempt,
                            'error': str(structured_error),
                            'reason': error_kind,
                        }
                        if isinstance(structured_error, ChapterValidationError):
                            validation_errors = getattr(structured_error, "errors", None)
                            if validation_errors:
                                status_payload['errors'] = validation_errors
                        if will_fallback:
                            status_payload['warning'] = 'content_sparse_fallback_pending'
                        emit('chapter_status', status_payload)
                        if will_fallback:
                            logger.warning(
                                "Chapter {title} reached max attempts; preserving the longest version (about {score} chars) as fallback output",
                                title=section.title,
                                score=best_sparse_score,
                            )
                            chapter_payload = self._finalize_sparse_chapter(best_sparse_candidate)
                            fallback_used = True
                            break
                        if attempt >= chapter_max_attempts:
                            raise
                        attempt += 1
                        continue
                    except (AttributeError, TypeError, KeyError, IndexError, ValueError, json.JSONDecodeError) as structure_error:
                        # Capture runtime errors caused by malformed JSON structures and wrap as retryable.
                        # Includes:
                        # - AttributeError: e.g., list.get() call failure
                        # - TypeError: type mismatch
                        # - KeyError: missing dict key
                        # - IndexError: list index out of range
                        # - ValueError: invalid value (e.g., empty LLM output, missing required fields)
                        # - json.JSONDecodeError: JSON parse failure not caught internally
                        error_type = type(structure_error).__name__
                        logger.warning(
                            "Chapter {title} encountered {error_type} during generation (attempt {attempt}/{total}); retrying: {error}",
                            title=section.title,
                            error_type=error_type,
                            attempt=attempt,
                            total=chapter_max_attempts,
                            error=structure_error,
                        )
                        emit('chapter_status', {
                            'chapterId': section.chapter_id,
                            'title': section.title,
                            'status': 'retrying' if attempt < chapter_max_attempts else 'error',
                            'attempt': attempt,
                            'error': str(structure_error),
                            'reason': 'structure_error',
                            'error_type': error_type
                        })
                        if attempt >= chapter_max_attempts:
                            # Max retries reached; wrap and raise as ChapterJsonParseError.
                            raise ChapterJsonParseError(
                                f"Chapter {section.title} could not be generated after {chapter_max_attempts} attempts due to {error_type}: {structure_error}"
                            ) from structure_error
                        attempt += 1
                        continue
                    except Exception as chapter_error:
                        if not self._should_retry_inappropriate_content_error(chapter_error):
                            raise
                        logger.warning(
                            "Chapter {title} triggered content safety restrictions (attempt {attempt}/{total}); preparing to retry: {error}",
                            title=section.title,
                            attempt=attempt,
                            total=chapter_max_attempts,
                            error=chapter_error,
                        )
                        emit('chapter_status', {
                            'chapterId': section.chapter_id,
                            'title': section.title,
                            'status': 'retrying' if attempt < chapter_max_attempts else 'error',
                            'attempt': attempt,
                            'error': str(chapter_error),
                            'reason': 'content_filter'
                        })
                        if attempt >= chapter_max_attempts:
                            raise
                        attempt += 1
                        continue
                if chapter_payload is None:
                    raise ChapterJsonParseError(
                        f"Chapter JSON for {section.title} could not be parsed after {chapter_max_attempts} attempts"
                    )
                chapters.append(chapter_payload)
                completed_chapters += 1  # Update completed chapter count
                # Compute progress: 20% + 80% * (completed chapters / total chapters), rounded
                chapter_progress = 20 + round(80 * completed_chapters / total_chapters)
                emit('progress', {
                    'progress': chapter_progress,
                    'message': f'Chapter {completed_chapters}/{total_chapters} completed'
                })
                completion_status = {
                    'chapterId': section.chapter_id,
                    'title': section.title,
                    'status': 'completed',
                    'attempt': attempt,
                }
                if fallback_used:
                    completion_status['warning'] = 'content_sparse_fallback'
                    completion_status['warningMessage'] = self._CONTENT_SPARSE_WARNING_TEXT
                emit('chapter_status', completion_status)

            document_ir = self.document_composer.build_document(
                report_id,
                manifest_meta,
                chapters
            )
            emit('stage', {'stage': 'chapters_compiled', 'chapter_count': len(chapters)})
            html_report = self.renderer.render(document_ir)
            emit('stage', {'stage': 'html_rendered', 'html_length': len(html_report)})

            self.state.html_content = html_report
            self.state.mark_completed()

            saved_files = {}
            if save_report:
                saved_files = self._save_report(html_report, document_ir, report_id)
                emit('stage', {'stage': 'report_saved', 'files': saved_files})

            generation_time = (datetime.now() - start_time).total_seconds()
            self.state.metadata.generation_time = generation_time
            logger.info(f"Report generation completed, elapsed: {generation_time:.2f} s")
            emit('metrics', {'generation_seconds': generation_time})
            return {
                'html_content': html_report,
                'report_id': report_id,
                **saved_files
            }

        except Exception as e:
            self.state.mark_failed(str(e))
            logger.exception(f"Error occurred during report generation: {str(e)}")
            emit('error', {'stage': 'agent_failed', 'message': str(e)})
            raise
    
    def _select_template(self, query: str, reports: List[Any], forum_logs: str, custom_template: str):
        """
        Select a report template.

        Prefer a user-provided template. Otherwise, send query, three-engine
        reports, and forum logs to `TemplateSelectionNode`, letting the LLM
        return the best-fit template name/content/reason and record it in state.

        Args:
            query: Report topic used to focus the prompt by industry/event.
            reports: Multi-source report text to help the LLM infer structural complexity.
            forum_logs: Forum/collaboration text for background context.
            custom_template: Custom Markdown template from CLI/frontend; used directly when non-empty.

        Returns:
            dict: Structured result with `template_name`, `template_content`, and `selection_reason`.
        """
        logger.info("Selecting report template...")
        
        # Use user-provided custom template directly when present
        if custom_template:
            logger.info("Using user-provided custom template")
            return {
                'template_name': 'custom',
                'template_content': custom_template,
                'selection_reason': 'User-specified custom template'
            }
        
        template_input = {
            'query': query,
            'reports': reports,
            'forum_logs': forum_logs
        }
        
        try:
            template_result = self.template_selection_node.run(template_input)
            
            # Update state
            self.state.metadata.template_used = template_result['template_name']
            
            logger.info(f"Selected template: {template_result['template_name']}")
            logger.info(f"Selection reason: {template_result['selection_reason']}")
            
            return template_result
        except Exception as e:
            logger.error(f"Template selection failed, using default template: {str(e)}")
            # Use fallback template directly
            fallback_template = {
                'template_name': 'Public Hotspot Event Analysis Report Template',
                'template_content': self._get_fallback_template_content(),
                'selection_reason': 'Template selection failed; used default public hotspot event analysis template'
            }
            self.state.metadata.template_used = fallback_template['template_name']
            return fallback_template
    
    def _slice_template(self, template_markdown: str) -> List[TemplateSection]:
        """
        Slice template into a chapter list, with fallback when empty.

        Delegates to `parse_template_sections` to parse Markdown headings/numbering
        into `TemplateSection` objects, ensuring stable chapter IDs downstream.
        Falls back to a built-in minimal skeleton on malformed templates.

        Args:
            template_markdown: Full template Markdown text.

        Returns:
            list[TemplateSection]: Parsed chapter sequence, or a one-chapter fallback on failure.
        """
        sections = parse_template_sections(template_markdown)
        if sections:
            return sections
        logger.warning("No chapters parsed from template; using default chapter skeleton")
        fallback = TemplateSection(
            title="1.0 Comprehensive Analysis",
            slug="section-1-0",
            order=10,
            depth=1,
            raw_title="1.0 Comprehensive Analysis",
            number="1.0",
            chapter_id="S1",
            outline=["1.1 Summary", "1.2 Data Highlights", "1.3 Risk Notes"],
        )
        return [fallback]

    def _build_generation_context(
        self,
        query: str,
        reports: Dict[str, str],
        forum_logs: str,
        template_result: Dict[str, Any],
        layout_design: Dict[str, Any],
        chapter_directives: Dict[str, Any],
        word_plan: Dict[str, Any],
        template_overview: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build shared context required for chapter generation.

        Consolidates template metadata, layout design, theme colors, word plan,
        forum logs, and more into a single `generation_context` reused by each
        chapter call to keep tone and visual constraints consistent.

        Args:
            query: User query text.
            reports: Normalized report map for query/media/insight.
            forum_logs: Discussion logs from three engines.
            template_result: Template metadata from template node.
            layout_design: Title/TOC/theme design output from layout node.
            chapter_directives: Chapter-level directives from word-budget node.
            word_plan: Raw word-budget output with global constraints.
            template_overview: Template-slice chapter skeleton summary.

        Returns:
            dict: Full context for LLM chapter generation, including theme/layout/constraints.
        """
        # Prefer design-draft theme tokens; otherwise use defaults
        theme_tokens = (
            layout_design.get("themeTokens")
            if layout_design else None
        ) or self._default_theme_tokens()

        return {
            "query": query,
            "template_name": template_result.get("template_name"),
            "reports": reports,
            "forum_logs": self._stringify(forum_logs),
            "theme_tokens": theme_tokens,
            "style_directives": {
                "tone": "analytical",
                "audience": "executive",
                "language": "zh-CN",
            },
            "data_bundles": [],
            "max_tokens": min(self.config.MAX_CONTENT_LENGTH, 6000),
            "layout": layout_design or {},
            "template_overview": template_overview or {},
            "chapter_directives": chapter_directives or {},
            "word_plan": word_plan or {},
        }

    def _normalize_reports(self, reports: List[Any]) -> Dict[str, str]:
        """
        Normalize reports from different sources into strings.

        Expected order is Query/Media/Insight. Engine outputs may be dicts or
        custom objects, so `_stringify` is used for robust conversion.

        Args:
            reports: Report list of arbitrary types, allowing missing or shuffled entries.

        Returns:
            dict: Map with string fields `query_engine`/`media_engine`/`insight_engine`.
        """
        keys = ["query_engine", "media_engine", "insight_engine"]
        normalized: Dict[str, str] = {}
        for idx, key in enumerate(keys):
            value = reports[idx] if idx < len(reports) else ""
            normalized[key] = self._stringify(value)
        return normalized

    def _should_retry_inappropriate_content_error(self, error: Exception) -> bool:
        """
        Determine whether an LLM exception was caused by content safety filtering.

        When vendor errors contain certain keywords, chapter generation is
        allowed to retry to bypass occasional moderation-triggered failures.

        Args:
            error: Exception raised by LLM client.

        Returns:
            bool: True if moderation-related keywords are matched; otherwise False.
        """
        message = str(error) if error else ""
        if not message:
            return False
        normalized = message.lower()
        keywords = [
            "inappropriate content",
            "content violation",
            "content moderation",
            "model-studio/error-code",
        ]
        return any(keyword in normalized for keyword in keywords)

    def _run_stage_with_retry(
        self,
        stage_name: str,
        fn: Callable[[], Any],
        expected_keys: Optional[List[str]] = None,
        postprocess: Optional[Callable[[Dict[str, Any], str], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run a single LLM stage with limited retries on structural output issues.

        This method only handles structure-related errors for local recovery/retry,
        avoiding a full agent restart.
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, self._STRUCTURAL_RETRY_ATTEMPTS + 1):
            try:
                raw_result = fn()
                result = self._ensure_mapping(raw_result, stage_name, expected_keys)
                if postprocess:
                    result = postprocess(result, stage_name)
                return result
            except StageOutputFormatError as exc:
                last_error = exc
                logger.warning(
                    "{stage} output structure is invalid (attempt {attempt}/{total}); trying repair or retry: {error}",
                    stage=stage_name,
                    attempt=attempt,
                    total=self._STRUCTURAL_RETRY_ATTEMPTS,
                    error=exc,
                )
                if attempt >= self._STRUCTURAL_RETRY_ATTEMPTS:
                    break
        raise last_error  # type: ignore[misc]

    def _ensure_mapping(
        self,
        value: Any,
        context: str,
        expected_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Ensure stage output is a dict; if it is a list, extract the best matching element.
        """
        if isinstance(value, dict):
            return value

        if isinstance(value, list):
            candidates = [item for item in value if isinstance(item, dict)]
            if candidates:
                best = candidates[0]
                if expected_keys:
                    candidates.sort(
                        key=lambda item: sum(1 for key in expected_keys if key in item),
                        reverse=True,
                    )
                    best = candidates[0]
                logger.warning(
                    "{context} returned a list; extracted the element with the most expected keys to continue",
                    context=context,
                )
                return best
            raise StageOutputFormatError(f"{context} returned a list but has no usable object element")

        if value is None:
            raise StageOutputFormatError(f"{context} returned an empty result")

        raise StageOutputFormatError(
            f"{context} returned type {type(value).__name__}, expected a dict"
        )

    def _normalize_word_plan(self, word_plan: Dict[str, Any], stage_name: str) -> Dict[str, Any]:
        """
        Normalize word-plan output to keep chapters/globalGuidelines/totalWords type-safe.
        """
        raw_chapters = word_plan.get("chapters", [])
        if isinstance(raw_chapters, dict):
            chapters_iterable = raw_chapters.values()
        elif isinstance(raw_chapters, list):
            chapters_iterable = raw_chapters
        else:
            chapters_iterable = []

        normalized: List[Dict[str, Any]] = []
        for idx, entry in enumerate(chapters_iterable):
            if isinstance(entry, dict):
                normalized.append(entry)
                continue
            if isinstance(entry, list):
                dict_candidate = next((item for item in entry if isinstance(item, dict)), None)
                if dict_candidate:
                    logger.warning(
                        "{stage} chapter entry #{idx} is a list; extracted the first object for downstream flow",
                        stage=stage_name,
                        idx=idx + 1,
                    )
                    normalized.append(dict_candidate)
                    continue
            logger.warning(
                "{stage} skipping unparseable chapter entry #{idx} (type: {type_name})",
                stage=stage_name,
                idx=idx + 1,
                type_name=type(entry).__name__,
            )

        if not normalized:
            raise StageOutputFormatError(f"{stage_name} is missing valid chapter plans and cannot continue")

        word_plan["chapters"] = normalized

        guidelines = word_plan.get("globalGuidelines")
        if not isinstance(guidelines, list):
            if guidelines is None or guidelines == "":
                word_plan["globalGuidelines"] = []
            else:
                logger.warning(
                    "{stage} globalGuidelines has an invalid type; converted to a list wrapper",
                    stage=stage_name,
                )
                word_plan["globalGuidelines"] = [guidelines]

        if not isinstance(word_plan.get("totalWords"), (int, float)):
            logger.warning(
                "{stage} totalWords has an invalid type; using default value 10000",
                stage=stage_name,
            )
            word_plan["totalWords"] = 10000

        return word_plan

    def _finalize_sparse_chapter(self, chapter: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build sparse-content fallback chapter: copy the original payload and insert a notice paragraph.
        """
        safe_chapter = deepcopy(chapter or {})
        if not isinstance(safe_chapter, dict):
            safe_chapter = {}
        self._ensure_sparse_warning_block(safe_chapter)
        return safe_chapter

    def _ensure_sparse_warning_block(self, chapter: Dict[str, Any]) -> None:
        """
        Insert a warning paragraph after the chapter heading to indicate low content volume.
        """
        warning_block = {
            "type": "paragraph",
            "inlines": [
                {
                    "text": self._CONTENT_SPARSE_WARNING_TEXT,
                    "marks": [{"type": "italic"}],
                }
            ],
            "meta": {"role": "content-sparse-warning"},
        }
        blocks = chapter.get("blocks")
        if isinstance(blocks, list) and blocks:
            inserted = False
            for idx, block in enumerate(blocks):
                if isinstance(block, dict) and block.get("type") == "heading":
                    blocks.insert(idx + 1, warning_block)
                    inserted = True
                    break
            if not inserted:
                blocks.insert(0, warning_block)
        else:
            chapter["blocks"] = [warning_block]
        meta = chapter.get("meta")
        if isinstance(meta, dict):
            meta["contentSparseWarning"] = True
        else:
            chapter["meta"] = {"contentSparseWarning": True}

    def _stringify(self, value: Any) -> str:
        """
        Safely convert an object to string.

        - dict/list are serialized to pretty JSON for prompt consumption;
        - other types use `str()`, and `None` becomes empty string to avoid propagation.

        Args:
            value: Any Python object.

        Returns:
            str: String representation suitable for prompts/logs.
        """
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, ensure_ascii=False, indent=2)
            except Exception:
                return str(value)
        return str(value)

    def _default_theme_tokens(self) -> Dict[str, Any]:
        """
        Build default theme tokens shared by renderer and LLM.

        Used when layout node does not return custom colors, keeping report style consistent.

        Returns:
            dict: Theme dictionary containing colors, fonts, spacing, and boolean flags.
        """
        return {
            "colors": {
                "bg": "#f8f9fa",
                "text": "#212529",
                "primary": "#007bff",
                "secondary": "#6c757d",
                "card": "#ffffff",
                "border": "#dee2e6",
                "accent1": "#17a2b8",
                "accent2": "#28a745",
                "accent3": "#ffc107",
                "accent4": "#dc3545",
            },
            "fonts": {
                "body": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif",
                "heading": "'Source Han Sans SC', 'PingFang SC', 'Microsoft YaHei', sans-serif",
            },
            "spacing": {"container": "1200px", "gutter": "24px"},
            "vars": {
                "header_sticky": True,
                "toc_depth": 3,
                "enable_dark_mode": True,
            },
        }

    def _build_template_overview(
        self,
        template_markdown: str,
        sections: List[TemplateSection],
    ) -> Dict[str, Any]:
        """
        Extract template title and chapter skeleton for shared use in design/word budgeting.

        Also records helper fields like chapter ID/slug/order to keep nodes aligned.

        Args:
            template_markdown: Raw template text for global title extraction.
            sections: `TemplateSection` list used as chapter skeleton.

        Returns:
            dict: Overview structure containing template title and chapter metadata.
        """
        fallback_title = sections[0].title if sections else ""
        overview = {
            "title": self._extract_template_title(template_markdown, fallback_title),
            "chapters": [],
        }
        for section in sections:
            overview["chapters"].append(
                {
                    "chapterId": section.chapter_id,
                    "title": section.title,
                    "rawTitle": section.raw_title,
                    "number": section.number,
                    "slug": section.slug,
                    "order": section.order,
                    "depth": section.depth,
                    "outline": section.outline,
                }
            )
        return overview

    @staticmethod
    def _extract_template_title(template_markdown: str, fallback: str = "") -> str:
        """
        Try to extract the first title from Markdown.

        Prefer the first `#` heading; if the first non-empty line is body text,
        fall back to that line or to caller-provided fallback.

        Args:
            template_markdown: Raw template text.
            fallback: Fallback title used when no explicit heading exists.

        Returns:
            str: Parsed title text.
        """
        for line in template_markdown.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
            if stripped:
                fallback = fallback or stripped
        return fallback or "Intelligent Public Opinion Analysis Report"
    
    def _get_fallback_template_content(self) -> str:
        """
        Get fallback template content.

        This Markdown template is used when template directory access is
        unavailable or LLM selection fails, ensuring downstream structured output.
        """
        return """# Public Hotspot Event Analysis Report

    ## Executive Summary
    This report provides a comprehensive analysis of the current public hotspot event, integrating viewpoints and data from multiple information sources.

    ## Event Overview
    ### Basic Information
    - Event Nature: {event_nature}
    - Occurrence Time: {event_time}
    - Scope Involved: {event_scope}

    ## Public Opinion Trend Analysis
    ### Overall Trend
    {sentiment_analysis}

    ### Distribution of Major Viewpoints
    {opinion_distribution}

    ## Media Coverage Analysis
    ### Mainstream Media Stance
    {media_analysis}

    ### Coverage Focus
    {report_focus}

    ## Social Impact Assessment
    ### Direct Impact
    {direct_impact}

    ### Potential Impact
    {potential_impact}

    ## Response Recommendations
    ### Immediate Measures
    {immediate_actions}

    ### Long-Term Strategy
    {long_term_strategy}

    ## Conclusion and Outlook
    {conclusion}

    ---
    *Report Type: Public Hotspot Event Analysis*
    *Generated At: {generation_time}*
    """
    
    def _save_report(self, html_content: str, document_ir: Dict[str, Any], report_id: str) -> Dict[str, Any]:
        """
        Save HTML and IR to files and return path metadata.

        Generates readable filenames based on query and timestamp, and writes
        runtime `ReportState` to JSON for downstream troubleshooting/resume.

        Args:
            html_content: Rendered HTML body.
            document_ir: Structured Document IR data.
            report_id: Current task ID for unique naming.

        Returns:
            dict: Absolute and relative path info for HTML/IR/state files.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_safe = "".join(
            c for c in self.state.metadata.query if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        query_safe = query_safe.replace(" ", "_")[:30] or "report"

        html_filename = f"final_report_{query_safe}_{timestamp}.html"
        html_path = Path(self.config.OUTPUT_DIR) / html_filename
        html_path.write_text(html_content, encoding="utf-8")
        html_abs = str(html_path.resolve())
        html_rel = os.path.relpath(html_abs, os.getcwd())

        ir_path = self._save_document_ir(document_ir, query_safe, timestamp)
        ir_abs = str(ir_path.resolve())
        ir_rel = os.path.relpath(ir_abs, os.getcwd())

        state_filename = f"report_state_{query_safe}_{timestamp}.json"
        state_path = Path(self.config.OUTPUT_DIR) / state_filename
        self.state.save_to_file(str(state_path))
        state_abs = str(state_path.resolve())
        state_rel = os.path.relpath(state_abs, os.getcwd())

        logger.info(f"HTML report saved: {html_path}")
        logger.info(f"Document IR saved: {ir_path}")
        logger.info(f"State saved to: {state_path}")
        
        return {
            'report_filename': html_filename,
            'report_filepath': html_abs,
            'report_relative_path': html_rel,
            'ir_filename': ir_path.name,
            'ir_filepath': ir_abs,
            'ir_relative_path': ir_rel,
            'state_filename': state_filename,
            'state_filepath': state_abs,
            'state_relative_path': state_rel,
        }

    def _save_document_ir(self, document_ir: Dict[str, Any], query_safe: str, timestamp: str) -> Path:
        """
        Save full IR to a dedicated directory.

        `Document IR` is persisted separately from HTML to simplify render-diff
        debugging and allow re-render/export without re-running the LLM.

        Args:
            document_ir: Full report IR structure.
            query_safe: Sanitized query phrase used in filename.
            timestamp: Run timestamp for uniqueness.

        Returns:
            Path: Path to the saved IR file.
        """
        filename = f"report_ir_{query_safe}_{timestamp}.json"
        ir_path = Path(self.config.DOCUMENT_IR_OUTPUT_DIR) / filename
        ir_path.write_text(
            json.dumps(document_ir, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return ir_path
    
    def _persist_planning_artifacts(
        self,
        run_dir: Path,
        layout_design: Dict[str, Any],
        word_plan: Dict[str, Any],
        template_overview: Dict[str, Any],
    ):
        """
        Persist document layout, word plan, and template overview as JSON artifacts.

        These intermediate files (`document_layout`/`word_plan`/`template_overview`)
        help debugging and retrospection by exposing how title/TOC/theme and
        word allocation were decided for later manual adjustment.

        Args:
            run_dir: Chapter output root directory.
            layout_design: Raw output from document layout node.
            word_plan: Output from word-budget node.
            template_overview: Template overview JSON.
        """
        artifacts = {
            "document_layout": layout_design,
            "word_plan": word_plan,
            "template_overview": template_overview,
        }
        for name, payload in artifacts.items():
            if not payload:
                continue
            path = run_dir / f"{name}.json"
            try:
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as exc:
                logger.warning(f"Failed to write {name}: {exc}")
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get progress summary and return a serializable state dict for API queries."""
        return self.state.to_dict()
    
    def load_state(self, filepath: str):
        """Load state from file and replace current state for checkpoint recovery."""
        self.state = ReportState.load_from_file(filepath)
        logger.info(f"State loaded from {filepath}")
    
    def save_state(self, filepath: str):
        """Save state to file, typically for post-run analysis and backup."""
        self.state.save_to_file(filepath)
        logger.info(f"State saved to {filepath}")
    
    def check_input_files(self, insight_dir: str, media_dir: str, query_dir: str, forum_log_path: str) -> Dict[str, Any]:
        """
        Check whether input files are ready (based on file-count increments).
        
        Args:
            insight_dir: InsightEngine report directory
            media_dir: MediaEngine report directory
            query_dir: QueryEngine report directory
            forum_log_path: Forum log file path
            
        Returns:
            Result dictionary including file counts, missing lists, latest file paths, etc.
        """
        # Check file-count changes in each report directory
        directories = {
            'insight': insight_dir,
            'media': media_dir,
            'query': query_dir
        }
        
        # Check for new files using the file baseline manager
        check_result = self.file_baseline.check_new_files(directories)
        
        # Check forum log
        forum_ready = os.path.exists(forum_log_path)
        
        # Build result object
        result = {
            'ready': check_result['ready'] and forum_ready,
            'baseline_counts': check_result['baseline_counts'],
            'current_counts': check_result['current_counts'],
            'new_files_found': check_result['new_files_found'],
            'missing_files': [],
            'files_found': [],
            'latest_files': {}
        }
        
        # Build detailed info
        for engine, new_count in check_result['new_files_found'].items():
            current_count = check_result['current_counts'][engine]
            baseline_count = check_result['baseline_counts'].get(engine, 0)
            
            if new_count > 0:
                result['files_found'].append(f"{engine}: {current_count} files ({new_count} new)")
            else:
                result['missing_files'].append(f"{engine}: {current_count} files (baseline {baseline_count}, no new files)")
        
        # Check forum log
        if forum_ready:
            result['files_found'].append(f"forum: {os.path.basename(forum_log_path)}")
        else:
            result['missing_files'].append("forum: log file does not exist")
        
        # Get latest file paths (for actual report generation)
        if result['ready']:
            result['latest_files'] = self.file_baseline.get_latest_files(directories)
            if forum_ready:
                result['latest_files']['forum'] = forum_log_path
        
        return result
    
    def load_input_files(self, file_paths: Dict[str, str]) -> Dict[str, Any]:
        """
        Load input file content.
        
        Args:
            file_paths: File path dictionary
            
        Returns:
            Loaded content dictionary containing `reports` list and `forum_logs` string
        """
        content = {
            'reports': [],
            'forum_logs': ''
        }
        
        # Load report files
        engines = ['query', 'media', 'insight']
        for engine in engines:
            if engine in file_paths:
                try:
                    with open(file_paths[engine], 'r', encoding='utf-8') as f:
                        report_content = f.read()
                    content['reports'].append(report_content)
                    logger.info(f"Loaded {engine} report: {len(report_content)} characters")
                except Exception as e:
                    logger.exception(f"Failed to load {engine} report: {str(e)}")
                    content['reports'].append("")
        
        # Load forum logs
        if 'forum' in file_paths:
            try:
                with open(file_paths['forum'], 'r', encoding='utf-8') as f:
                    content['forum_logs'] = f.read()
                logger.info(f"Loaded forum logs: {len(content['forum_logs'])} characters")
            except Exception as e:
                logger.exception(f"Failed to load forum logs: {str(e)}")
        
        return content


def create_agent(config_file: Optional[str] = None) -> ReportAgent:
    """
    Convenience function to create a Report Agent instance.
    
    Args:
        config_file: Configuration file path
        
    Returns:
        ReportAgent instance

    `Settings` is currently driven by environment variables.
    `config_file` is retained for future extension.
    """
    
    config = Settings() # Initialize from environment-variable-backed settings
    return ReportAgent(config)
