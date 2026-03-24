"""
Report Agenttranslated。

translated、translated、translated、IRtranslatedHTMLtranslated
translated，translatedReport Enginetranslated。translated：
1. translated，translated、translated；
2. translated→translated→translated→translated→translated；
3. translated、translated、translated。
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
    """translated。"""


class FileCountBaseline:
    """
    translated。

    translated：
    - translated Insight/Media/Query translated Markdown translated；
    - translated；
    - translated Flask translated“translated”translated。
    """
    
    def __init__(self):
        """
        translated。

        translated `logs/report_baseline.json` translated，
        translated `initialize_baseline` translated。
        """
        self.baseline_file = 'logs/report_baseline.json'
        self.baseline_data = self._load_baseline()
    
    def _load_baseline(self) -> Dict[str, int]:
        """
        translated。

        - translatedJSON；
        - translated，translated。
        """
        try:
            if os.path.exists(self.baseline_file):
                with open(self.baseline_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.exception(f"translated: {e}")
        return {}
    
    def _save_baseline(self):
        """
        translated。

        translated `ensure_ascii=False` + translated，translated；
        translated。
        """
        try:
            os.makedirs(os.path.dirname(self.baseline_file), exist_ok=True)
            with open(self.baseline_file, 'w', encoding='utf-8') as f:
                json.dump(self.baseline_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.exception(f"translated: {e}")
    
    def initialize_baseline(self, directories: Dict[str, str]) -> Dict[str, int]:
        """
        translated。

        translated `.md` translated，translated
        translated。translated `check_new_files` translated。
        """
        current_counts = {}
        
        for engine, directory in directories.items():
            if os.path.exists(directory):
                md_files = [f for f in os.listdir(directory) if f.endswith('.md')]
                current_counts[engine] = len(md_files)
            else:
                current_counts[engine] = 0
        
        # translated
        self.baseline_data = current_counts.copy()
        self._save_baseline()
        
        logger.info(f"translated: {current_counts}")
        return current_counts
    
    def check_new_files(self, directories: Dict[str, str]) -> Dict[str, Any]:
        """
        translated。

        translated：
        - translated，translated；
        - translated、translated，translated Web translated。
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
        translated。

        translated `os.path.getmtime` translated Markdown，
        translated。
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
    Report Agenttranslated。

    translated：
    - LLMtranslated；
    - translated、IRtranslated、translated；
    - translated、translated、translated。
    """
    _CONTENT_SPARSE_MIN_ATTEMPTS = 3
    _CONTENT_SPARSE_WARNING_TEXT = "translatedLLMtranslated，translated。"
    _STRUCTURAL_RETRY_ATTEMPTS = 2
    
    def __init__(self, config: Optional[Settings] = None):
        """
        translatedReport Agent。
        
        Args:
            config: translated，translated
        
        translated：
            1. translated/LLM/translated；
            2. translated（translated、translated、translated、translated）；
            3. translated；
            4. translated，translated。
        """
        # translated
        self.config = config or settings
        
        # translated
        self.file_baseline = FileCountBaseline()
        
        # translated
        self._setup_logging()
        
        # translatedLLMtranslated
        self.llm_client = self._initialize_llm()
        self.json_rescue_clients = self._initialize_rescue_llms()
        
        # translated/translated/translated
        self.chapter_storage = ChapterStorage(self.config.CHAPTER_OUTPUT_DIR)
        self.document_composer = DocumentComposer()
        self.validator = IRValidator()
        self.renderer = HTMLRenderer()
        
        # translated
        self._initialize_nodes()
        
        # translated
        self._initialize_file_baseline()
        
        # translated
        self.state = ReportState()
        
        # translated
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.config.DOCUMENT_IR_OUTPUT_DIR, exist_ok=True)
        
        logger.info("Report Agenttranslated")
        logger.info(f"translatedLLM: {self.llm_client.get_model_info()}")
        
    def _setup_logging(self):
        """
        translated。

        - translated；
        - translated loguru sink translated Report Engine translated log translated，
          translated。
        - 【translated】translated，translated，translated
        - 【translated】translatedhandler
        """
        # translated
        log_dir = os.path.dirname(self.config.LOG_FILE)
        os.makedirs(log_dir, exist_ok=True)

        def _exclude_other_engines(record):
            """
            translated(Insight/Media/Query/Forum)translated，translated。

            translated，translated。
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

        # 【translated】translatedhandler，translated
        # logurutranslated，translated
        log_file_path = str(Path(self.config.LOG_FILE).resolve())

        # translatedhandlers
        handler_exists = False
        for handler_id, handler_config in logger._core.handlers.items():
            if hasattr(handler_config, 'sink'):
                sink = handler_config.sink
                # translatedsinktranslated
                if hasattr(sink, '_name') and sink._name == log_file_path:
                    handler_exists = True
                    logger.debug(f"translatedhandlertranslated，translated: {log_file_path}")
                    break

        if not handler_exists:
            # 【translated】translatedlogger，translated
            # - enqueue=False: translated，translated
            # - buffering=1: translated，translated
            # - level="DEBUG": translated
            # - encoding="utf-8": translatedUTF-8translated
            # - mode="a": translated，translated
            handler_id = logger.add(
                self.config.LOG_FILE,
                level="DEBUG",
                enqueue=False,      # translated，translated
                buffering=1,        # translated，translated
                serialize=False,    # translated，translatedJSON
                encoding="utf-8",   # translatedUTF-8translated
                mode="a",           # translated
                filter=_exclude_other_engines # translated Engine translated，translated
            )
            logger.debug(f"translatedhandler (ID: {handler_id}): {self.config.LOG_FILE}")

        # 【translated】translated
        try:
            with open(self.config.LOG_FILE, 'a', encoding='utf-8') as f:
                f.write('')  # translated
                f.flush()    # translated
        except Exception as e:
            logger.error(f"translated: {self.config.LOG_FILE}, translated: {e}")
            raise
        
    def _initialize_file_baseline(self):
        """
        translated。

        translated Insight/Media/Query translated `FileCountBaseline`，
        translated，translated。
        """
        directories = {
            'insight': 'insight_engine_streamlit_reports',
            'media': 'media_engine_streamlit_reports',
            'query': 'query_engine_streamlit_reports'
        }
        self.file_baseline.initialize_baseline(directories)
    
    def _initialize_llm(self) -> LLMClient:
        """
        translatedLLMtranslated。

        translated API Key / translated / Base URL translated
        `LLMClient` translated，translated。
        """
        return LLMClient(
            api_key=self.config.REPORT_ENGINE_API_KEY,
            model_name=self.config.REPORT_ENGINE_MODEL_NAME,
            base_url=self.config.REPORT_ENGINE_BASE_URL,
        )

    def _initialize_rescue_llms(self) -> List[Tuple[str, LLMClient]]:
        """
        translatedLLMtranslated。

        translated“Report → Forum → Insight → Media”，translated。
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
                logger.warning(f"{label} LLMtranslated，translated: {exc}")
                continue
            clients.append((label, client))
        return clients
    
    def _initialize_nodes(self):
        """
        translated。

        translated、translated、translated、translated，
        translated IR translated。
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
        translated（translatedJSON → IR → HTML）。

        translated：
            1. translated + translated，translated；
            2. translated → translated → translated → translated；
            3. translatedLLM，translated；
            4. translatedDocument IR，translatedHTMLtranslated；
            5. translatedHTML/IR/translated，translated。

        translated:
            query: translated。
            reports: translated Query/Media/Insight translated，translated。
            forum_logs: translated/translated，translatedLLMtranslated。
            custom_template: translatedMarkdowntranslated，translated。
            save_report: translatedHTML、IRtranslated。
            stream_handler: translated，translatedpayload，translatedUItranslated。

        translated:
            dict: translated `html_content` translatedHTML/IR/translated；translated `save_report=False` translatedHTMLtranslated。

        translated:
            Exception: translated，translated。
        """
        start_time = datetime.now()
        report_id = f"report-{uuid4().hex[:8]}"
        self.state.task_id = report_id
        self.state.query = query
        self.state.metadata.query = query
        self.state.mark_processing()

        normalized_reports = self._normalize_reports(reports)

        def emit(event_type: str, payload: Dict[str, Any]):
            """translatedReport Enginetranslated，translated。"""
            if not stream_handler:
                return
            try:
                stream_handler(event_type, payload)
            except Exception as callback_error:  # pragma: no cover - translated
                logger.warning(f"translated: {callback_error}")

        logger.info(f"translated {report_id}: {query}")
        logger.info(f"translated - translated: {len(reports)}, translated: {len(str(forum_logs))}")
        emit('stage', {'stage': 'agent_start', 'report_id': report_id, 'query': query})

        try:
            template_result = self._select_template(query, reports, forum_logs, custom_template)
            template_result = self._ensure_mapping(
                template_result,
                "translated",
                expected_keys=["template_name", "template_content"],
            )
            self.state.metadata.template_used = template_result.get('template_name', '')
            emit('stage', {
                'stage': 'template_selected',
                'template': template_result.get('template_name'),
                'reason': template_result.get('selection_reason')
            })
            emit('progress', {'progress': 10, 'message': 'translated'})
            sections = self._slice_template(template_result.get('template_content', ''))
            if not sections:
                raise ValueError("translated，translated。")
            emit('stage', {'stage': 'template_sliced', 'section_count': len(sections)})

            template_text = template_result.get('template_content', '')
            template_overview = self._build_template_overview(template_text, sections)
            # translated+translated、translated
            layout_design = self._run_stage_with_retry(
                "translated",
                lambda: self.document_layout_node.run(
                    sections,
                    template_text,
                    normalized_reports,
                    forum_logs,
                    query,
                    template_overview,
                ),
                # toc translated tocPlan translated，translatedSchematranslated/translated
                expected_keys=["title", "hero", "tocPlan", "tocTitle"],
            )
            emit('stage', {
                'stage': 'layout_designed',
                'title': layout_design.get('title'),
                'toc': layout_design.get('tocTitle')
            })
            emit('progress', {'progress': 15, 'message': 'translated/translated'})
            # translated，translated
            word_plan = self._run_stage_with_retry(
                "translated",
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
            emit('progress', {'progress': 20, 'message': 'translated'})
            # translated/translated，translatedLLM
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
            # IR/translated，translated/translated/translated/translated
            manifest_meta = {
                "query": query,
                "title": layout_design.get("title") or (f"{query} - translated" if query else template_result.get("template_name")),
                "subtitle": layout_design.get("subtitle"),
                "tagline": layout_design.get("tagline"),
                "templateName": template_result.get("template_name"),
                "selectionReason": template_result.get("selection_reason"),
                "themeTokens": generation_context.get("theme_tokens", {}),
                "toc": {
                    "depth": 3,
                    "autoNumbering": True,
                    "title": layout_design.get("tocTitle") or "translated",
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
            # translatedmanifest，translated
            run_dir = self.chapter_storage.start_session(report_id, manifest_meta)
            self._persist_planning_artifacts(run_dir, layout_design, word_plan, template_overview)
            emit('stage', {'stage': 'storage_ready', 'run_dir': str(run_dir)})

            chapters = []
            chapter_max_attempts = max(
                self._CONTENT_SPARSE_MIN_ATTEMPTS, self.config.CHAPTER_JSON_MAX_ATTEMPTS
            )
            total_chapters = len(sections)  # translated
            completed_chapters = 0  # translated

            for section in sections:
                logger.info(f"translated: {section.title}")
                emit('chapter_status', {
                    'chapterId': section.chapter_id,
                    'title': section.title,
                    'status': 'running'
                })
                # translated：translatedLLMtranslateddeltatranslatedSSE，translated
                def chunk_callback(delta: str, meta: Dict[str, Any], section_ref: TemplateSection = section):
                    """
                    translated。

                    Args:
                        delta: LLMtranslated。
                        meta: translated，translated。
                        section_ref: translated，translated。
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
                            readable_label = "translated"
                        elif isinstance(structured_error, ChapterValidationError):
                            error_kind = "validation"
                            readable_label = "translated"
                        else:
                            error_kind = "json_parse"
                            readable_label = "JSONtranslated"
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
                            "translated {title} {label}（translated {attempt}/{total} translated）: {error}",
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
                                "translated {title} translated，translated（translated {score} translated）translated",
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
                        # translated JSON translated，translated
                        # translated：
                        # - AttributeError: translated list.get() translated
                        # - TypeError: translated
                        # - KeyError: translated
                        # - IndexError: translated
                        # - ValueError: translated（translated LLM translated、translated）
                        # - json.JSONDecodeError: JSON translated（translated）
                        error_type = type(structure_error).__name__
                        logger.warning(
                            "translated {title} translated {error_type}（translated {attempt}/{total} translated），translated: {error}",
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
                            # translated，translated ChapterJsonParseError translated
                            raise ChapterJsonParseError(
                                f"{section.title} translated {error_type} translated {chapter_max_attempts} translated: {structure_error}"
                            ) from structure_error
                        attempt += 1
                        continue
                    except Exception as chapter_error:
                        if not self._should_retry_inappropriate_content_error(chapter_error):
                            raise
                        logger.warning(
                            "translated {title} translated（translated {attempt}/{total} translated），translated: {error}",
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
                        f"{section.title} translatedJSONtranslated {chapter_max_attempts} translated"
                    )
                chapters.append(chapter_payload)
                completed_chapters += 1  # translated
                # translated：20% + 80% * (translated / translated)，translated
                chapter_progress = 20 + round(80 * completed_chapters / total_chapters)
                emit('progress', {
                    'progress': chapter_progress,
                    'message': f'translated {completed_chapters}/{total_chapters} translated'
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
            logger.info(f"translated，translated: {generation_time:.2f} translated")
            emit('metrics', {'generation_seconds': generation_time})
            return {
                'html_content': html_report,
                'report_id': report_id,
                **saved_files
            }

        except Exception as e:
            self.state.mark_failed(str(e))
            logger.exception(f"translated: {str(e)}")
            emit('error', {'stage': 'agent_failed', 'message': str(e)})
            raise
    
    def _select_template(self, query: str, reports: List[Any], forum_logs: str, custom_template: str):
        """
        translated。

        translated；translated、translated
        translated TemplateSelectionNode，translated LLM translated
        translated、translated，translated。

        translated:
            query: translated，translated/translated。
            reports: translated，translatedLLMtranslated。
            forum_logs: translated，translated。
            custom_template: CLI/translatedMarkdowntranslated，translated。

        translated:
            dict: translated `template_name`、`template_content` translated `selection_reason` translated，translated。
        """
        logger.info("translated...")
        
        # translated，translated
        if custom_template:
            logger.info("translated")
            return {
                'template_name': 'custom',
                'template_content': custom_template,
                'selection_reason': 'translated'
            }
        
        template_input = {
            'query': query,
            'reports': reports,
            'forum_logs': forum_logs
        }
        
        try:
            template_result = self.template_selection_node.run(template_input)
            
            # translated
            self.state.metadata.template_used = template_result['template_name']
            
            logger.info(f"translated: {template_result['template_name']}")
            logger.info(f"translated: {template_result['selection_reason']}")
            
            return template_result
        except Exception as e:
            logger.error(f"translated，translated: {str(e)}")
            # translated
            fallback_template = {
                'template_name': 'translated',
                'template_content': self._get_fallback_template_content(),
                'selection_reason': 'translated，translated'
            }
            self.state.metadata.template_used = fallback_template['template_name']
            return fallback_template
    
    def _slice_template(self, template_markdown: str) -> List[TemplateSection]:
        """
        translated，translatedfallback。

        translated `parse_template_sections` translatedMarkdowntranslated/translated
        `TemplateSection` translated，translatedID。
        translated，translated。

        translated:
            template_markdown: translatedMarkdowntranslated。

        translated:
            list[TemplateSection]: translated；translated。
        """
        sections = parse_template_sections(template_markdown)
        if sections:
            return sections
        logger.warning("translated，translated")
        fallback = TemplateSection(
            title="1.0 translated",
            slug="section-1-0",
            order=10,
            depth=1,
            raw_title="1.0 translated",
            number="1.0",
            chapter_id="S1",
            outline=["1.1 translated", "1.2 translated", "1.3 translated"],
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
        translated。

        translated、translated、translated、translated、translated
        translated `generation_context`，translated LLM translated
        translated，translated。

        translated:
            query: translated。
            reports: translated query/media/insight translated。
            forum_logs: translated。
            template_result: translated。
            layout_design: translated/translated/translated。
            chapter_directives: translated。
            word_plan: translated，translated。
            template_overview: translated。

        translated:
            dict: LLMtranslated，translated、translated、translated。
        """
        # translated，translated
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
        translated。

        translated Query/Media/Insight，translated
        translated，translated `_stringify` translated。

        translated:
            reports: translated，translated。

        translated:
            dict: translated `query_engine`/`media_engine`/`insight_engine` translated。
        """
        keys = ["query_engine", "media_engine", "insight_engine"]
        normalized: Dict[str, str] = {}
        for idx, key in enumerate(keys):
            value = reports[idx] if idx < len(reports) else ""
            normalized[key] = self._stringify(value)
        return normalized

    def _should_retry_inappropriate_content_error(self, error: Exception) -> bool:
        """
        translatedLLMtranslated/translated。

        translated，translated
        translated，translated。

        translated:
            error: LLMtranslated。

        translated:
            bool: translatedTrue，translatedFalse。
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
        translatedLLMtranslated。

        translated/translated，translatedAgenttranslated。
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
                    "{stage} translated（translated {attempt}/{total} translated），translated: {error}",
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
        translateddict；translated。
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
                    "{context} translated，translated",
                    context=context,
                )
                return best
            raise StageOutputFormatError(f"{context} translated")

        if value is None:
            raise StageOutputFormatError(f"{context} translated")

        raise StageOutputFormatError(
            f"{context} translated {type(value).__name__}，translated"
        )

    def _normalize_word_plan(self, word_plan: Dict[str, Any], stage_name: str) -> Dict[str, Any]:
        """
        translated，translated chapters/globalGuidelines/totalWords translated。
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
                        "{stage} translated {idx} translated，translated",
                        stage=stage_name,
                        idx=idx + 1,
                    )
                    normalized.append(dict_candidate)
                    continue
            logger.warning(
                "{stage} translated#{idx}（translated: {type_name}）",
                stage=stage_name,
                idx=idx + 1,
                type_name=type(entry).__name__,
            )

        if not normalized:
            raise StageOutputFormatError(f"{stage_name} translated，translated")

        word_plan["chapters"] = normalized

        guidelines = word_plan.get("globalGuidelines")
        if not isinstance(guidelines, list):
            if guidelines is None or guidelines == "":
                word_plan["globalGuidelines"] = []
            else:
                logger.warning(
                    "{stage} globalGuidelines translated，translated",
                    stage=stage_name,
                )
                word_plan["globalGuidelines"] = [guidelines]

        if not isinstance(word_plan.get("totalWords"), (int, float)):
            logger.warning(
                "{stage} totalWords translated，translated 10000",
                stage=stage_name,
            )
            word_plan["totalWords"] = 10000

        return word_plan

    def _finalize_sparse_chapter(self, chapter: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        translated：translatedpayloadtranslated。
        """
        safe_chapter = deepcopy(chapter or {})
        if not isinstance(safe_chapter, dict):
            safe_chapter = {}
        self._ensure_sparse_warning_block(safe_chapter)
        return safe_chapter

    def _ensure_sparse_warning_block(self, chapter: Dict[str, Any]) -> None:
        """
        translated，translated。
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
        translated。

        - dict/list translated JSON，translated；
        - translated `str()`，None translated，translated None translated。

        translated:
            value: translatedPythontranslated。

        translated:
            str: translated/translated。
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
        translated，translated/LLMtranslated。

        translated，translated。

        translated:
            dict: translated、translated、translated、translated。
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
        translated，translated/translated。

        translatedID/slug/ordertranslated，translated。

        translated:
            template_markdown: translated，translated。
            sections: `TemplateSection` translated，translated。

        translated:
            dict: translated。
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
        translatedMarkdowntranslated。

        translated `#` translated；translated，translated
        translated fallback。

        translated:
            template_markdown: translated。
            fallback: translated，translated。

        translated:
            str: translated。
        """
        for line in template_markdown.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
            if stripped:
                fallback = fallback or stripped
        return fallback or "translated"
    
    def _get_fallback_template_content(self) -> str:
        """
        translated。

        translatedLLMtranslated Markdown translated，
        translated。
        """
        return """# translated

## translated
translated，translated。

## translated
### translated
- translated：{event_nature}
- translated：{event_time}
- translated：{event_scope}

## translated
### translated
{sentiment_analysis}

### translated
{opinion_distribution}

## translated
### translated
{media_analysis}

### translated
{report_focus}

## translated
### translated
{direct_impact}

### translated
{potential_impact}

## translated
### translated
{immediate_actions}

### translated
{long_term_strategy}

## translated
{conclusion}

---
*translated：translated*
*translated：{generation_time}*
"""
    
    def _save_report(self, html_content: str, document_ir: Dict[str, Any], report_id: str) -> Dict[str, Any]:
        """
        translatedHTMLtranslatedIRtranslated。

        translated，translated
        `ReportState` translated JSON，translated。

        translated:
            html_content: translatedHTMLtranslated。
            document_ir: Document IRtranslated。
            report_id: translatedID，translated。

        translated:
            dict: translatedHTML/IR/Statetranslated。
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

        logger.info(f"HTMLtranslated: {html_path}")
        logger.info(f"Document IRtranslated: {ir_path}")
        logger.info(f"translated: {state_path}")
        
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
        translatedIRtranslated。

        `Document IR` translated HTML translated，translated
        translated LLM translated。

        translated:
            document_ir: translatedIRtranslated。
            query_safe: translated，translated。
            timestamp: translated，translated。

        translated:
            Path: translatedIRtranslated。
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
        translated、translatedJSON。

        translated（document_layout/word_plan/template_overview）
        translated：translated/translated/translated、
        translated，translated。

        translated:
            run_dir: translated。
            layout_design: translated。
            word_plan: translated。
            template_overview: translatedJSON。
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
                logger.warning(f"translated{name}translated: {exc}")
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """translated，translatedAPItranslated。"""
        return self.state.to_dict()
    
    def load_state(self, filepath: str):
        """translatedstate，translated。"""
        self.state = ReportState.load_from_file(filepath)
        logger.info(f"translated {filepath} translated")
    
    def save_state(self, filepath: str):
        """translated，translated。"""
        self.state.save_to_file(filepath)
        logger.info(f"translated {filepath}")
    
    def check_input_files(self, insight_dir: str, media_dir: str, query_dir: str, forum_log_path: str) -> Dict[str, Any]:
        """
        translated（translated）。
        
        Args:
            insight_dir: InsightEnginetranslated
            media_dir: MediaEnginetranslated
            query_dir: QueryEnginetranslated
            forum_log_path: translated
            
        Returns:
            translated，translated、translated、translated
        """
        # translated
        directories = {
            'insight': insight_dir,
            'media': media_dir,
            'query': query_dir
        }
        
        # translated
        check_result = self.file_baseline.check_new_files(directories)
        
        # translated
        forum_ready = os.path.exists(forum_log_path)
        
        # translated
        result = {
            'ready': check_result['ready'] and forum_ready,
            'baseline_counts': check_result['baseline_counts'],
            'current_counts': check_result['current_counts'],
            'new_files_found': check_result['new_files_found'],
            'missing_files': [],
            'files_found': [],
            'latest_files': {}
        }
        
        # translated
        for engine, new_count in check_result['new_files_found'].items():
            current_count = check_result['current_counts'][engine]
            baseline_count = check_result['baseline_counts'].get(engine, 0)
            
            if new_count > 0:
                result['files_found'].append(f"{engine}: {current_count}translated (translated{new_count}translated)")
            else:
                result['missing_files'].append(f"{engine}: {current_count}translated (translated{baseline_count}translated，translated)")
        
        # translated
        if forum_ready:
            result['files_found'].append(f"forum: {os.path.basename(forum_log_path)}")
        else:
            result['missing_files'].append("forum: translated")
        
        # translated（translated）
        if result['ready']:
            result['latest_files'] = self.file_baseline.get_latest_files(directories)
            if forum_ready:
                result['latest_files']['forum'] = forum_log_path
        
        return result
    
    def load_input_files(self, file_paths: Dict[str, str]) -> Dict[str, Any]:
        """
        translated
        
        Args:
            file_paths: translated
            
        Returns:
            translated，translated `reports` translated `forum_logs` translated
        """
        content = {
            'reports': [],
            'forum_logs': ''
        }
        
        # translated
        engines = ['query', 'media', 'insight']
        for engine in engines:
            if engine in file_paths:
                try:
                    with open(file_paths[engine], 'r', encoding='utf-8') as f:
                        report_content = f.read()
                    content['reports'].append(report_content)
                    logger.info(f"translated {engine} translated: {len(report_content)} translated")
                except Exception as e:
                    logger.exception(f"translated {engine} translated: {str(e)}")
                    content['reports'].append("")
        
        # translated
        if 'forum' in file_paths:
            try:
                with open(file_paths['forum'], 'r', encoding='utf-8') as f:
                    content['forum_logs'] = f.read()
                logger.info(f"translated: {len(content['forum_logs'])} translated")
            except Exception as e:
                logger.exception(f"translated: {str(e)}")
        
        return content


def create_agent(config_file: Optional[str] = None) -> ReportAgent:
    """
    translatedReport Agenttranslated。
    
    Args:
        config_file: translated
        
    Returns:
        ReportAgenttranslated

    translated `Settings`，translated `config_file` translated。
    """
    
    config = Settings() # translated，translated
    return ReportAgent(config)

