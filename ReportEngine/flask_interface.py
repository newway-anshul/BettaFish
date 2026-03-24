"""
Report Engine Flasktranslated。

translated/CLItranslatedHTTP/SSEtranslated，translated：
1. translated ReportAgent translated；
2. translated、translated、translated；
3. translated、translated。
"""

import os
import json
import threading
import time
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from flask import Blueprint, request, jsonify, Response, send_file, stream_with_context
from typing import Dict, Any, List, Optional
from loguru import logger
from .agent import ReportAgent, create_agent
from .nodes import ChapterJsonParseError
from .utils.config import settings


# translatedBlueprint
report_bp = Blueprint('report_engine', __name__)

# translated
report_agent = None
current_task = None
task_lock = threading.Lock()

# ====== translated ======
# translateddequetranslated，translatedSSEtranslated
MAX_TASK_HISTORY = 5
STREAM_HEARTBEAT_INTERVAL = 15  # translated
STREAM_IDLE_TIMEOUT = 120  # translated，translatedSSEtranslated
STREAM_TERMINAL_STATUSES = {"completed", "error", "cancelled"}
stream_lock = threading.Lock()
stream_subscribers = defaultdict(list)
tasks_registry: Dict[str, 'ReportTask'] = {}
LOG_STREAM_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
log_stream_handler_id: Optional[int] = None

EXCLUDED_ENGINE_PATH_KEYWORDS = ("ForumEngine", "InsightEngine", "MediaEngine", "QueryEngine")

def _is_excluded_engine_log(record: Dict[str, Any]) -> bool:
    """
    translated（Insight/Media/Query/Forum），translated。

    translated:
        bool: True translated（translated/translated）。
    """
    try:
        file_path = record["file"].path
        if any(keyword in file_path for keyword in EXCLUDED_ENGINE_PATH_KEYWORDS):
            return True
    except Exception:
        pass

    # translated：translated，translatedfiletranslated
    try:
        module_name = record.get("module", "")
        if isinstance(module_name, str):
            lowered = module_name.lower()
            return any(keyword.lower() in lowered for keyword in EXCLUDED_ENGINE_PATH_KEYWORDS)
    except Exception:
        pass

    return False


def _stream_log_to_task(message):
    """
    translatedlogurutranslatedSSEtranslated，translated。

    translated，translated。
    """
    try:
        record = message.record
        level_name = record["level"].name
        if level_name not in LOG_STREAM_LEVELS:
            return
        if _is_excluded_engine_log(record):
            return

        with task_lock:
            task = current_task

        if not task or task.status not in ("running", "pending"):
            return

        timestamp = record["time"].strftime("%H:%M:%S.%f")[:-3]
        formatted_line = f"[{timestamp}] [{level_name}] {record['message']}"
        task.publish_event(
            "log",
            {
                "line": formatted_line,
                "level": level_name.lower(),
                "timestamp": timestamp,
                "message": record["message"],
                "module": record.get("module", ""),
                "function": record.get("function", ""),
            },
        )
    except Exception:
        # translated
        pass


def _setup_log_stream_forwarder():
    """translatedlogurutranslated，translatedSSEtranslated。"""
    global log_stream_handler_id
    if log_stream_handler_id is not None:
        return
    log_stream_handler_id = logger.add(
        _stream_log_to_task,
        level="DEBUG",
        enqueue=False,
        catch=True,
    )


def _register_stream(task_id: str) -> Queue:
    """
    translated，translatedSSEtranslated。

    translated Queue translated `stream_subscribers`，SSE translated。

    translated:
        task_id: translatedID。

    translated:
        Queue: translated。
    """
    queue = Queue()
    with stream_lock:
        stream_subscribers[task_id].append(queue)
    return queue


def _unregister_stream(task_id: str, queue: Queue):
    """
    translated，translated。

    translatedfinallytranslated，translated。

    translated:
        task_id: translatedID。
        queue: translated。
    """
    with stream_lock:
        listeners = stream_subscribers.get(task_id, [])
        if queue in listeners:
            listeners.remove(queue)
        if not listeners and task_id in stream_subscribers:
            stream_subscribers.pop(task_id, None)


def _broadcast_event(task_id: str, event: Dict[str, Any]):
    """
    translated，translated。

    translated，translated。

    translated:
        task_id: translatedID。
        event: translatedpayload。
    """
    with stream_lock:
        listeners = list(stream_subscribers.get(task_id, []))
    for queue in listeners:
        try:
            queue.put(event, timeout=0.1)
        except Exception:
            logger.exception("translated，translated")


def _prune_task_history_locked():
    """
    translatedtask_locktranslated，translated。

    translated `MAX_TASK_HISTORY` translated，translated。

    translated:
        translated `task_lock`，translated。
    """
    if len(tasks_registry) <= MAX_TASK_HISTORY:
        return
    # translated，translated
    sorted_tasks = sorted(tasks_registry.values(), key=lambda t: t.created_at)
    for task in sorted_tasks[:-MAX_TASK_HISTORY]:
        tasks_registry.pop(task.task_id, None)


def _get_task(task_id: str) -> Optional['ReportTask']:
    """
    translated，translated。

    translated，translatedAPItranslated。

    translated:
        task_id: translatedID。

    translated:
        ReportTask | None: translated，translatedNone。
    """
    with task_lock:
        if current_task and current_task.task_id == task_id:
            return current_task
        return tasks_registry.get(task_id)


def _format_sse(event: Dict[str, Any]) -> str:
    """
    translatedSSEtranslated。

    translated `id:/event:/data:` translated，translated。

    translated:
        event: translatedpayload，translated id/type。

    translated:
        str: SSEtranslated。
    """
    payload = json.dumps(event, ensure_ascii=False)
    event_id = event.get('id', 0)
    event_type = event.get('type', 'message')
    return f"id: {event_id}\nevent: {event_type}\ndata: {payload}\n\n"


def _safe_filename_segment(value: str, fallback: str = "report") -> str:
    """
    translated，translated。

    translated:
        value: translated。
        fallback: translated，translatedvaluetranslated。
    """
    sanitized = "".join(c for c in str(value) if c.isalnum() or c in (" ", "-", "_")).strip()
    sanitized = sanitized.replace(" ", "_")
    return sanitized or fallback


def initialize_report_engine():
    """
    translatedReport Engine。

    translated ReportAgent，translated API translated。

    translated:
        bool: translatedTrue，translatedFalse。
    """
    global report_agent
    try:
        report_agent = create_agent()
        logger.info("Report Enginetranslated")
        _setup_log_stream_forwarder()

        # translated PDF translated（Pango）
        try:
            from .utils.dependency_check import log_dependency_status
            log_dependency_status()
        except Exception as dep_err:
            logger.warning(f"translated: {dep_err}")

        return True
    except Exception as e:
        logger.exception(f"Report Enginetranslated: {str(e)}")
        return False


class ReportTask:
    """
    translated。

    translated、translated、translated，
    translated，translatedHTTPtranslated。
    """

    def __init__(self, query: str, task_id: str, custom_template: str = ""):
        """
        translated，translated、translated。

        Args:
            query: translated
            task_id: translatedID，translated
            custom_template: translatedMarkdowntranslated
        """
        self.task_id = task_id
        self.query = query
        self.custom_template = custom_template
        self.status = "pending"  # translated（pending/running/completed/error）
        self.progress = 0
        self.result = None
        self.error_message = ""
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.html_content = ""
        self.report_file_path = ""
        self.report_file_relative_path = ""
        self.report_file_name = ""
        self.state_file_path = ""
        self.state_file_relative_path = ""
        self.ir_file_path = ""
        self.ir_file_relative_path = ""
        self.markdown_file_path = ""
        self.markdown_file_relative_path = ""
        self.markdown_file_name = ""
        # ====== translated ======
        # translateddequetranslated，translated
        self.event_history: deque = deque(maxlen=1000)
        self._event_lock = threading.Lock()
        self.last_event_id = 0

    def update_status(self, status: str, progress: int = None, error_message: str = ""):
        """
        translated。

        translated `updated_at`、translated，translated `status` translated SSE。

        translated:
            status: translated（pending/running/completed/error/cancelled）。
            progress: translated。
            error_message: translated。
        """
        self.status = status
        if progress is not None:
            self.progress = progress
        if error_message:
            self.error_message = error_message
        self.updated_at = datetime.now()
        # translated，translated
        self.publish_event(
            'status',
            {
                'status': self.status,
                'progress': self.progress,
                'error_message': self.error_message,
                'hint': error_message or '',
                'task': self.to_dict(),
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """translated，translatedJSON API。"""
        return {
            'task_id': self.task_id,
            'query': self.query,
            'status': self.status,
            'progress': self.progress,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'has_result': bool(self.html_content),
            'report_file_ready': bool(self.report_file_path),
            'report_file_name': self.report_file_name,
            'report_file_path': self.report_file_relative_path or self.report_file_path,
            'state_file_ready': bool(self.state_file_path),
            'state_file_path': self.state_file_relative_path or self.state_file_path,
            'ir_file_ready': bool(self.ir_file_path),
            'ir_file_path': self.ir_file_relative_path or self.ir_file_path,
            'markdown_file_ready': bool(self.markdown_file_path),
            'markdown_file_name': self.markdown_file_name,
            'markdown_file_path': self.markdown_file_relative_path or self.markdown_file_path
        }

    def publish_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """
        translated，translated。

        translated:
            event_type: SSEtranslatedeventtranslated。
            payload: translated。
        """
        timestamp = datetime.utcnow().isoformat() + 'Z'
        event: Dict[str, Any] = {
            'id': 0,
            'type': event_type,
            'task_id': self.task_id,
            'timestamp': timestamp,
            'payload': payload,
        }
        with self._event_lock:
            self.last_event_id += 1
            event['id'] = self.last_event_id
            self.event_history.append(event)
        _broadcast_event(self.task_id, event)

    def history_since(self, last_event_id: Optional[int]) -> List[Dict[str, Any]]:
        """
        translatedLast-Event-IDtranslated，translated。

        translated:
            last_event_id: SSEtranslatedID。

        translated:
            list[dict]: translated last_event_id translated。
        """
        with self._event_lock:
            if last_event_id is None:
                return list(self.event_history)
            return [evt for evt in self.event_history if evt['id'] > last_event_id]


def check_engines_ready() -> Dict[str, Any]:
    """
    translated。

    translated ReportAgent translated，translated，
    translated /status、/generate translated。
    """
    directories = {
        'insight': 'insight_engine_streamlit_reports',
        'media': 'media_engine_streamlit_reports',
        'query': 'query_engine_streamlit_reports'
    }

    forum_log_path = 'logs/forum.log'

    if not report_agent:
        return {
            'ready': False,
            'error': 'Report Enginetranslated'
        }

    return report_agent.check_input_files(
        directories['insight'],
        directories['media'],
        directories['query'],
        forum_log_path
    )


def run_report_generation(task: ReportTask, query: str, custom_template: str = ""):
    """
    translated。

    translated：translated→translated→translatedReportAgent→translated→
    translated。translated。

    translated:
        task: translated，translated。
        query: translated。
        custom_template: translated。
    """
    global current_task

    try:
        # translated，translatedReportAgent
        def stream_handler(event_type: str, payload: Dict[str, Any]):
            """translated，translated。"""
            task.publish_event(event_type, payload)
            # translated，translated
            if event_type == 'progress' and 'progress' in payload:
                task.update_status("running", payload['progress'])

        task.update_status("running", 5)
        task.publish_event('stage', {'message': 'translated，translated', 'stage': 'prepare'})

        # translated
        check_result = check_engines_ready()
        if not check_result['ready']:
            task.update_status("error", 0, f"translated: {check_result.get('missing_files', [])}")
            return

        task.publish_event('stage', {
            'message': 'translated，translated',
            'stage': 'io_ready',
            'files': check_result.get('latest_files', {})
        })

        # translated
        content = report_agent.load_input_files(check_result['latest_files'])
        task.publish_event('stage', {'message': 'translated，translated', 'stage': 'data_loaded'})

        # translated（translated，translated）
        for attempt in range(1, 3):
            try:
                task.publish_event('stage', {
                    'message': f'translatedReportAgenttranslated（translated{attempt}translated）',
                    'stage': 'agent_running',
                    'attempt': attempt
                })
                generation_result = report_agent.generate_report(
                    query=query,
                    reports=content['reports'],
                    forum_logs=content['forum_logs'],
                    custom_template=custom_template,
                    save_report=True,
                    stream_handler=stream_handler
                )
                break
            except ChapterJsonParseError as err:
                hint_message = "translatedReport EnginetranslatedAPItranslated、translatedLLM"
                task.publish_event('warning', {
                    'message': hint_message,
                    'stage': 'agent_running',
                    'attempt': attempt,
                    'reason': 'chapter_json_parse',
                    'error': str(err),
                    'task': task.to_dict(),
                })
                # translated：translatedJSONtranslatedReport Engine
                # backoff = min(5 * attempt, 15)
                # task.publish_event('stage', {
                #     'message': f'{backoff} translated',
                #     'stage': 'retry_wait',
                #     'wait_seconds': backoff
                # })
                # time.sleep(backoff)
                raise ChapterJsonParseError(hint_message) from err
            except Exception as err:
                # translated，translated
                task.publish_event('warning', {
                    'message': f'ReportAgenttranslated: {str(err)}',
                    'stage': 'agent_running',
                    'attempt': attempt
                })
                if attempt == 2:
                    raise
                # translated，translated（translated）
                backoff = min(5 * attempt, 15)
                task.publish_event('stage', {
                    'message': f'{backoff} translated',
                    'stage': 'retry_wait',
                    'wait_seconds': backoff
                })
                time.sleep(backoff)

        if isinstance(generation_result, dict):
            html_report = generation_result.get('html_content', '')
        else:
            html_report = generation_result

        task.publish_event('stage', {'message': 'translated，translated', 'stage': 'persist'})

        # translated
        task.html_content = html_report
        if isinstance(generation_result, dict):
            task.report_file_path = generation_result.get('report_filepath', '')
            task.report_file_relative_path = generation_result.get('report_relative_path', '')
            task.report_file_name = generation_result.get('report_filename', '')
            task.state_file_path = generation_result.get('state_filepath', '')
            task.state_file_relative_path = generation_result.get('state_relative_path', '')
            task.ir_file_path = generation_result.get('ir_filepath', '')
            task.ir_file_relative_path = generation_result.get('ir_relative_path', '')
        task.publish_event('html_ready', {
            'message': 'HTMLtranslated，translated',
            'report_file': task.report_file_relative_path or task.report_file_path,
            'state_file': task.state_file_relative_path or task.state_file_path,
            'task': task.to_dict(),
        })
        task.update_status("completed", 100)
        task.publish_event('completed', {
            'message': 'translated',
            'duration_seconds': (task.updated_at - task.created_at).total_seconds(),
            'report_file': task.report_file_relative_path or task.report_file_path,
            'task': task.to_dict(),
        })

    except Exception as e:
        logger.exception(f"translated: {str(e)}")
        task.update_status("error", 0, str(e))
        task.publish_event('error', {
            'message': str(e),
            'stage': 'failed',
            'task': task.to_dict(),
        })
        # translated
        with task_lock:
            if current_task and current_task.task_id == task.task_id:
                current_task = None


@report_bp.route('/status', methods=['GET'])
def get_status():
    """
    translatedReport Enginetranslated，translated。

    translated:
        Response: JSONtranslatedinitialized/engines_ready/translated。
    """
    try:
        engines_status = check_engines_ready()

        return jsonify({
            'success': True,
            'initialized': report_agent is not None,
            'engines_ready': engines_status['ready'],
            'files_found': engines_status.get('files_found', []),
            'missing_files': engines_status.get('missing_files', []),
            'current_task': current_task.to_dict() if current_task else None
        })
    except Exception as e:
        logger.exception(f"translatedReport Enginetranslated: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@report_bp.route('/generate', methods=['POST'])
def generate_report():
    """
    translated。

    translated、translated、translatedSSEtranslated。

    translated:
        query: translated（translated）。
        custom_template: translated（translated）。

    translated:
        Response: JSON，translated task_id translated SSE stream url。
    """
    global current_task

    try:
        # translated
        with task_lock:
            if current_task and current_task.status == "running":
                return jsonify({
                    'success': False,
                    'error': 'translated',
                    'current_task': current_task.to_dict()
                }), 400

            # translated，translated
            if current_task and current_task.status in ["completed", "error"]:
                current_task = None

        # translated
        data = request.get_json() or {}
        if not isinstance(data, dict):
            logger.warning("generate_report translatedJSONtranslated，translated")
            data = {}
        query = data.get('query', 'translated')
        custom_template = data.get('custom_template', '')

        # translated
        clear_report_log()

        # translatedReport Enginetranslated
        if not report_agent:
            return jsonify({
                'success': False,
                'error': 'Report Enginetranslated'
            }), 500

        # translated
        engines_status = check_engines_ready()
        if not engines_status['ready']:
            return jsonify({
                'success': False,
                'error': 'translated',
                'missing_files': engines_status.get('missing_files', [])
            }), 400

        # translated
        task_id = f"report_{int(time.time())}"
        task = ReportTask(query, task_id, custom_template)

        with task_lock:
            current_task = task
            tasks_registry[task_id] = task
            _prune_task_history_locked()

        # translatedpendingtranslated
        task.publish_event(
            'status',
            {
                'status': task.status,
                'progress': task.progress,
                'message': 'translated，translated',
                'task': task.to_dict(),
            }
        )

        # translated
        thread = threading.Thread(
            target=run_report_generation,
            args=(task, query, custom_template),
            daemon=True
        )
        thread.start()

        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': 'translated',
            'task': task.to_dict(),
            'stream_url': f"/api/report/stream/{task_id}"
        })

    except Exception as e:
        logger.exception(f"translated: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@report_bp.route('/progress/<task_id>', methods=['GET'])
def get_progress(task_id: str):
    """
    translated，translated。

    translated:
        task_id: translated。

    translated:
        Response: JSONtranslated。
    """
    try:
        task = _get_task(task_id)
        if not task:
            # translated，translated，translated
            return jsonify({
                'success': True,
                'task': {
                    'task_id': task_id,
                    'status': 'completed',
                    'progress': 100,
                    'error_message': '',
                    'has_result': True,
                    'report_file_ready': False,
                    'report_file_name': '',
                    'report_file_path': '',
                    'state_file_ready': False,
                    'state_file_path': ''
                }
            })

        return jsonify({
            'success': True,
            'task': task.to_dict()
        })

    except Exception as e:
        logger.exception(f"translated: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@report_bp.route('/stream/<task_id>', methods=['GET'])
def stream_task(task_id: str):
    """
    translatedSSEtranslated。

    - translatedLast-Event-IDtranslated；
    - translated；
    - translated。

    translated:
        task_id: translated。

    translated:
        Response: `text/event-stream` translated。
    """
    task = _get_task(task_id)
    if not task:
        return jsonify({'success': False, 'error': 'translated'}), 404

    last_event_header = request.headers.get('Last-Event-ID')
    try:
        last_event_id = int(last_event_header) if last_event_header else None
    except ValueError:
        last_event_id = None

    def client_disconnected() -> bool:
        """
        translated，translatedBrokenPipe。

        eventlet translated Windows translated ConnectionAbortedError，
        translated。
        """
        try:
            env_input = request.environ.get('wsgi.input')
            return bool(getattr(env_input, 'closed', False))
        except Exception:
            return False

    def event_generator():
        """
        SSEtranslated。

        - translated；
        - translated；
        - translated。
        """
        queue = _register_stream(task_id)
        last_data_ts = time.time()
        try:
            # translated，translated，translated
            history = task.history_since(last_event_id)
            for event in history:
                yield _format_sse(event)
                if event.get('type') != 'heartbeat':
                    last_data_ts = time.time()

            finished = task.status in STREAM_TERMINAL_STATUSES
            while True:
                if finished:
                    break
                if client_disconnected():
                    logger.info(f"SSEtranslated，translated: {task_id}")
                    break
                event = None
                try:
                    event = queue.get(timeout=STREAM_HEARTBEAT_INTERVAL)
                except Empty:
                    if task.status in STREAM_TERMINAL_STATUSES:
                        logger.info(f"translated {task_id} translated，SSEtranslated")
                        break
                    heartbeat = {
                        'id': f"hb-{int(time.time() * 1000)}",
                        'type': 'heartbeat',
                        'task_id': task_id,
                        'timestamp': datetime.utcnow().isoformat() + 'Z',
                        'payload': {'status': task.status}
                    }
                    event = heartbeat
                if event is None:
                    logger.warning(f"SSEtranslated（task {task_id}），translated")
                    break

                try:
                    yield _format_sse(event)
                    if event.get('type') != 'heartbeat':
                        last_data_ts = time.time()
                except GeneratorExit:
                    logger.info(f"SSEtranslated，translated {task_id} translated")
                    break
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as exc:
                    logger.warning(f"SSEtranslated（task {task_id}）: {exc}")
                    break
                except Exception as exc:
                    event_type = event.get('type') if isinstance(event, dict) else 'unknown'
                    logger.exception(f"SSEtranslated（task {task_id}, event {event_type}）: {exc}")
                    break

                if event.get('type') in ("completed", "error", "cancelled"):
                    finished = True
                else:
                    finished = finished or task.status in STREAM_TERMINAL_STATUSES

                # translated，translated
                if task.status in STREAM_TERMINAL_STATUSES:
                    idle_for = time.time() - last_data_ts
                    if idle_for > STREAM_IDLE_TIMEOUT:
                        logger.info(f"translated {task_id} translated {int(idle_for)}s，translatedSSE")
                        break
        finally:
            _unregister_stream(task_id, queue)

    response = Response(
        stream_with_context(event_generator()),
        mimetype='text/event-stream'
    )
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


@report_bp.route('/result/<task_id>', methods=['GET'])
def get_result(task_id: str):
    """
    translated。

    translated:
        task_id: translatedID。

    translated:
        Response: JSON，translatedHTMLtranslated。
    """
    try:
        task = _get_task(task_id)
        if not task:
            return jsonify({
                'success': False,
                'error': 'translated'
            }), 404

        if task.status != "completed":
            return jsonify({
                'success': False,
                'error': 'translated',
                'task': task.to_dict()
            }), 400

        return Response(
            task.html_content,
            mimetype='text/html'
        )

    except Exception as e:
        logger.exception(f"translated: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@report_bp.route('/result/<task_id>/json', methods=['GET'])
def get_result_json(task_id: str):
    """translated（JSONtranslated）"""
    try:
        task = _get_task(task_id)
        if not task:
            return jsonify({
                'success': False,
                'error': 'translated'
            }), 404

        if task.status != "completed":
            return jsonify({
                'success': False,
                'error': 'translated',
                'task': task.to_dict()
            }), 400

        return jsonify({
            'success': True,
            'task': task.to_dict(),
            'html_content': task.html_content
        })

    except Exception as e:
        logger.exception(f"translated: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@report_bp.route('/download/<task_id>', methods=['GET'])
def download_report(task_id: str):
    """
    translatedHTMLtranslated。

    translated:
        task_id: translatedID。

    translated:
        Response: HTMLtranslated。
    """
    try:
        task = _get_task(task_id)
        if not task:
            return jsonify({
                'success': False,
                'error': 'translated'
            }), 404

        if task.status != "completed" or not task.report_file_path:
            return jsonify({
                'success': False,
                'error': 'translated'
            }), 400

        if not os.path.exists(task.report_file_path):
            return jsonify({
                'success': False,
                'error': 'translated'
            }), 404

        download_name = task.report_file_name or os.path.basename(task.report_file_path)
        return send_file(
            task.report_file_path,
            mimetype='text/html',
            as_attachment=True,
            download_name=download_name
        )

    except Exception as e:
        logger.exception(f"translated: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@report_bp.route('/cancel/<task_id>', methods=['POST'])
def cancel_task(task_id: str):
    """
    translated。

    translated:
        task_id: translatedID。

    translated:
        Response: JSON，translated。
    """
    global current_task

    try:
        with task_lock:
            if current_task and current_task.task_id == task_id:
                if current_task.status == "running":
                    current_task.update_status("cancelled", 0, "translated")
                    current_task.publish_event('cancelled', {
                        'message': 'translated',
                        'task': current_task.to_dict(),
                    })
                current_task = None
            task = tasks_registry.get(task_id)
            if task and task.status == 'running':
                task.update_status("cancelled", task.progress, "translated")
                task.publish_event('cancelled', {
                    'message': 'translated',
                    'task': task.to_dict(),
                })

                return jsonify({
                    'success': True,
                    'message': 'translated'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'translated'
                }), 404

    except Exception as e:
        logger.exception(f"translated: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@report_bp.route('/templates', methods=['GET'])
def get_templates():
    """
    translated，translatedMarkdowntranslated。

    translated:
        Response: JSON，translated/translated/translated。
    """
    try:
        if not report_agent:
            return jsonify({
                'success': False,
                'error': 'Report Enginetranslated'
            }), 500

        template_dir = settings.TEMPLATE_DIR
        templates = []

        if os.path.exists(template_dir):
            for filename in os.listdir(template_dir):
                if filename.endswith('.md'):
                    template_path = os.path.join(template_dir, filename)
                    try:
                        with open(template_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        templates.append({
                            'name': filename.replace('.md', ''),
                            'filename': filename,
                            'description': content.split('\n')[0] if content else 'translated',
                            'size': len(content)
                        })
                    except Exception as e:
                        logger.exception(f"translated {filename}: {str(e)}")

        return jsonify({
            'success': True,
            'templates': templates,
            'template_dir': template_dir
        })

    except Exception as e:
        logger.exception(f"translated: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# translated
@report_bp.errorhandler(404)
def not_found(error):
    """404translated：translatedJSONtranslated"""
    logger.exception(f"APItranslated: {str(error)}")
    return jsonify({
        'success': False,
        'error': 'APItranslated'
    }), 404


@report_bp.errorhandler(500)
def internal_error(error):
    """500translated：translated"""
    logger.exception(f"translated: {str(error)}")
    return jsonify({
        'success': False,
        'error': 'translated'
    }), 500


def clear_report_log():
    """
    translatedreport.logtranslated，translated。

    translated:
        None
    """
    try:
        log_file = settings.LOG_FILE

        # 【translated】translatedtruncatetranslated，translatedloggertranslated
        # translated，translatedtruncate，translated
        with open(log_file, 'r+', encoding='utf-8') as f:
            f.truncate(0)  # translated
            f.flush()      # translated

        logger.info(f"translated: {log_file}")
    except FileNotFoundError:
        # translated，translated
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write('')
            logger.info(f"translated: {log_file}")
        except Exception as e:
            logger.exception(f"translated: {str(e)}")
    except Exception as e:
        logger.exception(f"translated: {str(e)}")


@report_bp.route('/log', methods=['GET'])
def get_report_log():
    """
    translatedreport.logtranslated，translated。

    【translated】translated，translated

    translated:
        Response: JSON，translated。
    """
    try:
        log_file = settings.LOG_FILE

        if not os.path.exists(log_file):
            return jsonify({
                'success': True,
                'log_lines': []
            })

        # 【translated】translated，translated
        file_size = os.path.getsize(log_file)
        max_size = 10 * 1024 * 1024  # 10MBtranslated

        if file_size > max_size:
            # translated，translated10MB
            with open(log_file, 'rb') as f:
                f.seek(-max_size, 2)  # translated10MB
                # translated
                f.readline()
                content = f.read().decode('utf-8', errors='replace')
            lines = content.splitlines()
            logger.warning(f"translated ({file_size} bytes)，translated {max_size} bytes")
        else:
            # translated，translated
            with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()

        # translated
        log_lines = [line.rstrip('\n\r') for line in lines if line.strip()]

        return jsonify({
            'success': True,
            'log_lines': log_lines
        })

    except PermissionError as e:
        logger.error(f"translated: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'translated'
        }), 403
    except UnicodeDecodeError as e:
        logger.error(f"translated: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'translated'
        }), 500
    except Exception as e:
        logger.exception(f"translated: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'translated: {str(e)}'
        }), 500


@report_bp.route('/log/clear', methods=['POST'])
def clear_log():
    """
    translated，translatedRESTtranslated。

    translated:
        Response: JSON，translated。
    """
    try:
        clear_report_log()
        return jsonify({
            'success': True,
            'message': 'translated'
        })
    except Exception as e:
        logger.exception(f"translated: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'translated: {str(e)}'
        }), 500


@report_bp.route('/export/md/<task_id>', methods=['GET'])
def export_markdown(task_id: str):
    """
    translated Markdown translated。

    translated Document IR translated MarkdownRenderer，translated。
    """
    try:
        task = tasks_registry.get(task_id)
        if not task:
            return jsonify({
                'success': False,
                'error': 'translated'
            }), 404

        if task.status != 'completed':
            return jsonify({
                'success': False,
                'error': f'translated，translated: {task.status}'
            }), 400

        if not task.ir_file_path or not os.path.exists(task.ir_file_path):
            return jsonify({
                'success': False,
                'error': 'IRtranslated，translatedMarkdown'
            }), 404

        with open(task.ir_file_path, 'r', encoding='utf-8') as f:
            document_ir = json.load(f)

        from .renderers import MarkdownRenderer
        renderer = MarkdownRenderer()
        # translated ir_file_path，translated IR translated
        markdown_text = renderer.render(document_ir, ir_file_path=task.ir_file_path)

        metadata = document_ir.get('metadata') if isinstance(document_ir, dict) else {}
        topic = (metadata or {}).get('topic') or (metadata or {}).get('title') or (metadata or {}).get('query') or task.query
        safe_topic = _safe_filename_segment(topic or 'report')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"report_{safe_topic}_{timestamp}.md"

        output_dir = Path(settings.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        md_path = output_dir / filename
        md_path.write_text(markdown_text, encoding='utf-8')

        task.markdown_file_path = str(md_path.resolve())
        task.markdown_file_relative_path = os.path.relpath(task.markdown_file_path, os.getcwd())
        task.markdown_file_name = filename

        logger.info(f"translatedMarkdowntranslated: {md_path}")

        return send_file(
            task.markdown_file_path,
            mimetype='text/markdown',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        logger.exception(f"translatedMarkdowntranslated: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'translatedMarkdowntranslated: {str(e)}'
        }), 500


@report_bp.route('/export/pdf/<task_id>', methods=['GET'])
def export_pdf(task_id: str):
    """
    translatedPDFtranslated。

    translatedIR JSONtranslatedPDF，translated。

    translated:
        task_id: translatedID

    translated:
        optimize: translated（translatedtrue）

    translated:
        Response: PDFtranslated
    """
    try:
        # translated Pango translated
        from .utils.dependency_check import check_pango_available
        pango_available, pango_message = check_pango_available()
        if not pango_available:
            return jsonify({
                'success': False,
                'error': 'PDF translated：translated',
                'details': 'translated README.md “translated”translated（PDF translated）translated',
                'help_url': 'https://github.com/666ghj/BettaFish#2-translated-pdf-translated',
                'system_message': pango_message
            }), 503

        # translated
        task = tasks_registry.get(task_id)
        if not task:
            return jsonify({
                'success': False,
                'error': 'translated'
            }), 404

        # translated
        if task.status != 'completed':
            return jsonify({
                'success': False,
                'error': f'translated，translated: {task.status}'
            }), 400

        # translatedIRtranslated
        if not task.ir_file_path or not os.path.exists(task.ir_file_path):
            return jsonify({
                'success': False,
                'error': 'IRtranslated'
            }), 404

        # translatedIRtranslated
        with open(task.ir_file_path, 'r', encoding='utf-8') as f:
            document_ir = json.load(f)

        # translated
        optimize = request.args.get('optimize', 'true').lower() == 'true'

        # translatedPDFtranslatedPDF
        from .renderers import PDFRenderer
        renderer = PDFRenderer()

        logger.info(f"translatedPDF，translatedID: {task_id}，translated: {optimize}")

        # translatedPDFtranslated
        pdf_bytes = renderer.render_to_bytes(document_ir, optimize_layout=optimize)

        # translated
        topic = document_ir.get('metadata', {}).get('topic', 'report')
        pdf_filename = f"report_{topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        # translatedPDFtranslated
        return Response(
            pdf_bytes,
            mimetype='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="{pdf_filename}"',
                'Content-Type': 'application/pdf'
            }
        )

    except Exception as e:
        logger.exception(f"translatedPDFtranslated: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'translatedPDFtranslated: {str(e)}'
        }), 500


@report_bp.route('/export/pdf-from-ir', methods=['POST'])
def export_pdf_from_ir():
    """
    translatedIR JSONtranslatedPDF（translatedID）。

    translatedIRtranslated。

    translated:
        {
            "document_ir": {...},  // Document IR JSON
            "optimize": true       // translated（translated）
        }

    translated:
        Response: PDFtranslated
    """
    try:
        # translated Pango translated
        from .utils.dependency_check import check_pango_available
        pango_available, pango_message = check_pango_available()
        if not pango_available:
            return jsonify({
                'success': False,
                'error': 'PDF translated：translated',
                'details': 'translated README.md “translated”translated（PDF translated）translated',
                'help_url': 'https://github.com/666ghj/BettaFish#2-translated-pdf-translated',
                'system_message': pango_message
            }), 503

        data = request.get_json() or {}
        if not isinstance(data, dict):
            logger.warning("export_pdf_from_ir translatedJSONtranslated")
            return jsonify({
                'success': False,
                'error': 'translatedJSONtranslated'
            }), 400

        if not data or 'document_ir' not in data:
            return jsonify({
                'success': False,
                'error': 'translateddocument_irtranslated'
            }), 400

        document_ir = data['document_ir']
        optimize = data.get('optimize', True)

        # translatedPDFtranslatedPDF
        from .renderers import PDFRenderer
        renderer = PDFRenderer()

        logger.info(f"translatedIRtranslatedPDF，translated: {optimize}")

        # translatedPDFtranslated
        pdf_bytes = renderer.render_to_bytes(document_ir, optimize_layout=optimize)

        # translated
        topic = document_ir.get('metadata', {}).get('topic', 'report')
        pdf_filename = f"report_{topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        # translatedPDFtranslated
        return Response(
            pdf_bytes,
            mimetype='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="{pdf_filename}"',
                'Content-Type': 'application/pdf'
            }
        )

    except Exception as e:
        logger.exception(f"translatedIRtranslatedPDFtranslated: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'translatedPDFtranslated: {str(e)}'
        }), 500

