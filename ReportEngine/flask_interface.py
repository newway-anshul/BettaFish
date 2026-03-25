"""
Report Engine Flask interface。

This module provides a unified HTTP/SSE entrypoint for frontend/CLI, responsible for:
1. Initialize ReportAgent and coordinate background threads;
2. Manage task queueing, progress queries, streaming push, and log download;
3. Provide template listing, input file checks, and related capabilities.
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


# Create Blueprint
report_bp = Blueprint('report_engine', __name__)

# Global variables
report_agent = None
current_task = None
task_lock = threading.Lock()

# ====== Streaming push and task history management ======
# Use a bounded deque to cache recent events for fast replay after SSE reconnects
MAX_TASK_HISTORY = 5
STREAM_HEARTBEAT_INTERVAL = 15  # Heartbeat interval (seconds)
STREAM_IDLE_TIMEOUT = 120  # Maximum keepalive after terminal state to avoid orphan SSE blocking
STREAM_TERMINAL_STATUSES = {"completed", "error", "cancelled"}
stream_lock = threading.Lock()
stream_subscribers = defaultdict(list)
tasks_registry: Dict[str, 'ReportTask'] = {}
LOG_STREAM_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
log_stream_handler_id: Optional[int] = None

EXCLUDED_ENGINE_PATH_KEYWORDS = ("ForumEngine", "InsightEngine", "MediaEngine", "QueryEngine")

def _is_excluded_engine_log(record: Dict[str, Any]) -> bool:
    """
    Check whether a log comes from other engines (Insight/Media/Query/Forum) to filter mixed-in logs.

    Returns:
        bool: True means it should be filtered (not written/forwarded).
    """
    try:
        file_path = record["file"].path
        if any(keyword in file_path for keyword in EXCLUDED_ENGINE_PATH_KEYWORDS):
            return True
    except Exception:
        pass

    # Fallback: try module-name based filtering when file info is missing
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
    Forward loguru logs to the current task SSE events for real-time frontend visibility.

    Only push when a task is active to avoid unrelated log noise.
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
        # Avoid recursive logging inside the log hook
        pass


def _setup_log_stream_forwarder():
    """Attach a one-time loguru hook for real-time SSE forwarding."""
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
    Register an event queue for the specified task for SSE listeners.

    The returned Queue is stored in `stream_subscribers` and continuously consumed by the SSE generator.

    Args:
        task_id: Task ID to listen to.

    Returns:
        Queue: Thread-safe event queue.
    """
    queue = Queue()
    with stream_lock:
        stream_subscribers[task_id].append(queue)
    return queue


def _unregister_stream(task_id: str, queue: Queue):
    """
    Safely remove an event queue to avoid memory leaks.

    Call in finally to ensure resources are released on exceptions.

    Args:
        task_id: Task ID.
        queue: Previously registered event queue.
    """
    with stream_lock:
        listeners = stream_subscribers.get(task_id, [])
        if queue in listeners:
            listeners.remove(queue)
        if not listeners and task_id in stream_subscribers:
            stream_subscribers.pop(task_id, None)


def _broadcast_event(task_id: str, event: Dict[str, Any]):
    """
    Broadcast an event to all listeners with exception safety.

    Use a shallow copy of listeners to avoid concurrent modification errors.

    Args:
        task_id: Task ID to broadcast to.
        event: Structured event payload.
    """
    with stream_lock:
        listeners = list(stream_subscribers.get(task_id, []))
    for queue in listeners:
        try:
            queue.put(event, timeout=0.1)
        except Exception:
            logger.exception("Failed to push streaming event, skipping current listener queue")


def _prune_task_history_locked():
    """
    Call while task_lock is held to prune old task history.

    Keep only the latest `MAX_TASK_HISTORY` tasks to limit memory usage.

    Note:
        This function assumes the caller already holds `task_lock`; otherwise race conditions may occur.
    """
    if len(tasks_registry) <= MAX_TASK_HISTORY:
        return
    # Sort by creation time and remove the oldest tasks
    sorted_tasks = sorted(tasks_registry.values(), key=lambda t: t.created_at)
    for task in sorted_tasks[:-MAX_TASK_HISTORY]:
        tasks_registry.pop(task.task_id, None)


def _get_task(task_id: str) -> Optional['ReportTask']:
    """
    Unified task lookup, preferring the current task.

    Avoid duplicated lock logic and share across APIs.

    Args:
        task_id: Task ID.

    Returns:
        ReportTask | None: Return task instance if found, otherwise None.
    """
    with task_lock:
        if current_task and current_task.task_id == task_id:
            return current_task
        return tasks_registry.get(task_id)


def _format_sse(event: Dict[str, Any]) -> str:
    """
    Format message according to SSE protocol.

    Output `id:/event:/data:` sections consumable by browsers.

    Args:
        event: Event payload containing at least id/type.

    Returns:
        str: SSE-compliant string.
    """
    payload = json.dumps(event, ensure_ascii=False)
    event_id = event.get('id', 0)
    event_type = event.get('type', 'message')
    return f"id: {event_id}\nevent: {event_type}\ndata: {payload}\n\n"


def _safe_filename_segment(value: str, fallback: str = "report") -> str:
    """
    Generate a safe filename segment with alphanumeric characters and common separators.

    Args:
        value: Original string.
        fallback: Fallback text used when value is empty after sanitization.
    """
    sanitized = "".join(c for c in str(value) if c.isalnum() or c in (" ", "-", "_")).strip()
    sanitized = sanitized.replace(" ", "_")
    return sanitized or fallback


def initialize_report_engine():
    """
    Initialize Report Engine.

    Create a singleton ReportAgent so the API can accept tasks immediately.

    Returns:
        bool: Return True on success, False on exception.
    """
    global report_agent
    try:
        report_agent = create_agent()
        logger.info("Report Engine initialized successfully")
        _setup_log_stream_forwarder()

        # Check PDF generation dependency (Pango)
        try:
            from .utils.dependency_check import log_dependency_status
            log_dependency_status()
        except Exception as dep_err:
            logger.warning(f"Dependency check failed: {dep_err}")

        return True
    except Exception as e:
        logger.exception(f"Report Engine initialization failed: {str(e)}")
        return False


class ReportTask:
    """
    Report generation task.

    This object combines runtime status, progress, event history, and output file paths,
    for both background updates and HTTP reads.
    """

    def __init__(self, query: str, task_id: str, custom_template: str = ""):
        """
        Initialize task object with query, custom template, and runtime metadata.

        Args:
            query: Target report topic
            task_id: Unique task ID, typically timestamp-based
            custom_template: Optional custom Markdown template
        """
        self.task_id = task_id
        self.query = query
        self.custom_template = custom_template
        self.status = "pending"  # Four states (pending/running/completed/error)
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
        # ====== Streaming event cache and concurrency protection ======
        # Use deque to store recent events, protected by lock for thread safety
        self.event_history: deque = deque(maxlen=1000)
        self._event_lock = threading.Lock()
        self.last_event_id = 0

    def update_status(self, status: str, progress: int = None, error_message: str = ""):
        """
        Update task status and broadcast event.

        Automatically refresh `updated_at`, error info, and trigger `status` SSE events.

        Args:
            status: Task phase (pending/running/completed/error/cancelled).
            progress: Optional progress percentage.
            error_message: Human-readable error message.
        """
        self.status = status
        if progress is not None:
            self.progress = progress
        if error_message:
            self.error_message = error_message
        self.updated_at = datetime.now()
        # Push status change events for real-time frontend refresh
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
        """Convert to dict for JSON API responses."""
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
        Put any event into cache and broadcast it.

        Args:
            event_type: SSE event name.
            payload: Business payload data.
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
        Replay history after Last-Event-ID to ensure reconnect continuity.

        Args:
            last_event_id: Last event ID tracked by SSE client.

        Returns:
            list[dict]: Event list after last_event_id.
        """
        with self._event_lock:
            if last_event_id is None:
                return list(self.event_history)
            return [evt for evt in self.event_history if evt['id'] > last_event_id]


def check_engines_ready() -> Dict[str, Any]:
    """
    Check whether all three sub-engines have new files.

    Call ReportAgent baseline checks and include forum log existence,
    as pre-checks for /status and /generate.
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
            'error': 'Report Engine is not initialized'
        }

    return report_agent.check_input_files(
        directories['insight'],
        directories['media'],
        directories['query'],
        forum_log_path
    )


def run_report_generation(task: ReportTask, query: str, custom_template: str = ""):
    """
    Run report generation in background thread。

    Includes: check input -> load documents -> call ReportAgent -> persist outputs ->
    Push stage events. On error, events are pushed and status is updated automatically.

    Args:
        task: Task object for this run, holding event queues.
        query: Report topic.
        custom_template: Optional custom template string.
    """
    global current_task

    try:
        # Wrap push logic in a local closure for ReportAgent callback
        def stream_handler(event_type: str, payload: Dict[str, Any]):
            """All stage events are dispatched through one interface for consistent logging."""
            task.publish_event(event_type, payload)
            # If event contains progress, sync task progress
            if event_type == 'progress' and 'progress' in payload:
                task.update_status("running", payload['progress'])

        task.update_status("running", 5)
        task.publish_event('stage', {'message': 'Task started, checking input files', 'stage': 'prepare'})

        # Check input files
        check_result = check_engines_ready()
        if not check_result['ready']:
            task.update_status("error", 0, f"Input files are not ready: {check_result.get('missing_files', [])}")
            return

        task.publish_event('stage', {
            'message': 'Input check passed, preparing to load content',
            'stage': 'io_ready',
            'files': check_result.get('latest_files', {})
        })

        # Load input files
        content = report_agent.load_input_files(check_result['latest_files'])
        task.publish_event('stage', {'message': 'Source data loaded, starting generation', 'stage': 'data_loaded'})

        # Generate report with retry fallback for transient network jitter
        for attempt in range(1, 3):
            try:
                task.publish_event('stage', {
                    'message': f'Calling ReportAgent to generate report (attempt {attempt} )',
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
                hint_message = "Try switching the Report Engine API to a stronger long-context LLM"
                task.publish_event('warning', {
                    'message': hint_message,
                    'stage': 'agent_running',
                    'attempt': attempt,
                    'reason': 'chapter_json_parse',
                    'error': str(err),
                    'task': task.to_dict(),
                })
                # Old logic: restart Report Engine after JSON parse failure
                # backoff = min(5 * attempt, 15)
                # task.publish_event('stage', {
                #     'message': f'{backoff} seconds later, retry generation',
                #     'stage': 'retry_wait',
                #     'wait_seconds': backoff
                # })
                # time.sleep(backoff)
                raise ChapterJsonParseError(hint_message) from err
            except Exception as err:
                # Push errors to frontend immediately for retry visibility
                task.publish_event('warning', {
                    'message': f'ReportAgent execution failed: {str(err)}',
                    'stage': 'agent_running',
                    'attempt': attempt
                })
                if attempt == 2:
                    raise
                # Simple exponential backoff to avoid frequent rate limits (seconds)
                backoff = min(5 * attempt, 15)
                task.publish_event('stage', {
                    'message': f'{backoff} seconds later, retry generation',
                    'stage': 'retry_wait',
                    'wait_seconds': backoff
                })
                time.sleep(backoff)

        if isinstance(generation_result, dict):
            html_report = generation_result.get('html_content', '')
        else:
            html_report = generation_result

        task.publish_event('stage', {'message': 'Report generation complete, preparing persistence', 'stage': 'persist'})

        # Save result
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
            'message': 'HTML render complete, preview can be refreshed',
            'report_file': task.report_file_relative_path or task.report_file_path,
            'state_file': task.state_file_relative_path or task.state_file_path,
            'task': task.to_dict(),
        })
        task.update_status("completed", 100)
        task.publish_event('completed', {
            'message': 'Task completed',
            'duration_seconds': (task.updated_at - task.created_at).total_seconds(),
            'report_file': task.report_file_relative_path or task.report_file_path,
            'task': task.to_dict(),
        })

    except Exception as e:
        logger.exception(f"Error during report generation: {str(e)}")
        task.update_status("error", 0, str(e))
        task.publish_event('error', {
            'message': str(e),
            'stage': 'failed',
            'task': task.to_dict(),
        })
        # Clean current task only on error
        with task_lock:
            if current_task and current_task.task_id == task.task_id:
                current_task = None


@report_bp.route('/status', methods=['GET'])
def get_status():
    """
    Get Report Engine status, including engine readiness and current task.

    Returns:
        Response: JSON includes initialized/engines_ready/current_task.
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
        logger.exception(f"Failed to get Report Engine status: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@report_bp.route('/generate', methods=['POST'])
def generate_report():
    """
    Start generating report.

    Handle queueing, background thread startup, log clearing, and SSE URL return.

    Request body:
        query: Report topic (optional).
        custom_template: Custom template string (optional).

    Returns:
        Response: JSON containing task_id and SSE stream URL.
    """
    global current_task

    try:
        # Check whether a task is running
        with task_lock:
            if current_task and current_task.status == "running":
                return jsonify({
                    'success': False,
                    'error': 'A report generation task is already running',
                    'current_task': current_task.to_dict()
                }), 400

            # If a completed task exists, clear it
            if current_task and current_task.status in ["completed", "error"]:
                current_task = None

        # Read request parameters
        data = request.get_json() or {}
        if not isinstance(data, dict):
            logger.warning("generate_report received non-object JSON payload; ignored")
            data = {}
        query = data.get('query', 'Intelligent Public Opinion Analysis Report')
        custom_template = data.get('custom_template', '')

        # Clear log file
        clear_report_log()

        # Check whether Report Engine is initialized
        if not report_agent:
            return jsonify({
                'success': False,
                'error': 'Report Engine is not initialized'
            }), 500

        # Check whether input files are ready
        engines_status = check_engines_ready()
        if not engines_status['ready']:
            return jsonify({
                'success': False,
                'error': 'Input files are not ready',
                'missing_files': engines_status.get('missing_files', [])
            }), 400

        # Create new task
        task_id = f"report_{int(time.time())}"
        task = ReportTask(query, task_id, custom_template)

        with task_lock:
            current_task = task
            tasks_registry[task_id] = task
            _prune_task_history_locked()

        # Push pending event proactively to indicate queued task
        task.publish_event(
            'status',
            {
                'status': task.status,
                'progress': task.progress,
                'message': 'Task queued, waiting for available resources',
                'task': task.to_dict(),
            }
        )

        # Run report generation in background thread
        thread = threading.Thread(
            target=run_report_generation,
            args=(task, query, custom_template),
            daemon=True
        )
        thread.start()

        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': 'Report generation started',
            'task': task.to_dict(),
            'stream_url': f"/api/report/stream/{task_id}"
        })

    except Exception as e:
        logger.exception(f"Failed to start report generation: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@report_bp.route('/progress/<task_id>', methods=['GET'])
def get_progress(task_id: str):
    """
    Get report generation progress; if task was pruned, return a completed fallback.

    Args:
        task_id: Unique task identifier.

    Returns:
        Response: JSON containing current task status.
    """
    try:
        task = _get_task(task_id)
        if not task:
            # If task does not exist, history may have been pruned; return a completed fallback
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
        logger.exception(f"Failed to get generation progress: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@report_bp.route('/stream/<task_id>', methods=['GET'])
def stream_task(task_id: str):
    """
    SSE-based real-time streaming endpoint.

    - Auto-replay events after Last-Event-ID;
    - Send periodic heartbeats to prevent proxy interruptions;
    - Auto-unregister listeners after task completion.

    Args:
        task_id: Unique task identifier.

    Returns:
        Response: `text/event-stream` typed response.
    """
    task = _get_task(task_id)
    if not task:
        return jsonify({'success': False, 'error': 'Task does not exist'}), 404

    last_event_header = request.headers.get('Last-Event-ID')
    try:
        last_event_id = int(last_event_header) if last_event_header else None
    except ValueError:
        last_event_id = None

    def client_disconnected() -> bool:
        """
        Detect client disconnect early to avoid BrokenPipe writes.

        On Windows, eventlet may raise ConnectionAbortedError when closing a connection.
        Exiting generator early reduces meaningless logs.
        """
        try:
            env_input = request.environ.get('wsgi.input')
            return bool(getattr(env_input, 'closed', False))
        except Exception:
            return False

    def event_generator():
        """
        SSE event generator.

        - Registers and consumes the task event queue;
        - Replays history before listening to live events;
        - Sends periodic heartbeats and auto-unregisters after completion.
        """
        queue = _register_stream(task_id)
        last_data_ts = time.time()
        try:
            # On reconnect, replay history first to keep UI state consistent
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
                    logger.info(f"SSE client disconnected, stop streaming: {task_id}")
                    break
                event = None
                try:
                    event = queue.get(timeout=STREAM_HEARTBEAT_INTERVAL)
                except Empty:
                    if task.status in STREAM_TERMINAL_STATUSES:
                        logger.info(f"Task {task_id} has finished with no new events, SSE closes automatically")
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
                    logger.warning(f"SSE failed to get event (task {task_id}), ending early")
                    break

                try:
                    yield _format_sse(event)
                    if event.get('type') != 'heartbeat':
                        last_data_ts = time.time()
                except GeneratorExit:
                    logger.info(f"SSE generator closed, stop streaming task {task_id}")
                    break
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as exc:
                    logger.warning(f"SSE connection interrupted by client (task {task_id}): {exc}")
                    break
                except Exception as exc:
                    event_type = event.get('type') if isinstance(event, dict) else 'unknown'
                    logger.exception(f"SSE push failed (task {task_id}, event {event_type}): {exc}")
                    break

                if event.get('type') in ("completed", "error", "cancelled"):
                    finished = True
                else:
                    finished = finished or task.status in STREAM_TERMINAL_STATUSES

                # In terminal state, keepalive for limited time to avoid backend loop hanging
                if task.status in STREAM_TERMINAL_STATUSES:
                    idle_for = time.time() - last_data_ts
                    if idle_for > STREAM_IDLE_TIMEOUT:
                        logger.info(f"Task {task_id} is terminal and idle for {int(idle_for)}s, proactively close SSE")
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
    Get report generation result.

    Args:
        task_id: Task ID.

    Returns:
        Response: JSON including HTML preview and file paths.
    """
    try:
        task = _get_task(task_id)
        if not task:
            return jsonify({
                'success': False,
                'error': 'Task does not exist'
            }), 404

        if task.status != "completed":
            return jsonify({
                'success': False,
                'error': 'Report is not finished yet',
                'task': task.to_dict()
            }), 400

        return Response(
            task.html_content,
            mimetype='text/html'
        )

    except Exception as e:
        logger.exception(f"Failed to get report generation result: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@report_bp.route('/result/<task_id>/json', methods=['GET'])
def get_result_json(task_id: str):
    """Get report generation result (JSON)"""
    try:
        task = _get_task(task_id)
        if not task:
            return jsonify({
                'success': False,
                'error': 'Task does not exist'
            }), 404

        if task.status != "completed":
            return jsonify({
                'success': False,
                'error': 'Report is not finished yet',
                'task': task.to_dict()
            }), 400

        return jsonify({
            'success': True,
            'task': task.to_dict(),
            'html_content': task.html_content
        })

    except Exception as e:
        logger.exception(f"Failed to get report generation result: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@report_bp.route('/download/<task_id>', methods=['GET'])
def download_report(task_id: str):
    """
    Download generated report HTML file.

    Args:
        task_id: Task ID.

    Returns:
        Response: Attachment response for HTML file download.
    """
    try:
        task = _get_task(task_id)
        if not task:
            return jsonify({
                'success': False,
                'error': 'Task does not exist'
            }), 404

        if task.status != "completed" or not task.report_file_path:
            return jsonify({
                'success': False,
                'error': 'Report is not finished or not yet saved'
            }), 400

        if not os.path.exists(task.report_file_path):
            return jsonify({
                'success': False,
                'error': 'Report file does not exist or was deleted'
            }), 404

        download_name = task.report_file_name or os.path.basename(task.report_file_path)
        return send_file(
            task.report_file_path,
            mimetype='text/html',
            as_attachment=True,
            download_name=download_name
        )

    except Exception as e:
        logger.exception(f"Failed to download report: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@report_bp.route('/cancel/<task_id>', methods=['POST'])
def cancel_task(task_id: str):
    """
    Cancel report generation task.

    Args:
        task_id: Task ID to cancel.

    Returns:
        Response: JSON with cancellation result or error info.
    """
    global current_task

    try:
        with task_lock:
            if current_task and current_task.task_id == task_id:
                if current_task.status == "running":
                    current_task.update_status("cancelled", 0, "Task cancelled by user")
                    current_task.publish_event('cancelled', {
                        'message': 'Task terminated by user',
                        'task': current_task.to_dict(),
                    })
                current_task = None
            task = tasks_registry.get(task_id)
            if task and task.status == 'running':
                task.update_status("cancelled", task.progress, "Task cancelled by user")
                task.publish_event('cancelled', {
                    'message': 'Task terminated by user',
                    'task': task.to_dict(),
                })

                return jsonify({
                    'success': True,
                    'message': 'Task cancelled'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Task does not exist or cannot be cancelled'
                }), 404

    except Exception as e:
        logger.exception(f"Failed to cancel report generation task: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@report_bp.route('/templates', methods=['GET'])
def get_templates():
    """
    Get available templates for frontend template selection.

    Returns:
        Response: JSON listing template name/description/size.
    """
    try:
        if not report_agent:
            return jsonify({
                'success': False,
                'error': 'Report Engine is not initialized'
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
                            'description': content.split('\n')[0] if content else 'No description',
                            'size': len(content)
                        })
                    except Exception as e:
                        logger.exception(f"Failed to read template {filename}: {str(e)}")

        return jsonify({
            'success': True,
            'templates': templates,
            'template_dir': template_dir
        })

    except Exception as e:
        logger.exception(f"Failed to get template list: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Error handling
@report_bp.errorhandler(404)
def not_found(error):
    """404 fallback: ensure unified JSON response format"""
    logger.exception(f"API endpoint not found: {str(error)}")
    return jsonify({
        'success': False,
        'error': 'API endpoint not found'
    }), 404


@report_bp.errorhandler(500)
def internal_error(error):
    """500 fallback: capture uncaught exceptions"""
    logger.exception(f"Internal server error: {str(error)}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


def clear_report_log():
    """
    Clear report.log so new task views only current run logs.

    Returns:
        None
    """
    try:
        log_file = settings.LOG_FILE

        # Fix: use truncate instead of reopening to avoid logger file-handle conflicts
        # Open in append/update mode then truncate to keep file handle valid
        with open(log_file, 'r+', encoding='utf-8') as f:
            f.truncate(0)  # Truncate file content without closing file
            f.flush()      # Flush immediately

        logger.info(f"Log file cleared: {log_file}")
    except FileNotFoundError:
        # File does not exist; create empty file
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write('')
            logger.info(f"Created log file: {log_file}")
        except Exception as e:
            logger.exception(f"Failed to create log file: {str(e)}")
    except Exception as e:
        logger.exception(f"Failed to clear log file: {str(e)}")


@report_bp.route('/log', methods=['GET'])
def get_report_log():
    """
    Get report.log content and return non-empty lines.

    Fix: optimize large-file reads with error handling and file safety

    Returns:
        Response: JSON containing latest log lines.
    """
    try:
        log_file = settings.LOG_FILE

        if not os.path.exists(log_file):
            return jsonify({
                'success': True,
                'log_lines': []
            })

        # Fix: check file size to avoid memory issues from oversized reads
        file_size = os.path.getsize(log_file)
        max_size = 10 * 1024 * 1024  # 10MB limit

        if file_size > max_size:
            # File too large, read only last 10MB
            with open(log_file, 'rb') as f:
                f.seek(-max_size, 2)  # Seek 10MB backward from file end
                # Skip possibly incomplete first line
                f.readline()
                content = f.read().decode('utf-8', errors='replace')
            lines = content.splitlines()
            logger.warning(f"Log file is too large ({file_size} bytes), returning only last {max_size} bytes")
        else:
            # Normal size, read all
            with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()

        # Trim line endings and empty lines
        log_lines = [line.rstrip('\n\r') for line in lines if line.strip()]

        return jsonify({
            'success': True,
            'log_lines': log_lines
        })

    except PermissionError as e:
        logger.error(f"Insufficient permission to read logs: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Insufficient permission to read logs'
        }), 403
    except UnicodeDecodeError as e:
        logger.error(f"Log file encoding error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Log file encoding error'
        }), 500
    except Exception as e:
        logger.exception(f"Failed to read log: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to read log: {str(e)}'
        }), 500


@report_bp.route('/log/clear', methods=['POST'])
def clear_log():
    """
    Clear logs manually via REST endpoint for one-click frontend reset.

    Returns:
        Response: JSON indicating whether clear succeeded.
    """
    try:
        clear_report_log()
        return jsonify({
            'success': True,
            'message': 'Logs cleared'
        })
    except Exception as e:
        logger.exception(f"Failed to clear logs: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to clear logs: {str(e)}'
        }), 500


@report_bp.route('/export/md/<task_id>', methods=['GET'])
def export_markdown(task_id: str):
    """
    Export report as Markdown.

    Use saved Document IR with MarkdownRenderer to generate and download file.
    """
    try:
        task = tasks_registry.get(task_id)
        if not task:
            return jsonify({
                'success': False,
                'error': 'Task does not exist'
            }), 404

        if task.status != 'completed':
            return jsonify({
                'success': False,
                'error': f'Task not completed, current status: {task.status}'
            }), 400

        if not task.ir_file_path or not os.path.exists(task.ir_file_path):
            return jsonify({
                'success': False,
                'error': 'IR file does not exist, cannot generate Markdown'
            }), 404

        with open(task.ir_file_path, 'r', encoding='utf-8') as f:
            document_ir = json.load(f)

        from .renderers import MarkdownRenderer
        renderer = MarkdownRenderer()
        # Pass ir_file_path so repaired charts are auto-saved to IR file
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

        logger.info(f"Markdown export completed: {md_path}")

        return send_file(
            task.markdown_file_path,
            mimetype='text/markdown',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        logger.exception(f"Markdown export failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Markdown export failed: {str(e)}'
        }), 500


@report_bp.route('/export/pdf/<task_id>', methods=['GET'])
def export_pdf(task_id: str):
    """
    Export report as PDF.

    Generate optimized PDF from IR JSON with auto layout optimization.

    Args:
        task_id: Task ID

    Query params:
        optimize: Whether to enable layout optimization (default true)

    Returns:
        Response: PDF file stream or error message
    """
    try:
        # Check Pango dependency
        from .utils.dependency_check import check_pango_available
        pango_available, pango_message = check_pango_available()
        if not pango_available:
            return jsonify({
                'success': False,
                'error': 'PDF export unavailable: missing system dependency',
                'details': 'See README in repository root for PDF dependency installation steps',
                'help_url': 'https://github.com/666ghj/BettaFish',
                'system_message': pango_message
            }), 503

        # Get task info
        task = tasks_registry.get(task_id)
        if not task:
            return jsonify({
                'success': False,
                'error': 'Task does not exist'
            }), 404

        # Check whether task is completed
        if task.status != 'completed':
            return jsonify({
                'success': False,
                'error': f'Task not completed, current status: {task.status}'
            }), 400

        # Get IR file path
        if not task.ir_file_path or not os.path.exists(task.ir_file_path):
            return jsonify({
                'success': False,
                'error': 'IR file does not exist'
            }), 404

        # Read IR data
        with open(task.ir_file_path, 'r', encoding='utf-8') as f:
            document_ir = json.load(f)

        # Check whether layout optimization is enabled
        optimize = request.args.get('optimize', 'true').lower() == 'true'

        # Create PDF renderer and generate PDF
        from .renderers import PDFRenderer
        renderer = PDFRenderer()

        logger.info(f"Start PDF export, task ID: {task_id} , layout optimization: {optimize}")

        # Generate PDF byte stream
        pdf_bytes = renderer.render_to_bytes(document_ir, optimize_layout=optimize)

        # Determine download filename
        topic = document_ir.get('metadata', {}).get('topic', 'report')
        pdf_filename = f"report_{topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        # Return PDF file
        return Response(
            pdf_bytes,
            mimetype='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="{pdf_filename}"',
                'Content-Type': 'application/pdf'
            }
        )

    except Exception as e:
        logger.exception(f"PDF export failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'PDF export failed: {str(e)}'
        }), 500


@report_bp.route('/export/pdf-from-ir', methods=['POST'])
def export_pdf_from_ir():
    """
    Export PDF directly from IR JSON (no task ID required).

    For scenarios where frontend passes IR directly.

    Request body:
        {
            "document_ir": {...},  // Document IR JSON
            "optimize": true       // enable layout optimization (optional)
        }

    Returns:
        Response: PDF file stream or error message
    """
    try:
        # Check Pango dependency
        from .utils.dependency_check import check_pango_available
        pango_available, pango_message = check_pango_available()
        if not pango_available:
            return jsonify({
                'success': False,
                'error': 'PDF export unavailable: missing system dependency',
                'details': 'See README in repository root for PDF dependency installation steps',
                'help_url': 'https://github.com/666ghj/BettaFish',
                'system_message': pango_message
            }), 503

        data = request.get_json() or {}
        if not isinstance(data, dict):
            logger.warning("export_pdf_from_ir request body is not a JSON object")
            return jsonify({
                'success': False,
                'error': 'Request body must be a JSON object'
            }), 400

        if not data or 'document_ir' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing document_ir parameter'
            }), 400

        document_ir = data['document_ir']
        optimize = data.get('optimize', True)

        # Create PDF renderer and generate PDF
        from .renderers import PDFRenderer
        renderer = PDFRenderer()

        logger.info(f"Export PDF directly from IR, layout optimization: {optimize}")

        # Generate PDF byte stream
        pdf_bytes = renderer.render_to_bytes(document_ir, optimize_layout=optimize)

        # Determine download filename
        topic = document_ir.get('metadata', {}).get('topic', 'report')
        pdf_filename = f"report_{topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        # Return PDF file
        return Response(
            pdf_bytes,
            mimetype='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="{pdf_filename}"',
                'Content-Type': 'application/pdf'
            }
        )

    except Exception as e:
        logger.exception(f"Export PDF from IR failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'PDF export failed: {str(e)}'
        }), 500
