"""
Flask Main Application - Unified management of three Streamlit applications
"""

import os
import sys

# [Fix] Set environment variables early to ensure all modules use unbuffered mode
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'  # Disable Python output buffering to ensure real-time log output

import subprocess
import time
import threading
from datetime import datetime
from queue import Queue
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import atexit
import requests
from loguru import logger
import importlib
from pathlib import Path
from MindSpider.main import MindSpider

# Import ReportEngine
try:
    from ReportEngine.flask_interface import report_bp, initialize_report_engine
    REPORT_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.error(f"ReportEngine import failed: {e}")
    REPORT_ENGINE_AVAILABLE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Dedicated-to-creating-a-concise-and-versatile-public-opinion-analysis-platform'
socketio = SocketIO(app, cors_allowed_origins="*")

# eventlet occasionally raises ConnectionAbortedError on client disconnect; wrap it defensively
# to avoid polluting logs with meaningless stack traces (only active when eventlet is available).
def _patch_eventlet_disconnect_logging():
    try:
        import eventlet.wsgi  # type: ignore
    except Exception as exc:  # pragma: no cover - only effective in production
        logger.debug(f"eventlet not available, skipping disconnect patch: {exc}")
        return

    try:
        original_finish = eventlet.wsgi.HttpProtocol.finish  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover
        logger.debug(f"eventlet missing HttpProtocol.finish, skipping disconnect patch: {exc}")
        return

    def _safe_finish(self, *args, **kwargs):  # pragma: no cover - only triggered at runtime
        try:
            return original_finish(self, *args, **kwargs)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as exc:
            try:
                environ = getattr(self, 'environ', {}) or {}
                method = environ.get('REQUEST_METHOD', '')
                path = environ.get('PATH_INFO', '')
                logger.warning(f"Client actively disconnected, ignoring exception: {method} {path} ({exc})")
            except Exception:
                logger.warning(f"Client actively disconnected, ignoring exception: {exc}")
            return

    eventlet.wsgi.HttpProtocol.finish = _safe_finish  # type: ignore[attr-defined]
    logger.info("eventlet disconnect safety patch applied successfully")

_patch_eventlet_disconnect_logging()

# Register ReportEngine Blueprint
if REPORT_ENGINE_AVAILABLE:
    app.register_blueprint(report_bp, url_prefix='/api/report')
    logger.info("ReportEngine interface registered")
else:
    logger.info("ReportEngine unavailable, skipping interface registration")

# Create log directory
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)

CONFIG_MODULE_NAME = 'config'
CONFIG_FILE_PATH = Path(__file__).resolve().parent / 'config.py'
CONFIG_KEYS = [
    'HOST',
    'PORT',
    'DB_DIALECT',
    'DB_HOST',
    'DB_PORT',
    'DB_USER',
    'DB_PASSWORD',
    'DB_NAME',
    'DB_CHARSET',
    'INSIGHT_ENGINE_API_KEY',
    'INSIGHT_ENGINE_BASE_URL',
    'INSIGHT_ENGINE_MODEL_NAME',
    'MEDIA_ENGINE_API_KEY',
    'MEDIA_ENGINE_BASE_URL',
    'MEDIA_ENGINE_MODEL_NAME',
    'QUERY_ENGINE_API_KEY',
    'QUERY_ENGINE_BASE_URL',
    'QUERY_ENGINE_MODEL_NAME',
    'REPORT_ENGINE_API_KEY',
    'REPORT_ENGINE_BASE_URL',
    'REPORT_ENGINE_MODEL_NAME',
    'FORUM_HOST_API_KEY',
    'FORUM_HOST_BASE_URL',
    'FORUM_HOST_MODEL_NAME',
    'KEYWORD_OPTIMIZER_API_KEY',
    'KEYWORD_OPTIMIZER_BASE_URL',
    'KEYWORD_OPTIMIZER_MODEL_NAME',
    'TAVILY_API_KEY',
    'SEARCH_TOOL_TYPE',
    'BOCHA_WEB_SEARCH_API_KEY',
    'ANSPIRE_API_KEY'
]


def _load_config_module():
    """Load or reload the config module to ensure latest values are available."""
    importlib.invalidate_caches()
    module = sys.modules.get(CONFIG_MODULE_NAME)
    try:
        if module is None:
            module = importlib.import_module(CONFIG_MODULE_NAME)
        else:
            module = importlib.reload(module)
    except ModuleNotFoundError:
        return None
    return module


def read_config_values():
    """Return the current configuration values that are exposed to the frontend."""
    try:
        # Reload config to get the latest Settings instance
        from config import reload_settings, settings
        reload_settings()
        
        values = {}
        for key in CONFIG_KEYS:
            # Read value from Pydantic Settings instance
            value = getattr(settings, key, None)
            # Convert to string for uniform handling on the frontend.
            if value is None:
                values[key] = ''
            else:
                values[key] = str(value)
        return values
    except Exception as exc:
        logger.exception(f"Failed to read config: {exc}")
        return {}


def _serialize_config_value(value):
    """Serialize Python values back to a config.py assignment-friendly string."""
    if isinstance(value, bool):
        return 'True' if value else 'False'
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return 'None'

    value_str = str(value)
    escaped = value_str.replace('\\', '\\\\').replace('"', '\\"')
    return f'"{escaped}"'


def write_config_values(updates):
    """Persist configuration updates to .env file (Pydantic Settings source)."""
    from pathlib import Path
    
    # Determine .env file path (consistent with logic in config.py)
    project_root = Path(__file__).resolve().parent
    cwd_env = Path.cwd() / ".env"
    env_file_path = cwd_env if cwd_env.exists() else (project_root / ".env")
    
    # Read existing .env file content
    env_lines = []
    env_key_indices = {}  # Track the index position of each key in the file
    if env_file_path.exists():
        env_lines = env_file_path.read_text(encoding='utf-8').splitlines()
        # Extract existing keys and their indices
        for i, line in enumerate(env_lines):
            line_stripped = line.strip()
            if line_stripped and not line_stripped.startswith('#'):
                if '=' in line_stripped:
                    key = line_stripped.split('=')[0].strip()
                    env_key_indices[key] = i
    
    # Update or add configuration entries
    for key, raw_value in updates.items():
        # Format value for .env file (no quotes needed unless string contains spaces)
        if raw_value is None or raw_value == '':
            env_value = ''
        elif isinstance(raw_value, (int, float)):
            env_value = str(raw_value)
        elif isinstance(raw_value, bool):
            env_value = 'True' if raw_value else 'False'
        else:
            value_str = str(raw_value)
            # If the value contains spaces or special characters, wrap in quotes
            if ' ' in value_str or '\n' in value_str or '#' in value_str:
                escaped = value_str.replace('\\', '\\\\').replace('"', '\\"')
                env_value = f'"{escaped}"'
            else:
                env_value = value_str
        
        # Update or add configuration entry
        if key in env_key_indices:
            # Update existing line
            env_lines[env_key_indices[key]] = f'{key}={env_value}'
        else:
            # Append new line to end of file
            env_lines.append(f'{key}={env_value}')
    
    # Write to .env file
    env_file_path.parent.mkdir(parents=True, exist_ok=True)
    env_file_path.write_text('\n'.join(env_lines) + '\n', encoding='utf-8')
    
    # Reload the config module (this re-reads the .env file and creates a new Settings instance)
    _load_config_module()


system_state_lock = threading.Lock()
system_state = {
    'started': False,
    'starting': False,
    'shutdown_in_progress': False
}


def _set_system_state(*, started=None, starting=None):
    """Safely update the cached system state flags."""
    with system_state_lock:
        if started is not None:
            system_state['started'] = started
        if starting is not None:
            system_state['starting'] = starting


def _get_system_state():
    """Return a shallow copy of the system state flags."""
    with system_state_lock:
        return system_state.copy()


def _prepare_system_start():
    """Mark the system as starting if it is not already running or starting."""
    with system_state_lock:
        if system_state['started']:
            return False, 'System is already running'
        if system_state['starting']:
            return False, 'System is starting up'
        system_state['starting'] = True
        return True, None

def _mark_shutdown_requested():
    """Mark shutdown as requested; returns False if a shutdown is already in progress."""
    with system_state_lock:
        if system_state.get('shutdown_in_progress'):
            return False
        system_state['shutdown_in_progress'] = True
        return True


def initialize_system_components():
    """Start all dependent components (Streamlit sub-apps, ForumEngine, ReportEngine)."""
    logs = []
    errors = []
    
    spider = MindSpider()
    if spider.initialize_database():
        logger.info("Database initialized successfully")
    else:
        logger.error("Database initialization failed")

    try:
        stop_forum_engine()
        logs.append("Stopped ForumEngine monitor to avoid file conflicts")
    except Exception as exc:  # pragma: no cover - safety catch
        message = f"Exception while stopping ForumEngine: {exc}"
        logs.append(message)
        logger.exception(message)

    processes['forum']['status'] = 'stopped'

    for app_name, script_path in STREAMLIT_SCRIPTS.items():
        logs.append(f"Checking file: {script_path}")
        if os.path.exists(script_path):
            success, message = start_streamlit_app(app_name, script_path, processes[app_name]['port'])
            logs.append(f"{app_name}: {message}")
            if success:
                startup_success, startup_message = wait_for_app_startup(app_name, 30)
                logs.append(f"{app_name} startup check: {startup_message}")
                if not startup_success:
                    errors.append(f"{app_name} startup failed: {startup_message}")
            else:
                errors.append(f"{app_name} startup failed: {message}")
        else:
            msg = f"File not found: {script_path}"
            logs.append(f"Error: {msg}")
            errors.append(f"{app_name}: {msg}")

    forum_started = False
    try:
        start_forum_engine()
        processes['forum']['status'] = 'running'
        logs.append("ForumEngine started successfully")
        forum_started = True
    except Exception as exc:  # pragma: no cover - fallback catch
        error_msg = f"ForumEngine startup failed: {exc}"
        logs.append(error_msg)
        errors.append(error_msg)

    if REPORT_ENGINE_AVAILABLE:
        try:
            if initialize_report_engine():
                logs.append("ReportEngine initialized successfully")
            else:
                msg = "ReportEngine initialization failed"
                logs.append(msg)
                errors.append(msg)
        except Exception as exc:  # pragma: no cover
            msg = f"ReportEngine initialization exception: {exc}"
            logs.append(msg)
            errors.append(msg)

    if errors:
        cleanup_processes()
        processes['forum']['status'] = 'stopped'
        if forum_started:
            try:
                stop_forum_engine()
            except Exception:  # pragma: no cover
                logger.exception("Failed to stop ForumEngine")
        return False, logs, errors

    return True, logs, []

# Initialize the ForumEngine forum.log file
def init_forum_log():
    """Initialize the forum.log file."""
    try:
        forum_log_file = LOG_DIR / "forum.log"
        # Create and write a header if absent, or truncate and write a fresh header if present
        if not forum_log_file.exists():
            with open(forum_log_file, 'w', encoding='utf-8') as f:
                start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"=== ForumEngine System Initialization - {start_time} ===\n")
            logger.info("ForumEngine: forum.log initialized")
        else:
            with open(forum_log_file, 'w', encoding='utf-8') as f:
                start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"=== ForumEngine System Initialization - {start_time} ===\n")
            logger.info("ForumEngine: forum.log initialized")
    except Exception as e:
        logger.exception(f"ForumEngine: Failed to initialize forum.log: {e}")

# Initialize forum.log
init_forum_log()

# Start ForumEngine intelligent monitoring
def start_forum_engine():
    """Start the ForumEngine forum."""
    try:
        from ForumEngine.monitor import start_forum_monitoring
        logger.info("ForumEngine: Starting forum...")
        success = start_forum_monitoring()
        if not success:
            logger.info("ForumEngine: Forum startup failed")
    except Exception as e:
        logger.exception(f"ForumEngine: Failed to start forum: {e}")

# Stop ForumEngine intelligent monitoring
def stop_forum_engine():
    """Stop the ForumEngine forum."""
    try:
        from ForumEngine.monitor import stop_forum_monitoring
        logger.info("ForumEngine: Stopping forum...")
        stop_forum_monitoring()
        logger.info("ForumEngine: Forum stopped")
    except Exception as e:
        logger.exception(f"ForumEngine: Failed to stop forum: {e}")

def parse_forum_log_line(line):
    """Parse a forum.log line to extract conversation information."""
    import re
    
    # Match format: [time] [source] content (source allows mixed case and spaces)
    pattern = r'\[(\d{2}:\d{2}:\d{2})\]\s*\[([^\]]+)\]\s*(.*)'
    match = re.match(pattern, line)
    
    if not match:
        return None

    timestamp, raw_source, content = match.groups()
    source = raw_source.strip().upper()

    # Filter out system messages and empty content
    if source == 'SYSTEM' or not content.strip():
        return None
    
    # Support three Agents and the host
    if source not in ['QUERY', 'INSIGHT', 'MEDIA', 'HOST']:
        return None
    
    # Decode escaped newlines in the log, preserving multi-line format
    cleaned_content = content.replace('\\n', '\n').replace('\\r', '').strip()
    
    # Determine message type and sender based on source
    if source == 'HOST':
        message_type = 'host'
        sender = 'Forum Host'
    else:
        message_type = 'agent'
        sender = f'{source.title()} Engine'
    
    return {
        'type': message_type,
        'sender': sender,
        'content': cleaned_content,
        'timestamp': timestamp,
        'source': source
    }

# Forum log listener
# Store the historical log send position for each client
forum_log_positions = {}

def monitor_forum_log():
    """Monitor forum.log file for changes and push updates to the frontend."""
    import time
    from pathlib import Path

    forum_log_file = LOG_DIR / "forum.log"
    last_position = 0
    processed_lines = set()  # Track processed lines to avoid duplicates

    # If the file exists, get the initial position without skipping content
    if forum_log_file.exists():
        with open(forum_log_file, 'r', encoding='utf-8', errors='ignore') as f:
            # Record file size without adding to processed_lines
            # so history is available when the user opens the forum tab
            f.seek(0, 2)  # Seek to end of file
            last_position = f.tell()

    while True:
        try:
            if forum_log_file.exists():
                with open(forum_log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(last_position)
                    new_lines = f.readlines()

                    if new_lines:
                        for line in new_lines:
                            line = line.rstrip('\n\r')
                            if line.strip():
                                line_hash = hash(line.strip())

                                # Avoid processing the same line twice
                                if line_hash in processed_lines:
                                    continue

                                processed_lines.add(line_hash)

                                # Parse log line and send forum message
                                parsed_message = parse_forum_log_line(line)
                                if parsed_message:
                                    socketio.emit('forum_message', parsed_message)

                                # Only send console message when forum is displayed in the console
                                timestamp = datetime.now().strftime('%H:%M:%S')
                                formatted_line = f"[{timestamp}] {line}"
                                socketio.emit('console_output', {
                                    'app': 'forum',
                                    'line': formatted_line
                                })

                        last_position = f.tell()

                        # Trim processed_lines set to prevent memory leaks (keep last 1000 hashes)
                        if len(processed_lines) > 1000:
                            # Keep only the last 500 hashes
                            recent_hashes = list(processed_lines)[-500:]
                            processed_lines = set(recent_hashes)

            time.sleep(1)  # Check every second
        except Exception as e:
            logger.error(f"Forum log listener error: {e}")
            time.sleep(5)

# Start Forum log listener thread
forum_monitor_thread = threading.Thread(target=monitor_forum_log, daemon=True)
forum_monitor_thread.start()

# Global variable to store process information
processes = {
    'insight': {'process': None, 'port': 8501, 'status': 'stopped', 'output': [], 'log_file': None, 'healthcheck_started_at': None},
    'media': {'process': None, 'port': 8502, 'status': 'stopped', 'output': [], 'log_file': None, 'healthcheck_started_at': None},
    'query': {'process': None, 'port': 8503, 'status': 'stopped', 'output': [], 'log_file': None, 'healthcheck_started_at': None},
    'forum': {'process': None, 'port': None, 'status': 'stopped', 'output': [], 'log_file': None}  # marked as running after startup
}

STREAMLIT_SCRIPTS = {
    'insight': 'SingleEngineApp/insight_engine_streamlit_app.py',
    'media': 'SingleEngineApp/media_engine_streamlit_app.py',
    'query': 'SingleEngineApp/query_engine_streamlit_app.py'
}

def _log_shutdown_step(message: str):
    """Uniformly log shutdown steps for easier diagnosis."""
    logger.info(f"[Shutdown] {message}")


def _describe_running_children():
    """List currently alive child processes."""
    running = []
    for name, info in processes.items():
        proc = info.get('process')
        if proc is not None and proc.poll() is None:
            port_desc = f", port={info.get('port')}" if info.get('port') else ""
            running.append(f"{name}(pid={proc.pid}{port_desc})")
    return running

# Output queues
output_queues = {
    'insight': Queue(),
    'media': Queue(),
    'query': Queue(),
    'forum': Queue()
}

def write_log_to_file(app_name, line):
    """Write log output to file."""
    try:
        log_file_path = LOG_DIR / f"{app_name}.log"
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
            f.flush()
    except Exception as e:
        logger.error(f"Error writing log for {app_name}: {e}")

def read_log_from_file(app_name, tail_lines=None):
    """Read log output from file."""
    try:
        log_file_path = LOG_DIR / f"{app_name}.log"
        if not log_file_path.exists():
            return []
        
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.rstrip('\n\r') for line in lines if line.strip()]
            
            if tail_lines:
                return lines[-tail_lines:]
            return lines
    except Exception as e:
        logger.exception(f"Error reading log for {app_name}: {e}")
        return []

def read_process_output(process, app_name):
    """Read process output and write to file."""
    import select
    import sys
    
    while True:
        try:
            if process.poll() is not None:
                # Process exited, read remaining output
                remaining_output = process.stdout.read()
                if remaining_output:
                    lines = remaining_output.decode('utf-8', errors='replace').split('\n')
                    for line in lines:
                        line = line.strip()
                        if line:
                            timestamp = datetime.now().strftime('%H:%M:%S')
                            formatted_line = f"[{timestamp}] {line}"
                            write_log_to_file(app_name, formatted_line)
                            socketio.emit('console_output', {
                                'app': app_name,
                                'line': formatted_line
                            })
                break
            
            # Use non-blocking read
            if sys.platform == 'win32':
                # Use a different method on Windows
                output = process.stdout.readline()
                if output:
                    line = output.decode('utf-8', errors='replace').strip()
                    if line:
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        formatted_line = f"[{timestamp}] {line}"
                        
                        # Write to log file
                        write_log_to_file(app_name, formatted_line)
                        
                        # Send to frontend
                        socketio.emit('console_output', {
                            'app': app_name,
                            'line': formatted_line
                        })
                else:
                    # Brief sleep when no output
                    time.sleep(0.1)
            else:
                # Use select on Unix systems
                ready, _, _ = select.select([process.stdout], [], [], 0.1)
                if ready:
                    output = process.stdout.readline()
                    if output:
                        line = output.decode('utf-8', errors='replace').strip()
                        if line:
                            timestamp = datetime.now().strftime('%H:%M:%S')
                            formatted_line = f"[{timestamp}] {line}"
                            
                            # Write to log file
                            write_log_to_file(app_name, formatted_line)
                            
                            # Send to frontend
                            socketio.emit('console_output', {
                                'app': app_name,
                                'line': formatted_line
                            })
                            
        except Exception as e:
            error_msg = f"Error reading output for {app_name}: {e}"
            logger.exception(error_msg)
            write_log_to_file(app_name, f"[{datetime.now().strftime('%H:%M:%S')}] {error_msg}")
            break

def start_streamlit_app(app_name, script_path, port):
    """Start a Streamlit application."""
    try:
        if processes[app_name]['process'] is not None:
            return False, "Application is already running"
        
        # Check if the file exists
        if not os.path.exists(script_path):
            return False, f"File not found: {script_path}"
        
        # Clear the previous log file
        log_file_path = LOG_DIR / f"{app_name}.log"
        if log_file_path.exists():
            log_file_path.unlink()
        
        # Create startup log entry
        start_msg = f"[{datetime.now().strftime('%H:%M:%S')}] Starting {app_name} application..."
        write_log_to_file(app_name, start_msg)
        
        cmd = [
            sys.executable, '-m', 'streamlit', 'run',
            script_path,
            '--server.port', str(port),
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false',
            # '--logger.level', 'debug',  # increase log verbosity
            '--logger.level', 'info',
            '--server.enableCORS', 'false'
        ]
        
        # Set environment variables to ensure UTF-8 encoding and reduce buffering
        env = os.environ.copy()
        env.update({
            'PYTHONIOENCODING': 'utf-8',
            'PYTHONUTF8': '1',
            'LANG': 'en_US.UTF-8',
            'LC_ALL': 'en_US.UTF-8',
            'PYTHONUNBUFFERED': '1',  # Disable Python buffering
            'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false'
        })
        
        # Use current working directory instead of script directory
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,  # No buffering
            universal_newlines=False,
            cwd=os.getcwd(),
            env=env,
            encoding=None,  # Handle encoding manually
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        )
        
        processes[app_name]['process'] = process
        processes[app_name]['status'] = 'starting'
        processes[app_name]['output'] = []
        processes[app_name]['healthcheck_started_at'] = time.time()
        
        # Start the output reading thread
        output_thread = threading.Thread(
            target=read_process_output,
            args=(process, app_name),
            daemon=True
        )
        output_thread.start()
        
        return True, f"{app_name} application starting..."
        
    except Exception as e:
        error_msg = f"Startup failed: {str(e)}"
        write_log_to_file(app_name, f"[{datetime.now().strftime('%H:%M:%S')}] {error_msg}")
        return False, error_msg

def stop_streamlit_app(app_name):
    """Stop a Streamlit application."""
    try:
        process = processes[app_name]['process']
        if process is None:
            _log_shutdown_step(f"{app_name} not running, skipping stop")
            return False, "Application is not running"
        
        try:
            pid = process.pid
        except Exception:
            pid = 'unknown'

        _log_shutdown_step(f"Stopping {app_name} (pid={pid})")
        process.terminate()
        
        # Wait for process to exit
        try:
            process.wait(timeout=5)
            _log_shutdown_step(f"{app_name} exited, returncode={process.returncode}")
        except subprocess.TimeoutExpired:
            _log_shutdown_step(f"{app_name} termination timed out, attempting force kill (pid={pid})")
            process.kill()
            process.wait()
            _log_shutdown_step(f"{app_name} force killed, returncode={process.returncode}")
        
        processes[app_name]['process'] = None
        processes[app_name]['status'] = 'stopped'
        processes[app_name]['healthcheck_started_at'] = None
        
        return True, f"{app_name} application stopped"
        
    except Exception as e:
        _log_shutdown_step(f"{app_name} stop failed: {e}")
        return False, f"Stop failed: {str(e)}"

HEALTHCHECK_PATH = "/_stcore/health"
HEALTHCHECK_PROXIES = {'http': None, 'https': None}
HEALTHCHECK_GRACE_SECONDS = 15


def _build_healthcheck_url(port):
    return f"http://127.0.0.1:{port}{HEALTHCHECK_PATH}"


def _healthcheck_grace_active(app_name: str) -> bool:
    started_at = processes.get(app_name, {}).get('healthcheck_started_at')
    if not started_at:
        return False
    return (time.time() - started_at) < HEALTHCHECK_GRACE_SECONDS


def _log_healthcheck_failure(app_name: str, exc: Exception):
    if _healthcheck_grace_active(app_name):
        logger.debug(f"Starting {app_name}, please wait")
        return
    logger.warning(f"{app_name} health check failed: {exc}")


def check_app_status():
    """Check the status of all applications."""
    for app_name, info in processes.items():
        if info['process'] is not None:
            if info['process'].poll() is None:
                # Process is still running, check if port is accessible
                try:
                    response = requests.get(
                        _build_healthcheck_url(info['port']),
                        timeout=2,
                        proxies=HEALTHCHECK_PROXIES
                    )
                    if response.status_code == 200:
                        info['status'] = 'running'
                    else:
                        info['status'] = 'starting'
                except Exception as exc:
                    _log_healthcheck_failure(app_name, exc)
                    info['status'] = 'starting'
            else:
                # Process has exited
                info['process'] = None
                info['status'] = 'stopped'
                info['healthcheck_started_at'] = None

def wait_for_app_startup(app_name, max_wait_time=90):
    """Wait for an application to finish starting up."""
    import time
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        info = processes[app_name]
        if info['process'] is None:
            return False, "Process has stopped"
        
        if info['process'].poll() is not None:
            return False, "Process startup failed"
        
        try:
            response = requests.get(
                _build_healthcheck_url(info['port']),
                timeout=2,
                proxies=HEALTHCHECK_PROXIES
            )
            if response.status_code == 200:
                info['status'] = 'running'
                return True, "Startup successful"
        except Exception as exc:
            _log_healthcheck_failure(app_name, exc)

        time.sleep(1)

    return False, "Startup timed out"

def cleanup_processes():
    """Clean up all processes."""
    _log_shutdown_step("Starting serial cleanup of child processes")
    for app_name in STREAMLIT_SCRIPTS:
        stop_streamlit_app(app_name)

    processes['forum']['status'] = 'stopped'
    try:
        stop_forum_engine()
    except Exception:  # pragma: no cover
        logger.exception("Failed to stop ForumEngine")
    _log_shutdown_step("Child process cleanup complete")
    _set_system_state(started=False, starting=False)

def cleanup_processes_concurrent(timeout: float = 6.0):
    """Concurrently clean up all child processes, force-killing any survivors after timeout."""
    _log_shutdown_step(f"Starting concurrent cleanup of child processes (timeout {timeout}s)")
    _log_shutdown_step("Only terminating child processes started and tracked by this console, no port scanning")
    running_before = _describe_running_children()
    if running_before:
        _log_shutdown_step("Currently alive child processes: " + ", ".join(running_before))
    else:
        _log_shutdown_step("No alive child processes detected, shutdown signals will still be sent")

    threads = []

    # Concurrently shut down Streamlit child processes
    for app_name in STREAMLIT_SCRIPTS:
        t = threading.Thread(target=stop_streamlit_app, args=(app_name,), daemon=True)
        threads.append(t)
        t.start()

    # Concurrently shut down ForumEngine
    forum_thread = threading.Thread(target=stop_forum_engine, daemon=True)
    threads.append(forum_thread)
    forum_thread.start()

    # Wait for all threads to complete, up to timeout seconds
    end_time = time.time() + timeout
    for t in threads:
        remaining = end_time - time.time()
        if remaining <= 0:
            break
        t.join(timeout=remaining)

    # Second check: force kill any still-alive child processes
    for app_name in STREAMLIT_SCRIPTS:
        proc = processes[app_name]['process']
        if proc is not None and proc.poll() is None:
            try:
                _log_shutdown_step(f"{app_name} process still alive, triggering second termination (pid={proc.pid})")
                proc.terminate()
                proc.wait(timeout=1)
            except Exception:
                try:
                    _log_shutdown_step(f"{app_name} second termination failed, attempting kill (pid={proc.pid})")
                    proc.kill()
                    proc.wait(timeout=1)
                except Exception:
                    logger.warning(f"{app_name} force exit failed, continuing shutdown")
            finally:
                processes[app_name]['process'] = None
                processes[app_name]['status'] = 'stopped'

    processes['forum']['status'] = 'stopped'
    _log_shutdown_step("Concurrent cleanup finished, marking system as not started")
    _set_system_state(started=False, starting=False)

def _schedule_server_shutdown(delay_seconds: float = 0.1):
    """Exit as soon as cleanup is complete to avoid blocking the current request."""
    def _shutdown():
        time.sleep(delay_seconds)
        try:
            socketio.stop()
        except Exception as exc:  # pragma: no cover
            logger.warning(f"SocketIO stop raised an exception, continuing exit: {exc}")
        _log_shutdown_step("SocketIO stop signal sent, main process will exit shortly")
        os._exit(0)

    threading.Thread(target=_shutdown, daemon=True).start()

def _start_async_shutdown(cleanup_timeout: float = 3.0):
    """Asynchronously trigger cleanup and force exit to avoid blocking HTTP requests."""
    _log_shutdown_step(f"Shutdown command received, starting async cleanup (timeout {cleanup_timeout}s)")

    def _force_exit():
        _log_shutdown_step("Shutdown timed out, triggering forced exit")
        os._exit(0)

    # Hard timeout guard — exits even if the cleanup thread raises an exception
    hard_timeout = cleanup_timeout + 2.0
    force_timer = threading.Timer(hard_timeout, _force_exit)
    force_timer.daemon = True
    force_timer.start()

    def _cleanup_and_exit():
        try:
            cleanup_processes_concurrent(timeout=cleanup_timeout)
        except Exception as exc:  # pragma: no cover
            logger.exception(f"Shutdown cleanup exception: {exc}")
        finally:
            _log_shutdown_step("Cleanup thread finished, scheduling main process exit")
            _schedule_server_shutdown(0.05)

    threading.Thread(target=_cleanup_and_exit, daemon=True).start()

# Register cleanup function
atexit.register(cleanup_processes)

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get the status of all applications."""
    check_app_status()
    return jsonify({
        app_name: {
            'status': info['status'],
            'port': info['port'],
            'output_lines': len(info['output'])
        }
        for app_name, info in processes.items()
    })

@app.route('/api/start/<app_name>')
def start_app(app_name):
    """Start the specified application."""
    if app_name not in processes:
        return jsonify({'success': False, 'message': 'Unknown application'})

    if app_name == 'forum':
        try:
            start_forum_engine()
            processes['forum']['status'] = 'running'
            return jsonify({'success': True, 'message': 'ForumEngine started'})
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to manually start ForumEngine")
            return jsonify({'success': False, 'message': f'ForumEngine startup failed: {exc}'})

    script_path = STREAMLIT_SCRIPTS.get(app_name)
    if not script_path:
        return jsonify({'success': False, 'message': 'This application does not support the start operation'})

    success, message = start_streamlit_app(
        app_name,
        script_path,
        processes[app_name]['port']
    )

    if success:
        # Wait for application to start
        startup_success, startup_message = wait_for_app_startup(app_name, 15)
        if not startup_success:
            message += f" but startup check failed: {startup_message}"
    
    return jsonify({'success': success, 'message': message})

@app.route('/api/stop/<app_name>')
def stop_app(app_name):
    """Stop the specified application."""
    if app_name not in processes:
        return jsonify({'success': False, 'message': 'Unknown application'})

    if app_name == 'forum':
        try:
            stop_forum_engine()
            processes['forum']['status'] = 'stopped'
            return jsonify({'success': True, 'message': 'ForumEngine stopped'})
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to manually stop ForumEngine")
            return jsonify({'success': False, 'message': f'ForumEngine stop failed: {exc}'})

    success, message = stop_streamlit_app(app_name)
    return jsonify({'success': success, 'message': message})

@app.route('/api/output/<app_name>')
def get_output(app_name):
    """Get application output."""
    if app_name not in processes:
        return jsonify({'success': False, 'message': 'Unknown application'})
    
    # Special handling for Forum Engine
    if app_name == 'forum':
        try:
            forum_log_content = read_log_from_file('forum')
            return jsonify({
                'success': True,
                'output': forum_log_content,
                'total_lines': len(forum_log_content)
            })
        except Exception as e:
            return jsonify({'success': False, 'message': f'Failed to read forum log: {str(e)}'})
    
    # Read full log from file
    output_lines = read_log_from_file(app_name)
    
    return jsonify({
        'success': True,
        'output': output_lines
    })

@app.route('/api/test_log/<app_name>')
def test_log(app_name):
    """Test log writing functionality."""
    if app_name not in processes:
        return jsonify({'success': False, 'message': 'Unknown application'})
    
    # Write test message
    test_msg = f"[{datetime.now().strftime('%H:%M:%S')}] Test log message - {datetime.now()}"
    write_log_to_file(app_name, test_msg)
    
    # Send via Socket.IO
    socketio.emit('console_output', {
        'app': app_name,
        'line': test_msg
    })
    
    return jsonify({
        'success': True,
        'message': f'Test message written to {app_name} log'
    })

@app.route('/api/forum/start')
def start_forum_monitoring_api():
    """Manually start the ForumEngine forum."""
    try:
        from ForumEngine.monitor import start_forum_monitoring
        success = start_forum_monitoring()
        if success:
            return jsonify({'success': True, 'message': 'ForumEngine forum started'})
        else:
            return jsonify({'success': False, 'message': 'ForumEngine forum startup failed'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to start forum: {str(e)}'})

@app.route('/api/forum/stop')
def stop_forum_monitoring_api():
    """Manually stop the ForumEngine forum."""
    try:
        from ForumEngine.monitor import stop_forum_monitoring
        stop_forum_monitoring()
        return jsonify({'success': True, 'message': 'ForumEngine forum stopped'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to stop forum: {str(e)}'})

@app.route('/api/forum/log')
def get_forum_log():
    """Get the contents of ForumEngine's forum.log."""
    try:
        forum_log_file = LOG_DIR / "forum.log"
        if not forum_log_file.exists():
            return jsonify({
                'success': True,
                'log_lines': [],
                'parsed_messages': [],
                'total_lines': 0
            })
        
        with open(forum_log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            lines = [line.rstrip('\n\r') for line in lines if line.strip()]
        
        # Parse each log line and extract conversation information
        parsed_messages = []
        for line in lines:
            parsed_message = parse_forum_log_line(line)
            if parsed_message:
                parsed_messages.append(parsed_message)
        
        return jsonify({
            'success': True,
            'log_lines': lines,
            'parsed_messages': parsed_messages,
            'total_lines': len(lines)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to read forum.log: {str(e)}'})

@app.route('/api/forum/log/history', methods=['POST'])
def get_forum_log_history():
    """Get forum history log (supports reading from a specified position)."""
    try:
        data = request.get_json()
        start_position = data.get('position', 0)  # Last received position from the client
        max_lines = data.get('max_lines', 1000)   # Maximum number of lines to return

        forum_log_file = LOG_DIR / "forum.log"
        if not forum_log_file.exists():
            return jsonify({
                'success': True,
                'log_lines': [],
                'position': 0,
                'has_more': False
            })

        with open(forum_log_file, 'r', encoding='utf-8', errors='ignore') as f:
            # Read from the specified position
            f.seek(start_position)
            lines = []
            line_count = 0

            for line in f:
                if line_count >= max_lines:
                    break
                line = line.rstrip('\n\r')
                if line.strip():
                    # Add timestamp
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    formatted_line = f"[{timestamp}] {line}"
                    lines.append(formatted_line)
                    line_count += 1

            # Record current position
            current_position = f.tell()

            # Check if there is more content
            f.seek(0, 2)  # Seek to end of file
            end_position = f.tell()
            has_more = current_position < end_position

        return jsonify({
            'success': True,
            'log_lines': lines,
            'position': current_position,
            'has_more': has_more
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to read forum history: {str(e)}'})

@app.route('/api/search', methods=['POST'])
def search():
    """Unified search interface."""
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'success': False, 'message': 'Search query cannot be empty'})
    
    # ForumEngine forum is already running in the background and will automatically detect search activity
    # logger.info("ForumEngine: Search request received, forum will automatically detect log changes")
    
    # Check which applications are currently running
    check_app_status()
    running_apps = [name for name, info in processes.items() if info['status'] == 'running']
    
    if not running_apps:
        return jsonify({'success': False, 'message': 'No running applications'})
    
    # Send search request to running applications
    results = {}
    api_ports = {'insight': 8501, 'media': 8502, 'query': 8503}
    
    for app_name in running_apps:
        try:
            api_port = api_ports[app_name]
            # Call the Streamlit application's API endpoint
            response = requests.post(
                f"http://localhost:{api_port}/api/search",
                json={'query': query},
                timeout=10
            )
            if response.status_code == 200:
                results[app_name] = response.json()
            else:
                results[app_name] = {'success': False, 'message': 'API call failed'}
        except Exception as e:
            results[app_name] = {'success': False, 'message': str(e)}
    
    # After the search completes, monitoring can be stopped or left running to capture subsequent processing logs
    # Here we let monitoring continue; users can manually stop it via other endpoints
    
    return jsonify({
        'success': True,
        'query': query,
        'results': results
    })


@app.route('/api/config', methods=['GET'])
def get_config():
    """Expose selected configuration values to the frontend."""
    try:
        config_values = read_config_values()
        return jsonify({'success': True, 'config': config_values})
    except Exception as exc:
        logger.exception("Failed to read config")
        return jsonify({'success': False, 'message': f'Failed to read config: {exc}'}), 500


@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration values and persist them to config.py."""
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict) or not payload:
        return jsonify({'success': False, 'message': 'Request body cannot be empty'}), 400

    updates = {}
    for key, value in payload.items():
        if key in CONFIG_KEYS:
            updates[key] = value if value is not None else ''

    if not updates:
        return jsonify({'success': False, 'message': 'No updatable configuration entries'}), 400

    try:
        write_config_values(updates)
        updated_config = read_config_values()
        return jsonify({'success': True, 'config': updated_config})
    except Exception as exc:
        logger.exception("Failed to update config")
        return jsonify({'success': False, 'message': f'Failed to update config: {exc}'}), 500


@app.route('/api/system/status')
def get_system_status():
    """Return the system startup status."""
    state = _get_system_state()
    return jsonify({
        'success': True,
        'started': state['started'],
        'starting': state['starting']
    })


@app.route('/api/system/start', methods=['POST'])
def start_system():
    """Start the full system after receiving a request."""
    allowed, message = _prepare_system_start()
    if not allowed:
        return jsonify({'success': False, 'message': message}), 400

    try:
        success, logs, errors = initialize_system_components()
        if success:
            _set_system_state(started=True)
            return jsonify({'success': True, 'message': 'System started successfully', 'logs': logs})

        _set_system_state(started=False)
        return jsonify({
            'success': False,
            'message': 'System startup failed',
            'logs': logs,
            'errors': errors
        }), 500
    except Exception as exc:  # pragma: no cover - safety catch
        logger.exception("Exception occurred during system startup")
        _set_system_state(started=False)
        return jsonify({'success': False, 'message': f'System startup exception: {exc}'}), 500
    finally:
        _set_system_state(starting=False)

@app.route('/api/system/shutdown', methods=['POST'])
def shutdown_system():
    """Gracefully stop all components and shut down the current service process."""
    state = _get_system_state()
    if state['starting']:
        return jsonify({'success': False, 'message': 'System is starting/restarting, please wait'}), 400

    target_ports = [
        f"{name}:{info['port']}"
        for name, info in processes.items()
        if info.get('port')
    ]

    # If a shutdown is already in progress, return the currently alive child processes for frontend progress tracking
    if not _mark_shutdown_requested():
        running = _describe_running_children()
        detail = 'Shutdown command already issued, please wait...'
        if running:
            detail = f"Shutdown command issued, waiting for processes to exit: {', '.join(running)}"
        if target_ports:
            detail = f"{detail} (ports: {', '.join(target_ports)})"
        return jsonify({'success': True, 'message': detail, 'ports': target_ports})

    running = _describe_running_children()
    if running:
        _log_shutdown_step("Shutting down system, waiting for child processes to exit: " + ", ".join(running))
    else:
        _log_shutdown_step("Shutting down system, no alive child processes detected")

    try:
        _set_system_state(started=False, starting=False)
        _start_async_shutdown(cleanup_timeout=6.0)
        message = 'Shutdown command issued, stopping processes'
        if running:
            message = f"{message}: {', '.join(running)}"
        if target_ports:
            message = f"{message} (ports: {', '.join(target_ports)})"
        return jsonify({'success': True, 'message': message, 'ports': target_ports})
    except Exception as exc:  # pragma: no cover - safety catch
        logger.exception("Exception occurred during system shutdown")
        return jsonify({'success': False, 'message': f'System shutdown exception: {exc}'}), 500

@socketio.on('connect')
def handle_connect():
    """Client connected."""
    emit('status', 'Connected to Flask server')

@socketio.on('request_status')
def handle_status_request():
    """Request a status update."""
    check_app_status()
    emit('status_update', {
        app_name: {
            'status': info['status'],
            'port': info['port']
        }
        for app_name, info in processes.items()
    })

if __name__ == '__main__':
    # Read HOST and PORT from config
    from config import settings
    HOST = settings.HOST
    PORT = settings.PORT
    
    logger.info("Waiting for configuration confirmation, components will start after a frontend command...")
    logger.info(f"Flask server started, access at: http://{HOST}:{PORT}")
    
    try:
        socketio.run(app, host=HOST, port=PORT, debug=False)
    except KeyboardInterrupt:
        logger.info("\nShutting down application...")
        cleanup_processes()
        
    
