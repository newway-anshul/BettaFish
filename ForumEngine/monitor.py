"""
Log Monitor - Real-time monitoring of SummaryNode outputs in three log files
"""

import os
import time
import threading
from pathlib import Path
from datetime import datetime
import re
import json
from typing import Dict, Optional, List
from threading import Lock
from loguru import logger

# Import Forum Host module
try:
    from .llm_host import generate_host_speech
    HOST_AVAILABLE = True
except ImportError:
    logger.exception("ForumEngine: Forum host module not found, running in monitor-only mode")
    HOST_AVAILABLE = False

class LogMonitor:
    """Intelligent log monitor based on file change detection"""
   
    def __init__(self, log_dir: str = "logs"):
        """Initialize the log monitor"""
        self.log_dir = Path(log_dir)
        self.forum_log_file = self.log_dir / "forum.log"
       
        # Log files to monitor
        self.monitored_logs = {
            'insight': self.log_dir / 'insight.log',
            'media': self.log_dir / 'media.log',
            'query': self.log_dir / 'query.log'
        }
       
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.file_positions = {}  # Record the read position of each file
        self.file_line_counts = {}  # Record the line count of each file
        self.is_searching = False  # Whether a search is currently active
        self.search_inactive_count = 0  # Search inactivity counter
        self.write_lock = Lock()  # Write lock to prevent concurrent write conflicts
        
        # Host-related state
        self.agent_speeches_buffer = []  # Agent speech buffer
        self.host_speech_threshold = 5  # Trigger host speech every 5 agent speeches
        self.is_host_generating = False  # Whether host is currently generating speech
       
        # Target node identification patterns
        # 1. Class name (may be included in old format)
        # 2. Full module path (actual log format, includes engine prefix)
        # 3. Partial module path (for compatibility)
        # 4. Key identifying text
        self.target_node_patterns = [
            'FirstSummaryNode',  # Class name
            'ReflectionSummaryNode',  # Class name
            'InsightEngine.nodes.summary_node',  # InsightEngine full path
            'MediaEngine.nodes.summary_node',  # MediaEngine full path
            'QueryEngine.nodes.summary_node',  # QueryEngine full path
            'nodes.summary_node',  # Module path (compatibility, for partial matching)
            'Generating first paragraph summary',  # FirstSummaryNode identifier
            'Generating reflection summary',  # ReflectionSummaryNode identifier
        ]
        
        # Multi-line content capture state
        self.capturing_json = {}  # JSON capture state for each app
        self.json_buffer = {}     # JSON buffer for each app
        self.json_start_line = {} # JSON start line for each app
        self.in_error_block = {}  # Whether each app is inside an ERROR block
       
        # Ensure logs directory exists
        self.log_dir.mkdir(exist_ok=True)
   
    def clear_forum_log(self):
        """Clear the forum.log file"""
        try:
            if self.forum_log_file.exists():
                self.forum_log_file.unlink()
           
            # Create a new forum.log file and write the start marker
            start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # Use write_to_forum_log to write the start marker to ensure consistent format
            with open(self.forum_log_file, 'w', encoding='utf-8') as f:
                pass  # Create empty file first
            self.write_to_forum_log(f"=== ForumEngine Monitoring Started - {start_time} ===", "SYSTEM")
               
            logger.info(f"ForumEngine: forum.log has been cleared and initialized")
            
            # Reset JSON capture state
            self.capturing_json = {}
            self.json_buffer = {}
            self.json_start_line = {}
            self.in_error_block = {}
            
            # Reset host-related state
            self.agent_speeches_buffer = []
            self.is_host_generating = False
           
        except Exception as e:
            logger.exception(f"ForumEngine: Failed to clear forum.log: {e}")
   
    def write_to_forum_log(self, content: str, source: str = None):
        """Write content to forum.log (thread-safe)"""
        try:
            with self.write_lock:  # Use lock to ensure thread safety
                with open(self.forum_log_file, 'a', encoding='utf-8') as f:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    # Convert actual newlines in content to \n strings so the entire record stays on one line
                    content_one_line = content.replace('\n', '\\n').replace('\r', '\\r')
                    # If a source tag is provided, add it after the timestamp
                    if source:
                        f.write(f"[{timestamp}] [{source}] {content_one_line}\n")
                    else:
                        f.write(f"[{timestamp}] {content_one_line}\n")
                    f.flush()
        except Exception as e:
            logger.exception(f"ForumEngine: Failed to write to forum.log: {e}")
    
    def get_log_level(self, line: str) -> Optional[str]:
        """Detect the log level of a log line (INFO/ERROR/WARNING/DEBUG etc.)
        
        Supports loguru format: YYYY-MM-DD HH:mm:ss.SSS | LEVEL | ...
        
        Returns:
            'INFO', 'ERROR', 'WARNING', 'DEBUG' or None (unrecognized)
        """
        # Check loguru format: YYYY-MM-DD HH:mm:ss.SSS | LEVEL | ...
        # Match pattern: | LEVEL | or | LEVEL     |
        match = re.search(r'\|\s*(INFO|ERROR|WARNING|DEBUG|TRACE|CRITICAL)\s*\|', line)
        if match:
            return match.group(1)
        return None
    
    def is_target_log_line(self, line: str) -> bool:
        """Check if a line is a target log line (SummaryNode)
        
        Supports multiple identification methods:
        1. Class name: FirstSummaryNode, ReflectionSummaryNode
        2. Full module path: InsightEngine.nodes.summary_node, MediaEngine.nodes.summary_node, QueryEngine.nodes.summary_node
        3. Partial module path: nodes.summary_node (for compatibility)
        4. Key identifying text: generating first paragraph summary, generating reflection summary
        
        Exclusion conditions:
        - ERROR level logs (error logs should not be recognized as target nodes)
        - Logs containing error keywords (JSON parse failed, JSON repair failed, etc.)
        """
        # Exclude ERROR level logs
        log_level = self.get_log_level(line)
        if log_level == 'ERROR':
            return False
        
        # Compatible with old check method
        if "| ERROR" in line or "| ERROR    |" in line:
            return False
        
        # Exclude logs containing error keywords
        error_keywords = ["JSON parse failed", "JSON repair failed", "Traceback", "File \""]
        for keyword in error_keywords:
            if keyword in line:
                return False
        
        # Check if the line contains a target node pattern
        for pattern in self.target_node_patterns:
            if pattern in line:
                return True
        return False
    
    def is_valuable_content(self, line: str) -> bool:
        """Determine whether the content is valuable (exclude short prompts and error messages)"""
        # If line contains "Cleaned output", consider it valuable
        if "Cleaned output" in line:
            return True
        
        # Exclude common short prompts and error messages
        exclude_patterns = [
            "JSON parse failed",
            "JSON repair failed",
            "Use cleaned text directly",
            "JSON parsed successfully",
            "Generated successfully",
            "Paragraph updated",
            "Generating",
            "Processing started",
            "Processing completed",
            "HOST speech read",
            "Failed to read HOST speech",
            "HOST speech not found",
            "Debug output",
            "Info record"
        ]
        
        for pattern in exclude_patterns:
            if pattern in line:
                return False
        
        # If the line is too short, also consider it not valuable
        # Remove timestamps: supports old format and new format
        clean_line = re.sub(r'\[\d{2}:\d{2}:\d{2}\]', '', line)
        clean_line = re.sub(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\s*\|\s*[A-Z]+\s*\|\s*[^|]+?\s*-\s*', '', clean_line)
        clean_line = clean_line.strip()
        if len(clean_line) < 30:  # Threshold can be adjusted
            return False
            
        return True
    
    def is_json_start_line(self, line: str) -> bool:
        """Determine if a line is a JSON start line"""
        return "Cleaned output: {" in line
    
    def is_json_end_line(self, line: str) -> bool:
        """Determine if a line is a JSON end line
        
        Only identifies pure end-marker lines that contain no log format information (timestamps etc.).
        If a line contains a timestamp, it should be cleaned first; returning False here indicates further processing is needed.
        """
        stripped = line.strip()
        
        # If the line contains a timestamp (old or new format), it is not a pure end line
        # Old format: [HH:MM:SS]
        if re.match(r'^\[\d{2}:\d{2}:\d{2}\]', stripped):
            return False
        # New format: YYYY-MM-DD HH:mm:ss.SSS
        if re.match(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}', stripped):
            return False
        
        # Line without a timestamp — check if it is a pure end marker
        if stripped == "}" or stripped == "] }":
            return True
        return False
    
    def extract_json_content(self, json_lines: List[str]) -> Optional[str]:
        """Extract and parse JSON content from multiple lines"""
        try:
            # Find the position where JSON starts
            json_start_idx = -1
            for i, line in enumerate(json_lines):
                if "Cleaned output: {" in line:
                    json_start_idx = i
                    break
            
            if json_start_idx == -1:
                return None
            
            # Extract the JSON portion
            first_line = json_lines[json_start_idx]
            json_start_pos = first_line.find("Cleaned output: {")
            if json_start_pos == -1:
                return None
            
            json_part = first_line[json_start_pos + len("Cleaned output: "):]
            
            # If the first line contains the complete JSON, process it directly
            if json_part.strip().endswith("}") and json_part.count("{") == json_part.count("}"):
                try:
                    json_obj = json.loads(json_part.strip())
                    return self.format_json_content(json_obj)
                except json.JSONDecodeError:
                    # Single-line JSON parse failed, try to repair
                    fixed_json = self.fix_json_string(json_part.strip())
                    if fixed_json:
                        try:
                            json_obj = json.loads(fixed_json)
                            return self.format_json_content(json_obj)
                        except json.JSONDecodeError:
                            pass
                    return None
            
            # Handle multi-line JSON
            json_text = json_part
            for line in json_lines[json_start_idx + 1:]:
                # Remove timestamps: supports old format [HH:MM:SS] and new loguru format (YYYY-MM-DD HH:mm:ss.SSS | LEVEL | ...)
                # Old format: [HH:MM:SS]
                clean_line = re.sub(r'^\[\d{2}:\d{2}:\d{2}\]\s*', '', line)
                # New format: remove loguru timestamp and level information
                # Format: YYYY-MM-DD HH:mm:ss.SSS | LEVEL | module:function:line -
                clean_line = re.sub(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\s*\|\s*[A-Z]+\s*\|\s*[^|]+?\s*-\s*', '', clean_line)
                json_text += clean_line
            
            # Attempt to parse JSON
            try:
                json_obj = json.loads(json_text.strip())
                return self.format_json_content(json_obj)
            except json.JSONDecodeError:
                # Multi-line JSON parse failed, try to repair
                fixed_json = self.fix_json_string(json_text.strip())
                if fixed_json:
                    try:
                        json_obj = json.loads(fixed_json)
                        return self.format_json_content(json_obj)
                    except json.JSONDecodeError:
                        pass
                return None
            
        except Exception as e:
            # For any other exception, do not log; return None directly
            return None
    
    def format_json_content(self, json_obj: dict) -> str:
        """Format JSON content into a readable form"""
        try:
            # Extract main content; prefer reflection summary, fallback to first summary
            content = None
            
            if "updated_paragraph_latest_state" in json_obj:
                content = json_obj["updated_paragraph_latest_state"]
            elif "paragraph_latest_state" in json_obj:
                content = json_obj["paragraph_latest_state"]
            
            # If content was found, return it directly (keep newlines as \n)
            if content:
                return content
            
            # If the expected fields were not found, return the full JSON as a string
            return f"Cleaned output: {json.dumps(json_obj, ensure_ascii=False, indent=2)}"
            
        except Exception as e:
            logger.exception(f"ForumEngine: Error formatting JSON: {e}")
            return f"Cleaned output: {json.dumps(json_obj, ensure_ascii=False, indent=2)}"

    def extract_node_content(self, line: str) -> Optional[str]:
        """Extract node content, removing prefixes such as timestamps and node names"""
        content = line
        
        # Remove timestamp portion: supports old format and new format
        # Old format: [HH:MM:SS]
        match_old = re.search(r'\[\d{2}:\d{2}:\d{2}\]\s*(.+)', content)
        if match_old:
            content = match_old.group(1).strip()
        else:
            # New format: YYYY-MM-DD HH:mm:ss.SSS | LEVEL | module:function:line -
            match_new = re.search(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\s*\|\s*[A-Z]+\s*\|\s*[^|]+?\s*-\s*(.+)', content)
            if match_new:
                content = match_new.group(1).strip()
        
        if not content:
            return line.strip()
        
        # Remove all bracket tags (including node names and app names)
        content = re.sub(r'^\[.*?\]\s*', '', content)
        
        # Continue removing possible multiple consecutive tags
        while re.match(r'^\[.*?\]\s*', content):
            content = re.sub(r'^\[.*?\]\s*', '', content)
        
        # Remove common prefixes (e.g. "First summary: ", "Reflection summary: ", etc.)
        prefixes_to_remove = [
            "First summary: ",
            "Reflection summary: ",
            "Cleaned output: "
        ]
        
        for prefix in prefixes_to_remove:
            if content.startswith(prefix):
                content = content[len(prefix):]
                break
        
        # Remove possible app name tags (not inside brackets)
        app_names = ['INSIGHT', 'MEDIA', 'QUERY']
        for app_name in app_names:
            # Remove standalone APP_NAME at line start
            content = re.sub(rf'^{app_name}\s+', '', content, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        return content.strip()
   
    def get_file_size(self, file_path: Path) -> int:
        """Get the file size"""
        try:
            return file_path.stat().st_size if file_path.exists() else 0
        except:
            return 0
   
    def get_file_line_count(self, file_path: Path) -> int:
        """Get the file line count"""
        try:
            if not file_path.exists():
                return 0
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except:
            return 0
   
    def read_new_lines(self, file_path: Path, app_name: str) -> List[str]:
        """Read new lines from the file"""
        new_lines = []
       
        try:
            if not file_path.exists():
                return new_lines
           
            current_size = self.get_file_size(file_path)
            last_position = self.file_positions.get(app_name, 0)
           
            # If the file shrank, it was likely cleared; restart from the beginning
            if current_size < last_position:
                last_position = 0
                # Reset JSON capture state
                self.capturing_json[app_name] = False
                self.json_buffer[app_name] = []
                self.in_error_block[app_name] = False
           
            if current_size > last_position:
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.seek(last_position)
                    new_content = f.read()
                    new_lines = new_content.split('\n')
                   
                        # Update position
                    self.file_positions[app_name] = f.tell()
                   
                        # Filter empty lines
                    new_lines = [line.strip() for line in new_lines if line.strip()]
                   
        except Exception as e:
                    logger.exception(f"ForumEngine: Failed to read {app_name} log: {e}")
       
        return new_lines
   
    def process_lines_for_json(self, lines: List[str], app_name: str) -> List[str]:
        """Process lines to capture multi-line JSON content

        Implements ERROR-block filtering: if an ERROR-level log appears,
        skip processing until the next INFO-level log appears.
        """
        captured_contents = []
        
        # Initialize state
        if app_name not in self.capturing_json:
            self.capturing_json[app_name] = False
            self.json_buffer[app_name] = []
        if app_name not in self.in_error_block:
            self.in_error_block[app_name] = False
        
        for line in lines:
            if not line.strip():
                continue
            
            # First, check log level and update ERROR-block state
            log_level = self.get_log_level(line)
            if log_level == 'ERROR':
                # Enter ERROR-block state on ERROR
                self.in_error_block[app_name] = True
                # If currently capturing JSON, stop and clear buffer immediately
                if self.capturing_json[app_name]:
                    self.capturing_json[app_name] = False
                    self.json_buffer[app_name] = []
                # Skip current line
                continue
            elif log_level == 'INFO':
                # Exit ERROR-block state on INFO
                self.in_error_block[app_name] = False
            # Keep current state for other levels (WARNING, DEBUG, etc.)
            
            # If inside ERROR block, skip all processing
            if self.in_error_block[app_name]:
                # If currently capturing JSON, stop and clear buffer immediately
                if self.capturing_json[app_name]:
                    self.capturing_json[app_name] = False
                    self.json_buffer[app_name] = []
                # Skip current line
                continue
                
            # Check whether this is a target-node line and JSON start marker
            is_target = self.is_target_log_line(line)
            is_json_start = self.is_json_start_line(line)
            
            # Only target-node (SummaryNode) JSON output should be captured
            # Filter out output from other nodes such as SearchNode
            if is_target and is_json_start:
                # Start capturing JSON (must be target node and include "Cleaned output: {")
                self.capturing_json[app_name] = True
                self.json_buffer[app_name] = [line]
                self.json_start_line[app_name] = line
                
                # Check whether this is single-line JSON
                if line.strip().endswith("}"):
                    # Single-line JSON, process immediately
                    content = self.extract_json_content([line])
                    if content:  # Only successfully parsed content is recorded
                        # Remove duplicate tags and normalize formatting
                        clean_content = self._clean_content_tags(content, app_name)
                        captured_contents.append(f"{clean_content}")
                    self.capturing_json[app_name] = False
                    self.json_buffer[app_name] = []
                    
            elif is_target and self.is_valuable_content(line):
                # Other valuable SummaryNode content (must be a valuable target-node line)
                clean_content = self._clean_content_tags(self.extract_node_content(line), app_name)
                captured_contents.append(f"{clean_content}")
                    
            elif self.capturing_json[app_name]:
                # Subsequent lines while JSON capture is active
                self.json_buffer[app_name].append(line)
                
                # Check whether JSON ended
                # First remove timestamp formatting, then test end marker
                cleaned_line = line.strip()
                # Remove old-format timestamp: [HH:MM:SS]
                cleaned_line = re.sub(r'^\[\d{2}:\d{2}:\d{2}\]\s*', '', cleaned_line)
                # Remove new-format timestamp: YYYY-MM-DD HH:mm:ss.SSS | LEVEL | module:function:line -
                cleaned_line = re.sub(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\s*\|\s*[A-Z]+\s*\|\s*[^|]+?\s*-\s*', '', cleaned_line)
                cleaned_line = cleaned_line.strip()
                
                # After cleaning, check end marker
                if cleaned_line == "}" or cleaned_line == "] }":
                    # JSON ended; process the full JSON payload
                    content = self.extract_json_content(self.json_buffer[app_name])
                    if content:  # Only successfully parsed content is recorded
                        # Remove duplicate tags and normalize formatting
                        clean_content = self._clean_content_tags(content, app_name)
                        captured_contents.append(f"{clean_content}")
                    
                    # Reset state
                    self.capturing_json[app_name] = False
                    self.json_buffer[app_name] = []
        
        return captured_contents
    
    def _trigger_host_speech(self):
        """Trigger host speech (synchronous execution)"""
        if not HOST_AVAILABLE or self.is_host_generating:
            return
        
        try:
            # Set generation flag
            self.is_host_generating = True
            
            # Get 5 speeches from the buffer
            recent_speeches = self.agent_speeches_buffer[:5]
            if len(recent_speeches) < 5:
                self.is_host_generating = False
                return
            
            logger.info("ForumEngine: Generating host speech...")
            
            # Generate host speech using the most recent 5 entries
            host_speech = generate_host_speech(recent_speeches)
            
            if host_speech:
                # Write host speech to forum.log
                self.write_to_forum_log(host_speech, "HOST")
                logger.info(f"ForumEngine: Host speech recorded")
                
                # Remove processed 5 entries
                self.agent_speeches_buffer = self.agent_speeches_buffer[5:]
            else:
                logger.error("ForumEngine: Failed to generate host speech")
            
            # Reset generation flag
            self.is_host_generating = False
                
        except Exception as e:
            logger.exception(f"ForumEngine: Error while triggering host speech: {e}")
            self.is_host_generating = False
    
    def _clean_content_tags(self, content: str, app_name: str) -> str:
        """Clean duplicate tags and redundant prefixes in content"""
        if not content:
            return content
            
        # First remove all possible tag formats ([INSIGHT], [MEDIA], [QUERY], etc.)
        # Use stronger cleanup patterns
        all_app_names = ['INSIGHT', 'MEDIA', 'QUERY']
        
        for name in all_app_names:
            # Remove [APP_NAME] format (case-insensitive)
            content = re.sub(rf'\[{name}\]\s*', '', content, flags=re.IGNORECASE)
            # Remove standalone APP_NAME format
            content = re.sub(rf'^{name}\s+', '', content, flags=re.IGNORECASE)
        
        # Remove any remaining bracketed tags
        content = re.sub(r'^\[.*?\]\s*', '', content)
        
        # Remove repeated whitespace
        content = re.sub(r'\s+', ' ', content)
        
        return content.strip()
   
    def monitor_logs(self):
        """Intelligently monitor log files"""
        logger.info("ForumEngine: Forum creation in progress...")
       
        # Initialize file line counts and positions as baseline
        for app_name, log_file in self.monitored_logs.items():
            self.file_line_counts[app_name] = self.get_file_line_count(log_file)
            self.file_positions[app_name] = self.get_file_size(log_file)
            self.capturing_json[app_name] = False
            self.json_buffer[app_name] = []
            self.in_error_block[app_name] = False
            # logger.info(f"ForumEngine: {app_name} baseline line count: {self.file_line_counts[app_name]}")
       
        while self.is_monitoring:
            try:
                # Detect changes in all three log files
                any_growth = False
                any_shrink = False
                captured_any = False
               
                # Process each log file independently
                for app_name, log_file in self.monitored_logs.items():
                    current_lines = self.get_file_line_count(log_file)
                    previous_lines = self.file_line_counts.get(app_name, 0)
                   
                    if current_lines > previous_lines:
                        any_growth = True
                        # Read newly added content immediately
                        new_lines = self.read_new_lines(log_file, app_name)
                       
                        # First check whether search should be triggered (once only)
                        if not self.is_searching:
                            for line in new_lines:
                                # Check target-node patterns (supports multiple formats)
                                if line.strip() and self.is_target_log_line(line):
                                    # Confirm this is first-summary node
                                    if 'FirstSummaryNode' in line or 'Generating first paragraph summary' in line:
                                        logger.info(f"ForumEngine: Detected first forum publication in {app_name}")
                                        self.is_searching = True
                                        self.search_inactive_count = 0
                                        # Clear forum.log and start a new session
                                        self.clear_forum_log()
                                        break  # One hit is enough; break out
                       
                        # Process all newly added content (if search is active)
                        if self.is_searching:
                            # Use the new processing logic
                            captured_contents = self.process_lines_for_json(new_lines, app_name)
                            
                            for content in captured_contents:
                                # Convert app_name to uppercase tag (e.g., insight -> INSIGHT)
                                source_tag = app_name.upper()
                                self.write_to_forum_log(content, source_tag)
                                # logger.info(f"ForumEngine: Captured - {content}")
                                captured_any = True
                                
                                # Add speech to buffer (formatted as full log line)
                                timestamp = datetime.now().strftime('%H:%M:%S')
                                log_line = f"[{timestamp}] [{source_tag}] {content}"
                                self.agent_speeches_buffer.append(log_line)
                                
                                # Check whether host speech should be triggered
                                if len(self.agent_speeches_buffer) >= self.host_speech_threshold and not self.is_host_generating:
                                    # Trigger host speech synchronously
                                    self._trigger_host_speech()
                   
                    elif current_lines < previous_lines:
                        any_shrink = True
                        # logger.info(f"ForumEngine: Detected {app_name} log shrink; resetting baseline")
                        # Reset file position to new file end
                        self.file_positions[app_name] = self.get_file_size(log_file)
                        # Reset JSON capture state
                        self.capturing_json[app_name] = False
                        self.json_buffer[app_name] = []
                        self.in_error_block[app_name] = False
                   
                    # Update line-count record
                    self.file_line_counts[app_name] = current_lines
               
                # Check whether the current search session should end
                if self.is_searching:
                    if any_shrink:
                        # Log shrank; end current search session and return to waiting state
                        # logger.info("ForumEngine: Log shrank, ending current search session and returning to waiting state")
                        self.is_searching = False
                        self.search_inactive_count = 0
                        # Reset host-related state
                        self.agent_speeches_buffer = []
                        self.is_host_generating = False
                        # Write end marker
                        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        self.write_to_forum_log(f"=== ForumEngine Forum Ended - {end_time} ===", "SYSTEM")
                        # logger.info("ForumEngine: Baseline reset; waiting for next FirstSummaryNode trigger")
                    elif not any_growth and not captured_any:
                        # No growth and no captured content; increment inactivity counter
                        self.search_inactive_count += 1
                        if self.search_inactive_count >= 7200:  # Auto-end after long inactivity
                            logger.info("ForumEngine: No activity for a long time, ending forum")
                            self.is_searching = False
                            self.search_inactive_count = 0
                            # Reset host-related state
                            self.agent_speeches_buffer = []
                            self.is_host_generating = False
                            # Write end marker
                            end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            self.write_to_forum_log(f"=== ForumEngine Forum Ended - {end_time} ===", "SYSTEM")
                    else:
                        self.search_inactive_count = 0  # Reset counter
               
                # Brief sleep
                time.sleep(1)
               
            except Exception as e:
                logger.exception(f"ForumEngine: Error during forum logging: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(2)
       
        logger.info("ForumEngine: Stopped monitoring forum log files")
   
    def start_monitoring(self):
        """Start intelligent monitoring"""
        if self.is_monitoring:
            logger.info("ForumEngine: Forum is already running")
            return False
       
        try:
            # Start monitoring
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self.monitor_logs, daemon=True)
            self.monitor_thread.start()
           
            logger.info("ForumEngine: Forum started")
            return True
           
        except Exception as e:
            logger.exception(f"ForumEngine: Failed to start forum: {e}")
            self.is_monitoring = False
            return False
   
    def stop_monitoring(self):
        """Stop monitoring"""
        if not self.is_monitoring:
            logger.info("ForumEngine: Forum is not running")
            return
       
        try:
            self.is_monitoring = False
           
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=2)
           
            # Write end marker
            end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.write_to_forum_log(f"=== ForumEngine Forum Ended - {end_time} ===", "SYSTEM")
           
            logger.info("ForumEngine: Forum stopped")
           
        except Exception as e:
            logger.exception(f"ForumEngine: Failed to stop forum: {e}")
   
    def get_forum_log_content(self) -> List[str]:
        """Get the content of forum.log"""
        try:
            if not self.forum_log_file.exists():
                return []
           
            with open(self.forum_log_file, 'r', encoding='utf-8') as f:
                return [line.rstrip('\n\r') for line in f.readlines()]
               
        except Exception as e:
            logger.exception(f"ForumEngine: Failed to read forum.log: {e}")
            return []

    def fix_json_string(self, json_text: str) -> str:
        """Fix common issues in JSON strings, especially unescaped double quotes"""
        try:
            # Try direct parsing; if successful, return the original text
            json.loads(json_text)
            return json_text
        except json.JSONDecodeError:
            pass
        
        # Fix unescaped double quotes
        # This uses a smarter approach specifically for quotes inside string values
        
        try:
            # Use a state-machine approach to repair JSON
            # Iterate characters and track whether we're inside a string value
            
            fixed_text = ""
            i = 0
            in_string = False
            escape_next = False
            
            while i < len(json_text):
                char = json_text[i]
                
                if escape_next:
                    # Handle escaped character
                    fixed_text += char
                    escape_next = False
                    i += 1
                    continue
                
                if char == '\\':
                    # Escape character
                    fixed_text += char
                    escape_next = True
                    i += 1
                    continue
                
                if char == '"' and not escape_next:
                    # Encountered a double quote
                    if in_string:
                        # Inside a string; inspect next non-whitespace character
                        # If next is colon/comma/brace, this quote closes the string
                        next_char_pos = i + 1
                        while next_char_pos < len(json_text) and json_text[next_char_pos].isspace():
                            next_char_pos += 1
                        
                        if next_char_pos < len(json_text):
                            next_char = json_text[next_char_pos]
                            if next_char in [':', ',', '}']:
                                # String ends here
                                in_string = False
                                fixed_text += char
                            else:
                                # Quote inside string; needs escaping
                                fixed_text += '\\"'
                        else:
                            # End of text; close string state
                            in_string = False
                            fixed_text += char
                    else:
                        # String starts
                        in_string = True
                        fixed_text += char
                else:
                    # Any other character
                    fixed_text += char
                
                i += 1
            
            # Try parsing repaired JSON
            try:
                json.loads(fixed_text)
                return fixed_text
            except json.JSONDecodeError:
                # Repair failed
                return None
                
        except Exception:
            return None

# Global monitor instance
_monitor_instance = None

def get_monitor() -> LogMonitor:
    """Get the global monitor instance"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = LogMonitor()
    return _monitor_instance

def start_forum_monitoring():
    """Start ForumEngine intelligent monitoring"""
    return get_monitor().start_monitoring()

def stop_forum_monitoring():
    """Stop ForumEngine monitoring"""
    get_monitor().stop_monitoring()

def get_forum_log():
    """Get forum.log content"""
    return get_monitor().get_forum_log_content()