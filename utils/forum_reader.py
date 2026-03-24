"""
Forum log reader utility.
Used to read the latest HOST messages from forum.log.
"""

import re
from pathlib import Path
from typing import Optional, List, Dict
from loguru import logger

def get_latest_host_speech(log_dir: str = "logs") -> Optional[str]:
    """Get the latest HOST message from forum.log.

    Args:
        log_dir: Log directory path.

    Returns:
        Latest HOST message content, or None if not found.
    """
    try:
        forum_log_path = Path(log_dir) / "forum.log"
        
        if not forum_log_path.exists():
            logger.debug("forum.log does not exist")
            return None
            
        with open(forum_log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Search backward for the latest HOST message.
        host_speech = None
        for line in reversed(lines):
            # Match format: [time] [HOST] content
            match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]\s*\[HOST\]\s*(.+)', line)
            if match:
                _, content = match.groups()
                # Restore escaped newlines.
                host_speech = content.replace('\\n', '\n').strip()
                break
        
        if host_speech:
            logger.info(f"Found latest HOST message, length: {len(host_speech)} characters")
        else:
            logger.debug("No HOST message found")
            
        return host_speech
        
    except Exception as e:
        logger.error(f"Failed to read forum.log: {str(e)}")
        return None


def get_all_host_speeches(log_dir: str = "logs") -> List[Dict[str, str]]:
    """Get all HOST messages from forum.log.

    Args:
        log_dir: Log directory path.

    Returns:
        A list of HOST messages. Each item includes timestamp and content.
    """
    try:
        forum_log_path = Path(log_dir) / "forum.log"
        
        if not forum_log_path.exists():
            logger.debug("forum.log does not exist")
            return []
            
        with open(forum_log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        host_speeches = []
        for line in lines:
            # Match format: [time] [HOST] content
            match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]\s*\[HOST\]\s*(.+)', line)
            if match:
                timestamp, content = match.groups()
                # Restore escaped newlines.
                content = content.replace('\\n', '\n').strip()
                host_speeches.append({
                    'timestamp': timestamp,
                    'content': content
                })
        
        logger.info(f"Found {len(host_speeches)} HOST messages")
        return host_speeches
        
    except Exception as e:
        logger.error(f"Failed to read forum.log: {str(e)}")
        return []


def get_recent_agent_speeches(log_dir: str = "logs", limit: int = 5) -> List[Dict[str, str]]:
    """Get recent agent messages from forum.log (excluding HOST).

    Args:
        log_dir: Log directory path.
        limit: Maximum number of messages to return.

    Returns:
        List of recent agent messages.
    """
    try:
        forum_log_path = Path(log_dir) / "forum.log"
        
        if not forum_log_path.exists():
            return []
            
        with open(forum_log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        agent_speeches = []
        for line in reversed(lines):  # Read from latest to oldest.
            # Match format: [time] [AGENT_NAME] content
            match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]\s*\[(INSIGHT|MEDIA|QUERY)\]\s*(.+)', line)
            if match:
                timestamp, agent, content = match.groups()
                # Restore escaped newlines.
                content = content.replace('\\n', '\n').strip()
                agent_speeches.append({
                    'timestamp': timestamp,
                    'agent': agent,
                    'content': content
                })
                if len(agent_speeches) >= limit:
                    break
        
        agent_speeches.reverse()  # Restore chronological order.
        return agent_speeches
        
    except Exception as e:
        logger.error(f"Failed to read forum.log: {str(e)}")
        return []


def format_host_speech_for_prompt(host_speech: str) -> str:
    """Format a HOST message for prompt injection.

    Args:
        host_speech: HOST message content.

    Returns:
        Formatted content string.
    """
    if not host_speech:
        return ""
    
    return f"""
### Latest Forum Host Summary
Below is the forum host's latest summary and guidance based on the agent discussion. Please refer to the viewpoints and suggestions:

{host_speech}

---
"""
