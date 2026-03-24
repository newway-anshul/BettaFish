"""
Deep Search Agent
A framework-free deep search AI agent implementation
"""

from .agent import DeepSearchAgent, AnspireSearchAgent, create_agent
from .utils.config import Settings

__version__ = "1.0.0"
__author__ = "Deep Search Agent Team"

__all__ = ["DeepSearchAgent", "AnspireSearchAgent", "create_agent", "Settings"]
