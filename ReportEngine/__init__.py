"""
Report Engine.

An intelligent AI report generation agent that aggregates Query/Media/Insight sub-engine
Markdown outputs and forum discussions into a structured HTML report.
"""

from .agent import ReportAgent, create_agent

__version__ = "1.0.0"
__author__ = "Report Engine Team"

__all__ = ["ReportAgent", "create_agent"]
