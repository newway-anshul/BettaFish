"""
Report Engine state management module.

Exports ReportState and ReportMetadata for shared use by the Agent and Flask interface.
"""

from .state import ReportState, ReportMetadata

__all__ = ["ReportState", "ReportMetadata"]
