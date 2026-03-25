"""
Report Engine core utility collection.

This package encapsulates three core capabilities: template slicing,
chapter storage, and chapter stitching. All upper-level nodes reuse
these utilities to ensure structural consistency.
"""

from .template_parser import TemplateSection, parse_template_sections
from .chapter_storage import ChapterStorage
from .stitcher import DocumentComposer

__all__ = [
    "TemplateSection",
    "parse_template_sections",
    "ChapterStorage",
    "DocumentComposer",
]
