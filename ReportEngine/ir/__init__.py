"""
Report Engine executable JSON contract (IR) definitions and validation utilities.

This module exposes a unified Schema text and validator, shared by prompts, chapter
generation, and the final stitching pipeline to ensure structural consistency
from LLM output to rendered artifacts.
"""

from .schema import (
    IR_VERSION,
    CHAPTER_JSON_SCHEMA,
    CHAPTER_JSON_SCHEMA_TEXT,
    ALLOWED_BLOCK_TYPES,
    ALLOWED_INLINE_MARKS,
    ENGINE_AGENT_TITLES,
)
from .validator import IRValidator

__all__ = [
    "IR_VERSION",
    "CHAPTER_JSON_SCHEMA",
    "CHAPTER_JSON_SCHEMA_TEXT",
    "ALLOWED_BLOCK_TYPES",
    "ALLOWED_INLINE_MARKS",
    "ENGINE_AGENT_TITLES",
    "IRValidator",
]
