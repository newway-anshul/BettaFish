"""
Report Engine LLM submodule.

Currently, it mainly exposes the OpenAI-compatible `LLMClient` wrapper.
"""

from .base import LLMClient

__all__ = ["LLMClient"]
