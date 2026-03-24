"""
Query Engine configuration management module

This module uses pydantic-settings to manage Query Engine settings,
with automatic loading from environment variables and .env files.
Data model definition location:
- This file - configuration model definitions
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from loguru import logger


# Resolve .env priority: current working directory first, then project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CWD_ENV: Path = Path.cwd() / ".env"
ENV_FILE: str = str(CWD_ENV if CWD_ENV.exists() else (PROJECT_ROOT / ".env"))


class Settings(BaseSettings):
    """
    Global Query Engine settings with automatic loading from .env and environment variables.
    Variable names remain uppercase to match the original config.py for smooth migration.
    """
    
    # ======================= LLM Settings =======================
    QUERY_ENGINE_API_KEY: str = Field(..., description="Query Engine LLM API key for the primary LLM. You can change the API used by each component-level LLM. As long as it is compatible with the OpenAI request format and KEY, BASE_URL, and MODEL_NAME are defined, it will work.")
    QUERY_ENGINE_BASE_URL: Optional[str] = Field(None, description="Query Engine LLM API base URL; custom provider APIs are supported")
    QUERY_ENGINE_MODEL_NAME: str = Field(..., description="Query Engine LLM model name")
    QUERY_ENGINE_PROVIDER: Optional[str] = Field(None, description="Query Engine LLM provider (compatibility field)")
    
    # ================== Network Tool Settings ====================
    TAVILY_API_KEY: str = Field(..., description="Tavily API key (signup: https://www.tavily.com/) used for Tavily web search")
    
    # ================== Search Parameter Settings ====================
    SEARCH_TIMEOUT: int = Field(240, description="Search timeout (seconds)")
    SEARCH_CONTENT_MAX_LENGTH: int = Field(20000, description="Maximum content length used in prompts")
    MAX_REFLECTIONS: int = Field(2, description="Maximum number of reflection rounds")
    MAX_PARAGRAPHS: int = Field(5, description="Maximum number of paragraphs")
    MAX_SEARCH_RESULTS: int = Field(20, description="Maximum number of search results")
    
    # ================== Output Settings ====================
    OUTPUT_DIR: str = Field("reports", description="Output directory")
    SAVE_INTERMEDIATE_STATES: bool = Field(True, description="Whether to save intermediate states")
    
    class Config:
        env_file = ENV_FILE
        env_prefix = ""
        case_sensitive = False
        extra = "allow"


# Create global settings instance
settings = Settings()

def print_config(config: Settings):
    """
    Print configuration details
    
    Args:
        config: Settings object
    """
    message = ""
    message += "=== Query Engine Configuration ===\n"
    message += f"LLM Model: {config.QUERY_ENGINE_MODEL_NAME}\n"
    message += f"LLM Base URL: {config.QUERY_ENGINE_BASE_URL or '(default)'}\n"
    message += f"Tavily API Key: {'Configured' if config.TAVILY_API_KEY else 'Not configured'}\n"
    message += f"Search timeout: {config.SEARCH_TIMEOUT} seconds\n"
    message += f"Max content length: {config.SEARCH_CONTENT_MAX_LENGTH}\n"
    message += f"Max reflection rounds: {config.MAX_REFLECTIONS}\n"
    message += f"Max paragraphs: {config.MAX_PARAGRAPHS}\n"
    message += f"Max search results: {config.MAX_SEARCH_RESULTS}\n"
    message += f"Output directory: {config.OUTPUT_DIR}\n"
    message += f"Save intermediate states: {config.SAVE_INTERMEDIATE_STATES}\n"
    message += f"LLM API Key: {'Configured' if config.QUERY_ENGINE_API_KEY else 'Not configured'}\n"
    message += "========================\n"
    logger.info(message)
