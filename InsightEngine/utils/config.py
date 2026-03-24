"""
Configuration management module for the Insight Engine.
Handles environment variables and config file parameters.
"""

import os
from dataclasses import dataclass
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from loguru import logger

class Settings(BaseSettings):
    INSIGHT_ENGINE_API_KEY: Optional[str] = Field(None, description="Insight Engine LLM API key")
    INSIGHT_ENGINE_BASE_URL: Optional[str] = Field(None, description="Insight Engine LLM base URL (optional)")
    INSIGHT_ENGINE_MODEL_NAME: Optional[str] = Field(None, description="Insight Engine LLM model name")
    INSIGHT_ENGINE_PROVIDER: Optional[str] = Field(None, description="Insight Engine model provider (deprecated)")
    DB_HOST: Optional[str] = Field(None, description="Database host")
    DB_USER: Optional[str] = Field(None, description="Database username")
    DB_PASSWORD: Optional[str] = Field(None, description="Database password")
    DB_NAME: Optional[str] = Field(None, description="Database name")
    DB_PORT: int = Field(3306, description="Database port")
    DB_CHARSET: str = Field("utf8mb4", description="Database charset")
    DB_DIALECT: Optional[str] = Field("mysql", description="Database dialect, e.g. mysql or postgresql (SQLAlchemy backend selection)")
    MAX_REFLECTIONS: int = Field(3, description="Maximum number of reflections")
    MAX_PARAGRAPHS: int = Field(6, description="Maximum number of paragraphs")
    SEARCH_TIMEOUT: int = Field(240, description="Timeout for a single search request")
    MAX_CONTENT_LENGTH: int = Field(500000, description="Maximum content length for search")
    DEFAULT_SEARCH_HOT_CONTENT_LIMIT: int = Field(100, description="Default maximum count for trending content")
    DEFAULT_SEARCH_TOPIC_GLOBALLY_LIMIT_PER_TABLE: int = Field(50, description="Maximum count per table for global topic search")
    DEFAULT_SEARCH_TOPIC_BY_DATE_LIMIT_PER_TABLE: int = Field(100, description="Maximum count per table for date-based topic search")
    DEFAULT_GET_COMMENTS_FOR_TOPIC_LIMIT: int = Field(500, description="Maximum comment count per topic")
    DEFAULT_SEARCH_TOPIC_ON_PLATFORM_LIMIT: int = Field(200, description="Maximum count for platform-specific topic search")
    MAX_SEARCH_RESULTS_FOR_LLM: int = Field(0, description="Maximum search results to pass to LLM")
    MAX_HIGH_CONFIDENCE_SENTIMENT_RESULTS: int = Field(0, description="Maximum high-confidence sentiment analysis results")
    OUTPUT_DIR: str = Field("reports", description="Output directory path")
    SAVE_INTERMEDIATE_STATES: bool = Field(True, description="Whether to save intermediate states")

    class Config:
        env_file = ".env"
        env_prefix = ""
        case_sensitive = False
        extra = "allow"

settings = Settings()