# -*- coding: utf-8 -*-
"""
Store database connection settings and API keys.
"""

from pydantic_settings import BaseSettings
from typing import Optional
from pydantic import Field
from pathlib import Path

# Compute .env lookup priority: current working directory first, then project root (parent of MindSpider)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CWD_ENV: Path = Path.cwd() / ".env"
ENV_FILE: str = str(CWD_ENV if CWD_ENV.exists() else (PROJECT_ROOT / ".env"))

class Settings(BaseSettings):
    """Global configuration loaded from environment variables and .env, supporting unified MySQL/PostgreSQL naming."""
    DB_DIALECT: str = Field("mysql", description="Database type. Supported values: 'mysql' or 'postgresql'")
    DB_HOST: str = Field("your_host", description="Database host name or IP address")
    DB_PORT: int = Field(3306, description="Database port")
    DB_USER: str = Field("your_username", description="Database username")
    DB_PASSWORD: str = Field("your_password", description="Database password")
    DB_NAME: str = Field("mindspider", description="Database name")
    DB_CHARSET: str = Field("utf8mb4", description="Database charset")
    MINDSPIDER_API_KEY: Optional[str] = Field(None, description="MINDSPIDER API key")
    MINDSPIDER_BASE_URL: Optional[str] = Field("https://api.deepseek.com", description="MINDSPIDER API base URL. deepseek-chat typically uses https://api.deepseek.com")
    MINDSPIDER_MODEL_NAME: Optional[str] = Field("deepseek-chat", description="MINDSPIDER API model name, recommended: deepseek-chat")

    class Config:
        env_file = ENV_FILE
        env_prefix = ""
        case_sensitive = False
        extra = "allow"

settings = Settings()
