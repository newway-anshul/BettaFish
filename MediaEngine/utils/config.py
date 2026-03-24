"""
Configuration management module for the Media Engine (pydantic_settings style).
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, Literal


# Compute .env priority: current working directory first, then project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CWD_ENV: Path = Path.cwd() / ".env"
ENV_FILE: str = str(CWD_ENV if CWD_ENV.exists() else (PROJECT_ROOT / ".env"))

class Settings(BaseSettings):
    """
    Global configuration; supports automatic loading from .env and environment variables.
    Variable names stay uppercase to match the original config.py for smooth migration.
    """
    # ====================== Database Configuration ======================
    DB_HOST: str = Field("your_db_host", description="Database host, for example localhost or 127.0.0.1. We also provide convenient cloud database resource configuration with 100k+ daily data capacity and free application support. Contact: 670939375@qq.com. NOTE: To conduct data compliance review and service upgrades, the cloud database has suspended new application intake since October 1, 2025.")
    DB_PORT: int = Field(3306, description="Database port, default is 3306")
    DB_USER: str = Field("your_db_user", description="Database username")
    DB_PASSWORD: str = Field("your_db_password", description="Database password")
    DB_NAME: str = Field("your_db_name", description="Database name")
    DB_CHARSET: str = Field("utf8mb4", description="Database charset; utf8mb4 is recommended for emoji compatibility")
    DB_DIALECT: str = Field("mysql", description="Database dialect, such as 'mysql' or 'postgresql'. Used to support multiple database backends (for example SQLAlchemy; configure together with connection settings)")

    # ======================= LLM Settings =======================
    INSIGHT_ENGINE_API_KEY: str = Field(None, description="Insight Agent API key for the main LLM (Kimi recommended: https://platform.moonshot.cn/). You can change the API used by each LLM component; any provider compatible with OpenAI request format will work by defining KEY, BASE_URL, and MODEL_NAME. Important reminder: we strongly recommend applying for the suggested API and validating the default setup first before making custom changes.")
    INSIGHT_ENGINE_BASE_URL: Optional[str] = Field("https://api.moonshot.cn/v1", description="Insight Agent LLM BaseUrl; provider endpoint can be customized")
    INSIGHT_ENGINE_MODEL_NAME: str = Field("kimi-k2-0711-preview", description="Insight Agent LLM model name, such as kimi-k2-0711-preview")
    
    MEDIA_ENGINE_API_KEY: str = Field(None, description="Media Agent API key (Gemini recommended; this setup uses a proxy provider, but you can replace it with your own. Application URL: https://www.chataiapi.com/)")
    MEDIA_ENGINE_BASE_URL: Optional[str] = Field("https://www.chataiapi.com/v1", description="Media Agent LLM BaseUrl")
    MEDIA_ENGINE_MODEL_NAME: str = Field("gemini-2.5-pro", description="Media Agent LLM model name, such as gemini-2.5-pro")
    
    BOCHA_WEB_SEARCH_API_KEY: Optional[str] = Field(None, description="Bocha Web Search API Key")
    BOCHA_API_KEY: Optional[str] = Field(None, description="Bocha compatible key (alias)")
    
    SEARCH_TIMEOUT: int = Field(240, description="Search timeout (seconds)")
    SEARCH_CONTENT_MAX_LENGTH: int = Field(20000, description="Maximum content length used for prompts")
    MAX_REFLECTIONS: int = Field(2, description="Maximum number of reflection rounds")
    MAX_PARAGRAPHS: int = Field(5, description="Maximum number of paragraphs")
    
    MINDSPIDER_API_KEY: Optional[str] = Field(None, description="MindSpider API key")
    MINDSPIDER_BASE_URL: Optional[str] = Field("https://api.deepseek.com", description="MindSpider LLM BaseUrl")
    MINDSPIDER_MODEL_NAME: str = Field("deepseek-reasoner", description="MindSpider LLM model name, such as deepseek-reasoner")
    
    OUTPUT_DIR: str = Field("reports", description="Output directory")
    SAVE_INTERMEDIATE_STATES: bool = Field(True, description="Whether to save intermediate states")

    
    QUERY_ENGINE_API_KEY: str = Field(None, description="Query Agent API key (DeepSeek recommended: https://www.deepseek.com/)")
    QUERY_ENGINE_BASE_URL: Optional[str] = Field("https://api.deepseek.com", description="Query Agent LLM BaseUrl")
    QUERY_ENGINE_MODEL_NAME: str = Field("deepseek-reasoner", description="Query Agent LLM model, such as deepseek-reasoner")
    
    REPORT_ENGINE_API_KEY: str = Field(None, description="Report Agent API key (Gemini recommended; this setup uses a proxy provider, but you can replace it with your own. Application URL: https://www.chataiapi.com/)")
    REPORT_ENGINE_BASE_URL: Optional[str] = Field("https://www.chataiapi.com/v1", description="Report Agent LLM BaseUrl")
    REPORT_ENGINE_MODEL_NAME: str = Field("gemini-2.5-pro", description="Report Agent LLM model, such as gemini-2.5-pro")
    
    FORUM_HOST_API_KEY: str = Field(None, description="Forum Host API key (latest Qwen3 model; this setup uses SiliconFlow platform. Application URL: https://cloud.siliconflow.cn/)")
    FORUM_HOST_BASE_URL: Optional[str] = Field("https://api.siliconflow.cn/v1", description="Forum Host LLM BaseUrl")
    FORUM_HOST_MODEL_NAME: str = Field("Qwen/Qwen3-235B-A22B-Instruct-2507", description="Forum Host LLM model name, such as Qwen/Qwen3-235B-A22B-Instruct-2507")
    
    KEYWORD_OPTIMIZER_API_KEY: str = Field(None, description="SQL Keyword Optimizer API key (small-parameter Qwen3 model; this setup uses SiliconFlow platform. Application URL: https://cloud.siliconflow.cn/)")
    KEYWORD_OPTIMIZER_BASE_URL: Optional[str] = Field("https://api.siliconflow.cn/v1", description="Keyword Optimizer BaseUrl")
    KEYWORD_OPTIMIZER_MODEL_NAME: str = Field("Qwen/Qwen3-30B-A3B-Instruct-2507", description="Keyword Optimizer LLM model name, such as Qwen/Qwen3-30B-A3B-Instruct-2507")

    # ================== Network Tool Configuration ====================
    TAVILY_API_KEY: str = Field(None, description="Tavily API key (application URL: https://www.tavily.com/) used for Tavily web search")
    
    SEARCH_TOOL_TYPE: Literal["AnspireAPI", "BochaAPI"] = Field("AnspireAPI", description="Web search tool type. Supports BochaAPI or AnspireAPI; default is AnspireAPI")
    BOCHA_BASE_URL: Optional[str] = Field("https://api.bochaai.com/v1/ai-search", description="Bocha AI search BaseUrl or Bocha web search BaseUrl")
    BOCHA_WEB_SEARCH_API_KEY: Optional[str] = Field(None, description="Bocha API key (application URL: https://open.bochaai.com/) used for Bocha search")
    # Anspire AI Search API (application URL: https://open.anspire.cn/)
    ANSPIRE_BASE_URL: Optional[str] = Field("https://plugin.anspire.cn/api/ntsearch/search", description="Anspire AI search BaseUrl")
    ANSPIRE_API_KEY: Optional[str] = Field(None, description="Anspire AI Search API key (application URL: https://open.anspire.cn/) used for Anspire search")

    class Config:
        env_file = ENV_FILE
        env_prefix = ""
        case_sensitive = False
        extra = "allow"


settings = Settings()
