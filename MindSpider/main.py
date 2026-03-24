#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MindSpider - Main entry point for the AI crawler project.
Integrates the BroadTopicExtraction and DeepSentimentCrawling core modules.
"""

import os
import sys
import argparse
import difflib
import re
from datetime import date, datetime
from pathlib import Path
import subprocess
import asyncio
import pymysql
from pymysql.cursors import DictCursor
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy import inspect, text
from config import settings
from loguru import logger
from urllib.parse import quote_plus

# Add project root to import path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    import config
except ImportError:
    logger.error("Error: failed to import config.py")
    logger.error("Make sure config.py exists in the project root and contains database/API settings")
    sys.exit(1)

class MindSpider:
    """MindSpider main program."""
    
    def __init__(self):
        """Initialize MindSpider."""
        self.project_root = project_root
        self.broad_topic_path = self.project_root / "BroadTopicExtraction"
        self.deep_sentiment_path = self.project_root / "DeepSentimentCrawling"
        self.schema_path = self.project_root / "schema"
        
        logger.info("MindSpider AI crawler project")
        logger.info(f"Project path: {self.project_root}")
    
    def check_config(self) -> bool:
        """Validate base configuration."""
        logger.info("Checking base configuration...")
        
        # Check required settings
        required_configs = [
            'DB_HOST', 'DB_PORT', 'DB_USER', 'DB_PASSWORD', 'DB_NAME', 'DB_CHARSET',
            'MINDSPIDER_API_KEY', 'MINDSPIDER_BASE_URL', 'MINDSPIDER_MODEL_NAME'
        ]
        
        missing_configs = []
        for config_name in required_configs:
            if not hasattr(settings, config_name) or not getattr(settings, config_name):
                missing_configs.append(config_name)
        
        if missing_configs:
            logger.error(f"Missing configuration: {', '.join(missing_configs)}")
            logger.error("Please verify environment variables in .env")
            return False
        
        logger.info("Base configuration check passed")
        return True
    
    def check_database_connection(self) -> bool:
        """Check database connectivity."""
        logger.info("Checking database connection...")
        
        def build_async_url() -> str:
            dialect = (settings.DB_DIALECT or "mysql").lower()
            if dialect in ("postgresql", "postgres"):
                return f"postgresql+asyncpg://{settings.DB_USER}:{quote_plus(settings.DB_PASSWORD)}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
            # Default to asyncmy for MySQL async connections
            return (
                f"mysql+asyncmy://{settings.DB_USER}:{quote_plus(settings.DB_PASSWORD)}"
                f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}?charset={settings.DB_CHARSET}"
            )

        async def _test_connection(db_url: str) -> None:
            engine: AsyncEngine = create_async_engine(db_url, pool_pre_ping=True)
            try:
                async with engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
            finally:
                await engine.dispose()

        try:
            db_url: str = build_async_url()
            asyncio.run(_test_connection(db_url))
            logger.info("Database connection is healthy")
            return True
        except Exception as e:
            logger.exception(f"Database connection failed: {e}")
            return False
    
    def check_database_tables(self) -> bool:
        """Check whether required database tables exist."""
        logger.info("Checking database tables...")
        
        def build_async_url() -> str:
            dialect = (settings.DB_DIALECT or "mysql").lower()
            if dialect in ("postgresql", "postgres"):
                return f"postgresql+asyncpg://{settings.DB_USER}:{quote_plus(settings.DB_PASSWORD)}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
            return (
                f"mysql+asyncmy://{settings.DB_USER}:{quote_plus(settings.DB_PASSWORD)}"
                f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}?charset={settings.DB_CHARSET}"
            )

        async def _check_tables(db_url: str) -> list[str]:
            engine: AsyncEngine = create_async_engine(db_url, pool_pre_ping=True)
            try:
                async with engine.connect() as conn:
                    def _get_tables(sync_conn):
                        return inspect(sync_conn).get_table_names()
                    tables = await conn.run_sync(_get_tables)
                    return tables
            finally:
                await engine.dispose()

        try:
            db_url: str = build_async_url()
            existing_tables = asyncio.run(_check_tables(db_url))
            required_tables = ['daily_news', 'daily_topics']
            missing_tables = [t for t in required_tables if t not in existing_tables]
            if missing_tables:
                logger.error(f"Missing database tables: {', '.join(missing_tables)}")
                return False
            logger.info("Database table check passed")
            return True
        except Exception as e:
            logger.exception(f"Database table check failed: {e}")
            return False
    
    def initialize_database(self) -> bool:
        """Initialize database schema."""
        logger.info("Initializing database...")
        
        try:
            # Run database initialization script
            init_script = self.schema_path / "init_database.py"
            if not init_script.exists():
                logger.error("Error: database initialization script not found")
                return False
            
            result = subprocess.run(
                [sys.executable, str(init_script)],
                cwd=self.schema_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("Database initialization succeeded")
                return True
            else:
                logger.error(f"Database initialization failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.exception(f"Database initialization exception: {e}")
            return False
    
    def _ensure_database_ready(self) -> bool:
        """Ensure database tables are ready; auto-initialize if missing."""
        if not self.check_database_connection():
            logger.error("Database connection failed; cannot continue")
            return False
        
        if not self.check_database_tables():
            logger.warning("Database tables are missing, running auto-initialization...")
            if not self.initialize_database():
                logger.error("Database auto-initialization failed")
                return False
            logger.info("Database tables auto-initialized successfully")
        
        return True

    def check_dependencies(self) -> bool:
        """Check runtime dependencies."""
        logger.info("Checking dependencies...")
        
        # Check Python packages
        required_packages = ['pymysql', 'requests', 'playwright']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing Python packages: {', '.join(missing_packages)}")
            logger.info("Please run: pip install -r requirements.txt")
            return False
        
        # Check and install MediaCrawler dependencies
        mediacrawler_path = self.deep_sentiment_path / "MediaCrawler"
        if not mediacrawler_path.exists():
            logger.error("Error: MediaCrawler directory not found")
            return False
        
        # Auto-install MediaCrawler dependencies
        self._install_mediacrawler_dependencies()
        
        logger.info("Dependency check passed")
        return True
    
    def _install_mediacrawler_dependencies(self) -> bool:
        """Automatically install dependencies for the MediaCrawler submodule."""
        mediacrawler_req = self.deep_sentiment_path / "MediaCrawler" / "requirements.txt"
        
        if not mediacrawler_req.exists():
            logger.warning(f"MediaCrawler requirements.txt does not exist: {mediacrawler_req}")
            return False
        
        # Skip when dependencies are already installed (marker file)
        marker_file = self.deep_sentiment_path / "MediaCrawler" / ".deps_installed"
        req_mtime = mediacrawler_req.stat().st_mtime
        
        if marker_file.exists():
            marker_mtime = marker_file.stat().st_mtime
            if marker_mtime >= req_mtime:
                logger.debug("MediaCrawler dependencies already installed, skipping")
                return True
        
        logger.info("Installing MediaCrawler dependencies...")
        install_commands = [
            [sys.executable, "-m", "pip", "install", "-r", str(mediacrawler_req), "-q"],
            ["uv", "pip", "install", "-r", str(mediacrawler_req), "-q"],
        ]
        try:
            for cmd in install_commands:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                if result.returncode == 0:
                    marker_file.touch()
                    logger.info(f"MediaCrawler dependencies installed successfully (via {cmd[0]})")
                    return True
                logger.debug(f"{cmd[0]} install failed, trying next option: {result.stderr.strip()}")

            logger.error("MediaCrawler dependency installation failed: all install methods failed")
            return False

        except subprocess.TimeoutExpired:
            logger.error("MediaCrawler dependency installation timed out")
            return False
        except Exception as e:
            logger.exception(f"MediaCrawler dependency installation exception: {e}")
            return False

    def run_broad_topic_extraction(self, extract_date: date = None, keywords_count: int = 100) -> bool:
        """Run the BroadTopicExtraction module."""
        logger.info("Running BroadTopicExtraction module...")
        
        # Automatically check and initialize database tables
        if not self._ensure_database_ready():
            return False
        
        if not extract_date:
            extract_date = date.today()
        
        try:
            cmd = [
                sys.executable, "main.py",
                "--keywords", str(keywords_count)
            ]
            
            logger.info(f"Executing command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=self.broad_topic_path,
                timeout=1800  # 30 minute timeout
            )
            
            if result.returncode == 0:
                logger.info("BroadTopicExtraction module executed successfully")
                return True
            else:
                logger.error(f"BroadTopicExtraction module failed with return code: {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("BroadTopicExtraction module execution timed out")
            return False
        except Exception as e:
            logger.exception(f"BroadTopicExtraction module execution exception: {e}")
            return False
    
    def run_deep_sentiment_crawling(self, target_date: date = None, platforms: list = None,
                                   max_keywords: int = 50, max_notes: int = 50,
                                   test_mode: bool = False) -> bool:
        """Run the DeepSentimentCrawling module."""
        logger.info("Running DeepSentimentCrawling module...")

        # Automatically check and initialize database tables
        if not self._ensure_database_ready():
            return False

        # Automatically install MediaCrawler dependencies
        self._install_mediacrawler_dependencies()
        
        if not target_date:
            target_date = date.today()
        
        try:
            cmd = [sys.executable, "main.py"]
            
            if target_date:
                cmd.extend(["--date", target_date.strftime("%Y-%m-%d")])
            
            if platforms:
                cmd.extend(["--platforms"] + platforms)
            
            cmd.extend([
                "--max-keywords", str(max_keywords),
                "--max-notes", str(max_notes)
            ])
            
            if test_mode:
                cmd.append("--test")
            
            logger.info(f"Executing command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=self.deep_sentiment_path,
                timeout=3600  # 60 minute timeout
            )
            
            if result.returncode == 0:
                logger.info("DeepSentimentCrawling module executed successfully")
                return True
            else:
                logger.error(f"DeepSentimentCrawling module failed with return code: {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("DeepSentimentCrawling module execution timed out")
            return False
        except Exception as e:
            logger.exception(f"DeepSentimentCrawling module execution exception: {e}")
            return False
    
    def run_complete_workflow(self, target_date: date = None, platforms: list = None,
                             keywords_count: int = 100, max_keywords: int = 50,
                             max_notes: int = 50, test_mode: bool = False) -> bool:
        """Run the complete workflow."""
        logger.info("Starting full MindSpider workflow")
        
        # Automatically check and initialize database tables
        if not self._ensure_database_ready():
            return False
        
        if not target_date:
            target_date = date.today()
        
        logger.info(f"Target date: {target_date}")
        logger.info(f"Platforms: {platforms if platforms else 'all supported platforms'}")
        logger.info(f"Test mode: {'yes' if test_mode else 'no'}")
        
        # Step 1: Run topic extraction
        logger.info("=== Step 1: Topic extraction ===")
        if not self.run_broad_topic_extraction(target_date, keywords_count):
            logger.error("Topic extraction failed, stopping workflow")
            return False
        
        # Step 2: Run sentiment crawling
        logger.info("=== Step 2: Sentiment crawling ===")
        if not self.run_deep_sentiment_crawling(target_date, platforms, max_keywords, max_notes, test_mode):
            logger.error("Sentiment crawling failed, but topic extraction completed")
            return False
        
        logger.info("Complete workflow finished successfully")
        return True
    
    def show_status(self):
        """Show project status."""
        logger.info("MindSpider project status:")
        logger.info(f"Project path: {self.project_root}")
        
        # Configuration status
        config_ok = self.check_config()
        logger.info(f"Configuration: {'OK' if config_ok else 'ERROR'}")
        
        # Database status
        if config_ok:
            db_conn_ok = self.check_database_connection()
            logger.info(f"Database connection: {'OK' if db_conn_ok else 'ERROR'}")
            
            if db_conn_ok:
                db_tables_ok = self.check_database_tables()
                logger.info(f"Database tables: {'OK' if db_tables_ok else 'INITIALIZATION REQUIRED'}")
        
        # Dependency status
        deps_ok = self.check_dependencies()
        logger.info(f"Dependencies: {'OK' if deps_ok else 'ERROR'}")
        
        # Module status
        broad_topic_exists = self.broad_topic_path.exists()
        deep_sentiment_exists = self.deep_sentiment_path.exists()
        logger.info(f"BroadTopicExtraction module: {'present' if broad_topic_exists else 'missing'}")
        logger.info(f"DeepSentimentCrawling module: {'present' if deep_sentiment_exists else 'missing'}")
    
    def setup_project(self) -> bool:
        """Run project setup."""
        logger.info("Starting MindSpider project setup...")
        
        # 1. Check configuration
        if not self.check_config():
            return False
        
        # 2. Check dependencies
        if not self.check_dependencies():
            return False
        
        # 3. Check database connection
        if not self.check_database_connection():
            return False
        
        # 4. Check and initialize database tables
        if not self.check_database_tables():
            logger.info("Database tables require initialization...")
            if not self.initialize_database():
                return False
        
        logger.info("MindSpider project setup completed")
        return True

PLATFORM_CHOICES = ['xhs', 'dy', 'ks', 'bili', 'wb', 'tieba', 'zhihu']

PLATFORM_ALIASES = {
    'weibo': 'wb', 'webo': 'wb',
    'douyin': 'dy',
    'kuaishou': 'ks',
    'bilibili': 'bili', 'bstation': 'bili',
    'xiaohongshu': 'xhs', 'redbook': 'xhs',
    'zhihu': 'zhihu',
    'tieba': 'tieba',
}

class SuggestiveArgumentParser(argparse.ArgumentParser):
    """Provide similar candidates when an argument value is invalid."""

    def error(self, message: str):
        match = re.search(r"invalid choice: '([^']+)'", message)
        if match:
            bad = match.group(1)
            alias = PLATFORM_ALIASES.get(bad.lower())
            suggestions = difflib.get_close_matches(bad, PLATFORM_CHOICES, n=3, cutoff=0.3)
            if alias:
                print(f"Error: '{bad}' is not a valid platform code. Did you mean '{alias}'?", file=sys.stderr)
            elif suggestions:
                print(f"Error: '{bad}' is not a valid platform code. Closest matches: {suggestions}", file=sys.stderr)
            else:
                print(f"Error: '{bad}' is not a valid platform code. Valid platforms: {PLATFORM_CHOICES}", file=sys.stderr)
            print(f"Full error: {message}", file=sys.stderr)
        else:
            print(f"Error: {message}", file=sys.stderr)
        self.print_usage(sys.stderr)
        sys.exit(2)

def main():
    """Command line entry point."""
    parser = SuggestiveArgumentParser(description="MindSpider - Main program for the AI crawler project")
    
    # Basic operations
    parser.add_argument("--setup", action="store_true", help="Initialize project setup")
    parser.add_argument("--status", action="store_true", help="Show project status")
    parser.add_argument("--init-db", action="store_true", help="Initialize database")
    
    # Module execution
    parser.add_argument("--broad-topic", action="store_true", help="Run only the topic extraction module")
    parser.add_argument("--deep-sentiment", action="store_true", help="Run only the sentiment crawling module")
    parser.add_argument("--complete", action="store_true", help="Run the complete workflow")
    
    # Runtime parameters
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD), defaults to today")
    parser.add_argument("--platforms", type=str, nargs='+',
                       choices=PLATFORM_CHOICES,
                       help="Specify platforms to crawl")
    parser.add_argument("--keywords-count", type=int, default=100, help="Keyword count for topic extraction")
    parser.add_argument("--max-keywords", type=int, default=50, help="Max keyword count per platform")
    parser.add_argument("--max-notes", type=int, default=50, help="Max content items per keyword")
    parser.add_argument("--test", action="store_true", help="Test mode (small data volume)")
    
    args = parser.parse_args()
    
    # Parse date
    target_date = None
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            logger.error("Error: invalid date format, use YYYY-MM-DD")
            return
    
    # Create MindSpider instance
    spider = MindSpider()
    
    try:
        # Show status
        if args.status:
            spider.show_status()
            return
        
        # Project setup
        if args.setup:
            if spider.setup_project():
                logger.info("Project setup completed, MindSpider is ready to use")
            else:
                logger.error("Project setup failed, please check configuration and environment")
            return
        
        # Initialize database
        if args.init_db:
            if spider.initialize_database():
                logger.info("Database initialization succeeded")
            else:
                logger.error("Database initialization failed")
            return
        
        # Run module(s)
        if args.broad_topic:
            spider.run_broad_topic_extraction(target_date, args.keywords_count)
        elif args.deep_sentiment:
            spider.run_deep_sentiment_crawling(
                target_date, args.platforms, args.max_keywords, args.max_notes, args.test
            )
        elif args.complete:
            spider.run_complete_workflow(
                target_date, args.platforms, args.keywords_count, 
                args.max_keywords, args.max_notes, args.test
            )
        else:
            # Default behavior: run complete workflow
            logger.info("Running complete MindSpider workflow...")
            spider.run_complete_workflow(
                target_date, args.platforms, args.keywords_count,
                args.max_keywords, args.max_notes, args.test
            )
    
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
    except Exception as e:
        logger.exception(f"Execution error: {e}")

if __name__ == "__main__":
    main()
