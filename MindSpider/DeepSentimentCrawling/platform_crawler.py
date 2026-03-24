#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSentimentCrawling module - Platform crawler manager
Configures and invokes MediaCrawler for multi-platform crawling
"""

import os
import sys
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import json
from loguru import logger

# Add project root to import path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import config
except ImportError:
    raise ImportError("Failed to import config.py")


class PlatformCrawler:
    """Platform crawler manager."""

    def __init__(self):
        """Initialize platform crawler manager."""
        self.mediacrawler_path = Path(__file__).parent / "MediaCrawler"
        self.supported_platforms = ["xhs", "dy", "ks", "bili", "wb", "tieba", "zhihu"]
        self.crawl_stats = {}

        # Ensure MediaCrawler submodule is initialized
        db_config_path = self.mediacrawler_path / "config" / "db_config.py"
        if not self.mediacrawler_path.exists() or not db_config_path.exists():
            logger.error("MediaCrawler submodule is missing or incomplete")
            logger.error("Initialize submodules from project root with:")
            logger.error("   git submodule update --init --recursive")
            raise FileNotFoundError(
                "MediaCrawler submodule is not initialized. Run: git submodule update --init --recursive"
            )

        logger.info(f"Initialized platform crawler manager, MediaCrawler path: {self.mediacrawler_path}")

    def configure_mediacrawler_db(self):
        """Configure MediaCrawler to use MindSpider database (MySQL or PostgreSQL)."""
        try:
            # Determine database type
            db_dialect = (config.settings.DB_DIALECT or "mysql").lower()
            is_postgresql = db_dialect in ("postgresql", "postgres")

            # MediaCrawler database config path
            db_config_path = self.mediacrawler_path / "config" / "db_config.py"

            # Read original config
            with open(db_config_path, "r", encoding="utf-8") as f:
                content = f.read()

            # PostgreSQL values: use MindSpider settings for postgres,
            # otherwise keep defaults/env-based values
            pg_password = config.settings.DB_PASSWORD if is_postgresql else "bettafish"
            pg_user = config.settings.DB_USER if is_postgresql else "bettafish"
            pg_host = config.settings.DB_HOST if is_postgresql else "127.0.0.1"
            pg_port = config.settings.DB_PORT if is_postgresql else 5444
            pg_db_name = config.settings.DB_NAME if is_postgresql else "bettafish"

            # Replace database configuration using MindSpider DB settings
            new_config = f'''# Statement: This code is for learning and research purposes only.
# Users must follow these principles:
# 1. No commercial use.
# 2. Comply with target platform terms and robots.txt when applicable.
# 3. Do not perform large-scale crawling or interfere with platform operations.
# 4. Control request frequency reasonably to avoid unnecessary load.
# 5. Do not use for illegal or improper purposes.
#
# See the LICENSE file in the project root for full license terms.
# Using this code means you agree to the above principles and all LICENSE terms.


import os

# mysql config - use MindSpider database settings
MYSQL_DB_PWD = "{config.settings.DB_PASSWORD}"
MYSQL_DB_USER = "{config.settings.DB_USER}"
MYSQL_DB_HOST = "{config.settings.DB_HOST}"
MYSQL_DB_PORT = {config.settings.DB_PORT}
MYSQL_DB_NAME = "{config.settings.DB_NAME}"

mysql_db_config = {{
    "user": MYSQL_DB_USER,
    "password": MYSQL_DB_PWD,
    "host": MYSQL_DB_HOST,
    "port": MYSQL_DB_PORT,
    "db_name": MYSQL_DB_NAME,
}}


# redis config
REDIS_DB_HOST = "127.0.0.1"  # your redis host
REDIS_DB_PWD = os.getenv("REDIS_DB_PWD", "123456")  # your redis password
REDIS_DB_PORT = os.getenv("REDIS_DB_PORT", 6379)  # your redis port
REDIS_DB_NUM = os.getenv("REDIS_DB_NUM", 0)  # your redis db num

# cache type
CACHE_TYPE_REDIS = "redis"
CACHE_TYPE_MEMORY = "memory"

# sqlite config
SQLITE_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "sqlite_tables.db")

sqlite_db_config = {{
    "db_path": SQLITE_DB_PATH
}}

# mongodb config
MONGODB_HOST = os.getenv("MONGODB_HOST", "localhost")
MONGODB_PORT = os.getenv("MONGODB_PORT", 27017)
MONGODB_USER = os.getenv("MONGODB_USER", "")
MONGODB_PWD = os.getenv("MONGODB_PWD", "")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "media_crawler")

mongodb_config = {{
    "host": MONGODB_HOST,
    "port": int(MONGODB_PORT),
    "user": MONGODB_USER,
    "password": MONGODB_PWD,
    "db_name": MONGODB_DB_NAME,
}}

# postgres config - use MindSpider DB settings when DB_DIALECT is postgresql, otherwise env/defaults
POSTGRES_DB_PWD = os.getenv("POSTGRES_DB_PWD", "{pg_password}")
POSTGRES_DB_USER = os.getenv("POSTGRES_DB_USER", "{pg_user}")
POSTGRES_DB_HOST = os.getenv("POSTGRES_DB_HOST", "{pg_host}")
POSTGRES_DB_PORT = os.getenv("POSTGRES_DB_PORT", "{pg_port}")
POSTGRES_DB_NAME = os.getenv("POSTGRES_DB_NAME", "{pg_db_name}")

postgres_db_config = {{
    "user": POSTGRES_DB_USER,
    "password": POSTGRES_DB_PWD,
    "host": POSTGRES_DB_HOST,
    "port": POSTGRES_DB_PORT,
    "db_name": POSTGRES_DB_NAME,
}}

'''

            # Write new config
            with open(db_config_path, "w", encoding="utf-8") as f:
                f.write(new_config)

            db_type = "PostgreSQL" if is_postgresql else "MySQL"
            logger.info(f"Configured MediaCrawler to use MindSpider {db_type} database")
            return True

        except Exception as e:
            logger.exception(f"Failed to configure MediaCrawler database: {e}")
            return False

    def create_base_config(
        self, platform: str, keywords: List[str], crawler_type: str = "search", max_notes: int = 50
    ) -> bool:
        """
        Create base config for MediaCrawler.

        Args:
            platform: Platform name
            keywords: Keyword list
            crawler_type: Crawler type
            max_notes: Maximum crawl count

        Returns:
            Whether configuration succeeded
        """
        try:
            # Determine SAVE_DATA_OPTION by database type
            db_dialect = (config.settings.DB_DIALECT or "mysql").lower()
            is_postgresql = db_dialect in ("postgresql", "postgres")
            save_data_option = "postgres" if is_postgresql else "db"

            base_config_path = self.mediacrawler_path / "config" / "base_config.py"

            # Convert keyword list to comma-separated string
            keywords_str = ",".join(keywords)

            # Read original config file
            with open(base_config_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Replace key config fields
            # skip_until_paren: when replacing a multiline assignment header
            # (line ending with "(") with a single-line value, skip continuation
            # lines until matching ")" is reached.
            lines = content.split("\n")
            new_lines = []
            skip_until_paren = False

            for line in lines:
                # Skip continuation lines of a multiline assignment
                if skip_until_paren:
                    if line.strip() == ")":
                        skip_until_paren = False
                    continue

                replaced = None
                if line.startswith("PLATFORM = "):
                    replaced = f'PLATFORM = "{platform}"  # platform: xhs | dy | ks | bili | wb | tieba | zhihu'
                elif line.startswith("KEYWORDS = "):
                    replaced = f'KEYWORDS = "{keywords_str}"  # keyword search config, comma-separated'
                elif line.startswith("CRAWLER_TYPE = "):
                    replaced = (
                        f'CRAWLER_TYPE = "{crawler_type}"  # crawler type: '
                        "search(keyword search) | detail(post details) | creator(creator profile data)"
                    )
                elif line.startswith("SAVE_DATA_OPTION = "):
                    replaced = f'SAVE_DATA_OPTION = "{save_data_option}"  # csv or db or json or sqlite or postgres'
                elif line.startswith("CRAWLER_MAX_NOTES_COUNT = "):
                    replaced = f"CRAWLER_MAX_NOTES_COUNT = {max_notes}"
                elif line.startswith("ENABLE_GET_COMMENTS = "):
                    replaced = "ENABLE_GET_COMMENTS = True"
                elif line.startswith("CRAWLER_MAX_COMMENTS_COUNT_SINGLENOTES = "):
                    replaced = "CRAWLER_MAX_COMMENTS_COUNT_SINGLENOTES = 20"
                elif line.startswith("HEADLESS = "):
                    replaced = "HEADLESS = True"

                if replaced is not None:
                    new_lines.append(replaced)
                    # If original was multiline assignment start, skip continuation lines
                    if line.rstrip().endswith("("):
                        skip_until_paren = True
                else:
                    new_lines.append(line)

            # Write updated config
            with open(base_config_path, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines))

            logger.info(
                f"Configured platform {platform}, crawler type: {crawler_type}, "
                f"keyword count: {len(keywords)}, max crawl count: {max_notes}, "
                f"save option: {save_data_option}"
            )
            return True

        except Exception as e:
            logger.exception(f"Failed to create base config: {e}")
            return False

    def run_crawler(
        self, platform: str, keywords: List[str], login_type: str = "qrcode", max_notes: int = 50
    ) -> Dict:
        """
        Run crawler.

        Args:
            platform: Platform name
            keywords: Keyword list
            login_type: Login type
            max_notes: Maximum crawl count

        Returns:
            Crawl result statistics
        """
        if platform not in self.supported_platforms:
            raise ValueError(f"Unsupported platform: {platform}")

        if not keywords:
            raise ValueError("Keyword list cannot be empty")

        start_message = f"\nStarting crawl for platform: {platform}"
        start_message += f"\nKeywords: {keywords[:5]}{'...' if len(keywords) > 5 else ''} (total {len(keywords)})"
        logger.info(start_message)

        start_time = datetime.now()

        try:
            # Configure database
            if not self.configure_mediacrawler_db():
                return {"success": False, "error": "Database configuration failed"}

            # Create base config
            if not self.create_base_config(platform, keywords, "search", max_notes):
                return {"success": False, "error": "Base configuration creation failed"}

            # Determine save_data_option by database type
            db_dialect = (config.settings.DB_DIALECT or "mysql").lower()
            is_postgresql = db_dialect in ("postgresql", "postgres")
            save_data_option = "postgres" if is_postgresql else "db"

            # Build command
            cmd = [
                sys.executable,
                "main.py",
                "--platform",
                platform,
                "--lt",
                login_type,
                "--type",
                "search",
                "--save_data_option",
                save_data_option,
                "--headless",
                "false",
            ]

            logger.info(f"Executing command: {' '.join(cmd)}")

            # Run from MediaCrawler directory
            result = subprocess.run(
                cmd,
                cwd=self.mediacrawler_path,
                timeout=3600,  # 60 minute timeout
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Build stats
            crawl_stats = {
                "platform": platform,
                "keywords_count": len(keywords),
                "duration_seconds": duration,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "return_code": result.returncode,
                "success": result.returncode == 0,
                "notes_count": 0,
                "comments_count": 0,
                "errors_count": 0,
            }

            # Persist stats
            self.crawl_stats[platform] = crawl_stats

            if result.returncode == 0:
                logger.info(f"✅ {platform} crawl completed, duration: {duration:.1f}s")
            else:
                logger.error(f"❌ {platform} crawl failed, return code: {result.returncode}")

            return crawl_stats

        except subprocess.TimeoutExpired:
            logger.exception(f"❌ {platform} crawl timed out")
            return {"success": False, "error": "Crawl timed out", "platform": platform}
        except Exception as e:
            logger.exception(f"❌ {platform} crawl exception: {e}")
            return {"success": False, "error": str(e), "platform": platform}

    def _parse_crawl_output(self, output_lines: List[str], error_lines: List[str]) -> Dict:
        """Parse crawl output and extract statistics."""
        stats = {
            "notes_count": 0,
            "comments_count": 0,
            "errors_count": 0,
            "login_required": False,
        }

        # Parse output lines
        for line in output_lines:
            if "note" in line.lower() or "content" in line.lower():
                try:
                    import re

                    numbers = re.findall(r"\d+", line)
                    if numbers:
                        stats["notes_count"] = int(numbers[0])
                except Exception:
                    pass
            elif "comment" in line.lower():
                try:
                    import re

                    numbers = re.findall(r"\d+", line)
                    if numbers:
                        stats["comments_count"] = int(numbers[0])
                except Exception:
                    pass
            elif "login" in line.lower() or "qr" in line.lower():
                stats["login_required"] = True

        # Parse error lines
        for line in error_lines:
            if "error" in line.lower() or "exception" in line.lower():
                stats["errors_count"] += 1

        return stats

    def run_multi_platform_crawl_by_keywords(
        self,
        keywords: List[str],
        platforms: List[str],
        login_type: str = "qrcode",
        max_notes_per_keyword: int = 50,
    ) -> Dict:
        """
        Multi-platform crawl by keywords - each keyword is crawled on all platforms.

        Args:
            keywords: Keyword list
            platforms: Platform list
            login_type: Login type
            max_notes_per_keyword: Max crawl count per keyword per platform

        Returns:
            Overall crawling statistics
        """

        start_message = "\n🚀 Starting full-platform keyword crawling"
        start_message += f"\n   Keyword count: {len(keywords)}"
        start_message += f"\n   Platform count: {len(platforms)}"
        start_message += f"\n   Login type: {login_type}"
        start_message += f"\n   Max crawl count per keyword per platform: {max_notes_per_keyword}"
        start_message += f"\n   Total tasks: {len(keywords)} × {len(platforms)} = {len(keywords) * len(platforms)}"
        logger.info(start_message)

        total_stats = {
            "total_keywords": len(keywords),
            "total_platforms": len(platforms),
            "total_tasks": len(keywords) * len(platforms),
            "successful_tasks": 0,
            "failed_tasks": 0,
            "total_notes": 0,
            "total_comments": 0,
            "keyword_results": {},
            "platform_summary": {},
        }

        # Initialize platform summary
        for platform in platforms:
            total_stats["platform_summary"][platform] = {
                "successful_keywords": 0,
                "failed_keywords": 0,
                "total_notes": 0,
                "total_comments": 0,
            }

        # Crawl all keywords once per platform
        for platform in platforms:
            logger.info(f"\n📝 Crawling all keywords on platform {platform}")
            logger.info(f"   Keywords: {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}")

            try:
                # Send all keywords at once to the platform
                result = self.run_crawler(platform, keywords, login_type, max_notes_per_keyword)

                if result.get("success"):
                    total_stats["successful_tasks"] += len(keywords)
                    total_stats["platform_summary"][platform]["successful_keywords"] = len(keywords)

                    notes_count = result.get("notes_count", 0)
                    comments_count = result.get("comments_count", 0)

                    total_stats["total_notes"] += notes_count
                    total_stats["total_comments"] += comments_count
                    total_stats["platform_summary"][platform]["total_notes"] = notes_count
                    total_stats["platform_summary"][platform]["total_comments"] = comments_count

                    # Record result for each keyword
                    for keyword in keywords:
                        if keyword not in total_stats["keyword_results"]:
                            total_stats["keyword_results"][keyword] = {}
                        total_stats["keyword_results"][keyword][platform] = result

                    logger.info("   ✅ Crawl succeeded")
                else:
                    total_stats["failed_tasks"] += len(keywords)
                    total_stats["platform_summary"][platform]["failed_keywords"] = len(keywords)

                    # Record failed result for each keyword
                    for keyword in keywords:
                        if keyword not in total_stats["keyword_results"]:
                            total_stats["keyword_results"][keyword] = {}
                        total_stats["keyword_results"][keyword][platform] = result

                    logger.error(f"   ❌ Failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                total_stats["failed_tasks"] += len(keywords)
                total_stats["platform_summary"][platform]["failed_keywords"] = len(keywords)
                error_result = {"success": False, "error": str(e)}

                # Record exception result for each keyword
                for keyword in keywords:
                    if keyword not in total_stats["keyword_results"]:
                        total_stats["keyword_results"][keyword] = {}
                    total_stats["keyword_results"][keyword][platform] = error_result

                logger.error(f"   ❌ Exception: {e}")

        # Print detailed stats
        finish_message = "\n📊 Full-platform keyword crawling completed"
        finish_message += f"\n   Total tasks: {total_stats['total_tasks']}"
        finish_message += f"\n   Success: {total_stats['successful_tasks']}"
        finish_message += f"\n   Failed: {total_stats['failed_tasks']}"
        finish_message += (
            f"\n   Success rate: {total_stats['successful_tasks'] / total_stats['total_tasks'] * 100:.1f}%"
        )
        logger.info(finish_message)

        platform_summary_message = "\n📈 Platform summary:"
        for platform, stats in total_stats["platform_summary"].items():
            success_rate = stats["successful_keywords"] / len(keywords) * 100 if keywords else 0
            platform_summary_message += (
                f"\n   {platform}: {stats['successful_keywords']}/{len(keywords)} keywords succeeded "
                f"({success_rate:.1f}%)"
            )
        logger.info(platform_summary_message)

        return total_stats

    def get_crawl_statistics(self) -> Dict:
        """Get crawl statistics."""
        return {
            "platforms_crawled": list(self.crawl_stats.keys()),
            "total_platforms": len(self.crawl_stats),
            "detailed_stats": self.crawl_stats,
        }

    def save_crawl_log(self, log_path: str = None):
        """Save crawl log."""
        if not log_path:
            log_path = f"crawl_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(self.crawl_stats, f, ensure_ascii=False, indent=2)
            logger.info(f"Crawl log saved to: {log_path}")
        except Exception as e:
            logger.exception(f"Failed to save crawl log: {e}")


if __name__ == "__main__":
    # Test platform crawler manager
    crawler = PlatformCrawler()

    # Test config
    test_keywords = ["technology", "AI", "programming"]
    result = crawler.run_crawler("xhs", test_keywords, max_notes=5)

    logger.info(f"Test result: {result}")
    logger.info("Platform crawler manager test completed")