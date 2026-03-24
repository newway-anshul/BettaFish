#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSentimentCrawling module - Keyword manager
Fetches keywords from BroadTopicExtraction and assigns them to crawling platforms
"""

import sys
import json
from datetime import date, timedelta, datetime
from pathlib import Path
from typing import List, Dict, Optional
import random
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Add project root to import path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import config
except ImportError:
    raise ImportError("Failed to import config.py")

from config import settings
from loguru import logger


class KeywordManager:
    """Keyword manager."""

    def __init__(self):
        """Initialize keyword manager."""
        self.engine: Engine = None
        self.connect()

    def connect(self):
        """Connect to database."""
        try:
            dialect = (settings.DB_DIALECT or "mysql").lower()
            if dialect in ("postgresql", "postgres"):
                url = f"postgresql+psycopg://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
            else:
                url = f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}?charset={settings.DB_CHARSET}"
            self.engine = create_engine(url, future=True)
            logger.info(f"Keyword manager connected to database successfully: {settings.DB_NAME}")
        except ModuleNotFoundError as e:
            missing: str = str(e)
            if "psycopg" in missing:
                logger.error("Database connection failed: PostgreSQL driver psycopg is not installed. Install with: psycopg[binary]. Example: uv pip install psycopg[binary]")
            elif "pymysql" in missing:
                logger.error("Database connection failed: MySQL driver pymysql is not installed. Install with: pymysql. Example: uv pip install pymysql")
            else:
                logger.error(f"Database connection failed (missing driver): {e}")
            raise
        except Exception as e:
            logger.exception(f"Keyword manager database connection failed: {e}")
            raise

    def get_latest_keywords(self, target_date: date = None, max_keywords: int = 100) -> List[str]:
        """
        Get latest keyword list.

        Args:
            target_date: Target date, default is today
            max_keywords: Maximum keyword count

        Returns:
            Keyword list
        """
        if not target_date:
            target_date = date.today()

        logger.info(f"Fetching keywords for {target_date}...")

        # Try keywords for the target date first
        topics_data = self.get_daily_topics(target_date)

        if topics_data and topics_data.get("keywords"):
            keywords = topics_data["keywords"]
            logger.info(f"Successfully fetched {len(keywords)} keywords for {target_date}")

            # If too many keywords, randomly sample
            if len(keywords) > max_keywords:
                keywords = random.sample(keywords, max_keywords)
                logger.info(f"Randomly selected {max_keywords} keywords")

            return keywords

        # If no data for today, try recent days
        logger.info(f"No keyword data for {target_date}, trying recent keywords...")
        recent_topics = self.get_recent_topics(days=7)

        if recent_topics:
            # Merge keywords from recent days
            all_keywords = []
            for topic in recent_topics:
                if topic.get("keywords"):
                    all_keywords.extend(topic["keywords"])

            # Deduplicate and cap size
            unique_keywords = list(set(all_keywords))
            if len(unique_keywords) > max_keywords:
                unique_keywords = random.sample(unique_keywords, max_keywords)

            logger.info(f"Fetched {len(unique_keywords)} keywords from last 7 days")
            return unique_keywords

        # Fall back to default keywords
        logger.info("No keyword data found, using default keywords")
        return self._get_default_keywords()

    def get_daily_topics(self, extract_date: date = None) -> Optional[Dict]:
        """
        Get daily topic analysis.

        Args:
            extract_date: Extraction date, default is today

        Returns:
            Topic analysis data, or None if unavailable
        """
        if not extract_date:
            extract_date = date.today()

        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("SELECT * FROM daily_topics WHERE extract_date = :d"),
                    {"d": extract_date},
                ).mappings().first()

            if result:
                # Convert to mutable dict before assignment
                result = dict(result)
                result["keywords"] = json.loads(result["keywords"]) if result.get("keywords") else []
                return result
            return None

        except Exception as e:
            logger.exception(f"Failed to get topic analysis: {e}")
            return None

    def get_recent_topics(self, days: int = 7) -> List[Dict]:
        """
        Get topic analysis from recent days.

        Args:
            days: Number of days

        Returns:
            Topic analysis list
        """
        try:
            start_date = date.today() - timedelta(days=days)
            with self.engine.connect() as conn:
                results = conn.execute(
                    text(
                        """
                        SELECT * FROM daily_topics
                        WHERE extract_date >= :start_date
                        ORDER BY extract_date DESC
                        """
                    ),
                    {"start_date": start_date},
                ).mappings().all()

            # Convert to mutable dict list before processing
            results = [dict(r) for r in results]
            for result in results:
                result["keywords"] = json.loads(result["keywords"]) if result.get("keywords") else []

            return results

        except Exception as e:
            logger.exception(f"Failed to get recent topic analysis: {e}")
            return []

    def _get_default_keywords(self) -> List[str]:
        """Get default keyword list."""
        return [
            "technology",
            "artificial intelligence",
            "AI",
            "programming",
            "internet",
            "startup",
            "investment",
            "personal finance",
            "stock market",
            "economy",
            "education",
            "learning",
            "exam",
            "university",
            "employment",
            "health",
            "wellness",
            "fitness",
            "food",
            "travel",
            "fashion",
            "beauty",
            "shopping",
            "lifestyle",
            "home",
            "movie",
            "music",
            "gaming",
            "entertainment",
            "celebrity",
            "news",
            "trending",
            "society",
            "policy",
            "environment",
        ]

    def get_all_keywords_for_platforms(
        self, platforms: List[str], target_date: date = None, max_keywords: int = 100
    ) -> List[str]:
        """
        Get the same keyword list for all platforms.

        Args:
            platforms: Platform list
            target_date: Target date
            max_keywords: Maximum keyword count

        Returns:
            Keyword list shared by all platforms
        """
        keywords = self.get_latest_keywords(target_date, max_keywords)

        if keywords:
            logger.info(f"Prepared the same {len(keywords)} keywords for {len(platforms)} platforms")
            logger.info("Each keyword will be crawled on all platforms")

        return keywords

    def get_keywords_for_platform(
        self, platform: str, target_date: date = None, max_keywords: int = 50
    ) -> List[str]:
        """
        Get keywords for a specific platform (currently shared across all platforms).

        Args:
            platform: Platform name
            target_date: Target date
            max_keywords: Maximum keyword count

        Returns:
            Keyword list (same as other platforms)
        """
        keywords = self.get_latest_keywords(target_date, max_keywords)

        logger.info(f"Prepared {len(keywords)} keywords for platform {platform} (same as other platforms)")
        return keywords

    def _filter_keywords_by_platform(self, keywords: List[str], platform: str) -> List[str]:
        """
        Filter keywords by platform characteristics.

        Args:
            keywords: Raw keyword list
            platform: Platform name

        Returns:
            Filtered keyword list
        """
        # Platform preference keywords (adjust as needed)
        platform_preferences = {
            "xhs": ["beauty", "fashion", "lifestyle", "food", "travel", "shopping", "health", "wellness"],
            "dy": ["entertainment", "music", "dance", "funny", "food", "lifestyle", "technology", "education"],
            "ks": ["lifestyle", "funny", "rural", "food", "craft", "music", "entertainment"],
            "bili": ["technology", "gaming", "anime", "learning", "programming", "digital", "science"],
            "wb": ["trending", "news", "entertainment", "celebrity", "society", "current affairs", "technology"],
            "tieba": ["gaming", "anime", "learning", "lifestyle", "interest", "discussion"],
            "zhihu": ["knowledge", "learning", "technology", "career", "investment", "education", "thinking"],
        }

        # Prefer platform-specific keywords when available
        preferred_keywords = platform_preferences.get(platform, [])

        if preferred_keywords:
            # First, select preferred keywords
            filtered = []
            remaining = []

            for keyword in keywords:
                if any(pref in keyword for pref in preferred_keywords):
                    filtered.append(keyword)
                else:
                    remaining.append(keyword)

            # If preferred set is too small, fill with remaining keywords
            if len(filtered) < len(keywords) // 2:
                filtered.extend(remaining[: len(keywords) - len(filtered)])

            return filtered

        # No specific preference
        return keywords

    def get_crawling_summary(self, target_date: date = None) -> Dict:
        """
        Get crawling task summary.

        Args:
            target_date: Target date

        Returns:
            Crawling summary information
        """
        if not target_date:
            target_date = date.today()

        topics_data = self.get_daily_topics(target_date)

        if topics_data:
            return {
                "date": target_date,
                "keywords_count": len(topics_data.get("keywords", [])),
                "summary": topics_data.get("summary", ""),
                "has_data": True,
            }
        return {
            "date": target_date,
            "keywords_count": 0,
            "summary": "No data available",
            "has_data": False,
        }

    def close(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("Keyword manager database connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Test keyword manager
    with KeywordManager() as km:
        # Test fetching keywords
        keywords = km.get_latest_keywords(max_keywords=20)
        logger.info(f"Fetched keywords: {keywords}")

        # Test platform distribution
        platforms = ["xhs", "dy", "bili"]
        distribution = km.distribute_keywords_by_platform(keywords, platforms)
        for platform, kws in distribution.items():
            logger.info(f"{platform}: {kws}")

        # Test crawling summary
        summary = km.get_crawling_summary()
        logger.info(f"Crawling summary: {summary}")

        logger.info("Keyword manager test completed")