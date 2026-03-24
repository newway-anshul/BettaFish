#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BroadTopicExtraction module - Main program
Integrates the full topic-extraction workflow and CLI tools
"""

import sys
import asyncio
import argparse
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

# Add project root to import path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from BroadTopicExtraction.get_today_news import NewsCollector, SOURCE_NAMES
    from BroadTopicExtraction.topic_extractor import TopicExtractor
    from BroadTopicExtraction.database_manager import DatabaseManager
except ImportError as e:
    logger.exception(f"Module import failed: {e}")
    logger.error("Please run from project root and ensure all dependencies are installed")
    sys.exit(1)

class BroadTopicExtraction:
    """Primary workflow for BroadTopicExtraction."""
    
    def __init__(self):
        """Initialize components."""
        self.news_collector = NewsCollector()
        self.topic_extractor = TopicExtractor()
        self.db_manager = DatabaseManager()
        
        logger.info("BroadTopicExtraction initialized")
    
    def close(self):
        """Close resources."""
        if self.news_collector:
            self.news_collector.close()
        if self.db_manager:
            self.db_manager.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    async def run_daily_extraction(self, 
                                  news_sources: Optional[List[str]] = None,
                                  max_keywords: int = 100) -> Dict:
        """
        Run daily topic extraction workflow.
        
        Args:
            news_sources: News source list. None means all supported sources
            max_keywords: Maximum keyword count
            
        Returns:
            Dictionary containing full extraction results
        """
        extraction_result_message = ""
        extraction_result_message += "\nMindSpider AI Crawler - Daily Topic Extraction\n"
        extraction_result_message += f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        extraction_result_message += f"Target date: {date.today()}\n"
        
        if news_sources:
            extraction_result_message += f"Specified platforms: {len(news_sources)}\n"
            for source in news_sources:
                source_name = SOURCE_NAMES.get(source, source)
                extraction_result_message += f"  - {source_name}\n"
        else:
            extraction_result_message += f"Crawl platforms: all {len(SOURCE_NAMES)} supported platforms\n"
        
        extraction_result_message += f"Keyword count: up to {max_keywords}\n"
        
        logger.info(extraction_result_message)
        
        extraction_result = {
            'success': False,
            'extraction_date': date.today().isoformat(),
            'start_time': datetime.now().isoformat(),
            'news_collection': {},
            'topic_extraction': {},
            'database_save': {},
            'error': None
        }
        
        try:
            # Step 1: collect news
            logger.info("[Step 1] Collecting trending news...")
            news_result = await self.news_collector.collect_and_save_news(
                sources=news_sources
            )
            
            extraction_result['news_collection'] = {
                'success': news_result['success'],
                'total_news': news_result.get('total_news', 0),
                'successful_sources': news_result.get('successful_sources', 0),
                'total_sources': news_result.get('total_sources', 0)
            }
            
            if not news_result['success'] or not news_result['news_list']:
                raise Exception("News collection failed or no news was fetched")
            
            # Step 2: extract keywords and generate summary
            logger.info("[Step 2] Extracting keywords and generating summary...")
            keywords, summary = self.topic_extractor.extract_keywords_and_summary(
                news_result['news_list'], 
                max_keywords=max_keywords
            )
            
            extraction_result['topic_extraction'] = {
                'success': len(keywords) > 0,
                'keywords_count': len(keywords),
                'keywords': keywords,
                'summary': summary
            }
            
            if not keywords:
                logger.warning("Warning: no valid keywords were extracted")
            
            # Step 3: save to database
            logger.info("[Step 3] Saving analysis results to database...")
            save_success = self.db_manager.save_daily_topics(
                keywords, summary, date.today()
            )
            
            extraction_result['database_save'] = {
                'success': save_success
            }
            
            extraction_result['success'] = True
            extraction_result['end_time'] = datetime.now().isoformat()
            
            logger.info("Daily topic extraction workflow completed")
            
            return extraction_result
            
        except Exception as e:
            logger.exception(f"Topic extraction workflow failed: {e}")
            extraction_result['error'] = str(e)
            extraction_result['end_time'] = datetime.now().isoformat()
            return extraction_result
    
    def print_extraction_results(self, extraction_result: Dict):
        """Print extraction results."""
        extraction_result_message = ""
        
        # News collection result
        news_data = extraction_result.get('news_collection', {})
        extraction_result_message += f"\n📰 News collection: {news_data.get('total_news', 0)} items\n"
        extraction_result_message += f"   Successful sources: {news_data.get('successful_sources', 0)}/{news_data.get('total_sources', 0)}\n"
        
        # Topic extraction result
        topic_data = extraction_result.get('topic_extraction', {})
        keywords = topic_data.get('keywords', [])
        summary = topic_data.get('summary', '')
        
        extraction_result_message += f"\n🔑 Extracted keywords: {len(keywords)}\n"
        if keywords:
            # Show 5 keywords per line
            for i in range(0, len(keywords), 5):
                keyword_group = keywords[i:i+5]
                extraction_result_message += f"   {', '.join(keyword_group)}\n"
        
        extraction_result_message += f"\n📝 News summary:\n   {summary}\n"
        
        # Database save result
        db_data = extraction_result.get('database_save', {})
        if db_data.get('success'):
            extraction_result_message += f"\n💾 Database save: success\n"
        else:
            extraction_result_message += f"\n💾 Database save: failed\n"
        
        logger.info(extraction_result_message)
    
    def get_keywords_for_crawling(self, extract_date: date = None) -> List[str]:
        """
        Get keyword list for crawling.
        
        Args:
            extract_date: Extraction date, default is today
            
        Returns:
            Keyword list
        """
        try:
            # Get topic analysis from database
            topics_data = self.db_manager.get_daily_topics(extract_date)
            
            if not topics_data:
                logger.info(f"No topic data found for {extract_date or date.today()}")
                return []
            
            keywords = topics_data['keywords']
            
            # Generate search keywords
            search_keywords = self.topic_extractor.get_search_keywords(keywords)
            
            logger.info(f"Prepared {len(search_keywords)} keywords for crawling")
            return search_keywords
            
        except Exception as e:
            logger.error(f"Failed to get crawling keywords: {e}")
            return []
    
    def get_daily_analysis(self, target_date: date = None) -> Optional[Dict]:
        """Get analysis result for the specified date."""
        try:
            return self.db_manager.get_daily_topics(target_date)
        except Exception as e:
            logger.error(f"Failed to get daily analysis: {e}")
            return None
    
    def get_recent_analysis(self, days: int = 7) -> List[Dict]:
        """Get analysis results from recent days."""
        try:
            return self.db_manager.get_recent_topics(days)
        except Exception as e:
            logger.error(f"Failed to get recent analysis: {e}")
            return []

# ==================== Command-Line Tools ====================

async def run_extraction_command(sources=None, keywords_count=100, show_details=True):
    """Run topic extraction command."""
    
    try:
        async with BroadTopicExtraction() as extractor:
            # Run topic extraction
            result = await extractor.run_daily_extraction(
                news_sources=sources,
                max_keywords=keywords_count
            )
            
            if result['success']:
                if show_details:
                    # Show detailed result
                    extractor.print_extraction_results(result)
                else:
                    # Show concise result
                    news_data = result.get('news_collection', {})
                    topic_data = result.get('topic_extraction', {})
                    
                    logger.info(f"✅ Topic extraction completed successfully")
                    logger.info(f"   Collected news: {news_data.get('total_news', 0)} items")
                    logger.info(f"   Extracted keywords: {len(topic_data.get('keywords', []))}")
                    logger.info(f"   Summary length: {len(topic_data.get('summary', ''))} characters")
                
                # Get crawling keywords
                crawling_keywords = extractor.get_keywords_for_crawling()
                
                if crawling_keywords:
                    logger.info(f"\n🔑 Search keywords prepared for DeepSentimentCrawling:")
                    logger.info(f"   {', '.join(crawling_keywords)}")
                    
                    # Save keywords to file
                    keywords_file = project_root / "data" / "daily_keywords.txt"
                    keywords_file.parent.mkdir(exist_ok=True)
                    
                    with open(keywords_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(crawling_keywords))
                    
                    logger.info(f"   Keywords saved to: {keywords_file}")
                
                return True
                
            else:
                logger.error(f"❌ Topic extraction failed: {result.get('error', 'Unknown error')}")
                return False
                
    except Exception as e:
        logger.error(f"❌ Error during execution: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="MindSpider daily topic extraction tool")
    parser.add_argument("--sources", nargs="+", help="Specify news source platforms",
                       choices=list(SOURCE_NAMES.keys()))
    parser.add_argument("--keywords", type=int, default=100, help="Maximum keyword count (default: 100)")
    parser.add_argument("--quiet", action="store_true", help="Concise output mode")
    parser.add_argument("--list-sources", action="store_true", help="List supported news sources")
    
    args = parser.parse_args()
    
    # List supported news sources
    if args.list_sources:
        logger.info("Supported news source platforms:")
        for source, name in SOURCE_NAMES.items():
            logger.info(f"  {source:<25} {name}")
        return
    
    # Validate arguments
    if args.keywords < 1 or args.keywords > 200:
        logger.error("Keyword count must be between 1 and 200")
        sys.exit(1)
    
    # Run extraction
    try:
        success = asyncio.run(run_extraction_command(
            sources=args.sources,
            keywords_count=args.keywords,
            show_details=not args.quiet
        ))
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
