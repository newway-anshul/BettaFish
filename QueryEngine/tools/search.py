"""
Public-opinion search toolkit for AI Agents (Tavily)

Version: 1.5
Last Updated: 2025-08-22

This script decomposes complex Tavily search capabilities into a set of focused,
low-parameter standalone tools designed for AI Agent usage.
The Agent only needs to choose the right tool based on task intent,
without understanding complex parameter combinations. All tools default to
news-oriented usage.

New Features:
- Added `basic_search_news` for standard, general-purpose news search.
- Each search result now includes `published_date` (news publish date).

Primary Tools:
- basic_search_news: (new) Perform standard, fast, general news search.
- deep_search_news: Perform the most comprehensive deep analysis for a topic.
- search_news_last_24_hours: Get the latest updates from the last 24 hours.
- search_news_last_week: Get major coverage from the past week.
- search_images_for_news: Find images related to a news topic.
- search_news_by_date: Search within a specified historical date range.
"""

import os
import sys
from typing import List, Dict, Any, Optional

# Add utils directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
utils_dir = os.path.join(root_dir, 'utils')
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

from retry_helper import with_graceful_retry, SEARCH_API_RETRY_CONFIG
from dataclasses import dataclass, field

# Ensure the Tavily package is installed before running: pip install tavily-python
try:
    from tavily import TavilyClient
except ImportError:
    raise ImportError("Tavily package is not installed. Please run `pip install tavily-python`.")

# --- 1. Data Structure Definitions ---

@dataclass
class SearchResult:
    """
    Web search result data class
    Includes published_date to store the news publication date
    """
    title: str
    url: str
    content: str
    score: Optional[float] = None
    raw_content: Optional[str] = None
    published_date: Optional[str] = None

@dataclass
class ImageResult:
    """Image search result data class"""
    url: str
    description: Optional[str] = None

@dataclass
class TavilyResponse:
    """Encapsulates the full Tavily API response for passing between tools"""
    query: str
    answer: Optional[str] = None
    results: List[SearchResult] = field(default_factory=list)
    images: List[ImageResult] = field(default_factory=list)
    response_time: Optional[float] = None


# --- 2. Core Client and Specialized Toolset ---

class TavilyNewsAgency:
    """
    A client containing multiple specialized news/public-opinion search tools.
    Each public method is designed as an independently callable tool for AI Agents.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize client.
        Args:
            api_key: Tavily API key. If not provided, reads from TAVILY_API_KEY.
        """
        if api_key is None:
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                raise ValueError("Tavily API Key not found. Set TAVILY_API_KEY or pass it during initialization")
        self._client = TavilyClient(api_key=api_key)

    @with_graceful_retry(SEARCH_API_RETRY_CONFIG, default_return=TavilyResponse(query="Search failed"))
    def _search_internal(self, **kwargs) -> TavilyResponse:
        """Internal generic search executor; all tools eventually call this method"""
        try:
            kwargs['topic'] = 'general'
            api_params = {k: v for k, v in kwargs.items() if v is not None}
            response_dict = self._client.search(**api_params)
            
            search_results = [
                SearchResult(
                    title=item.get('title'),
                    url=item.get('url'),
                    content=item.get('content'),
                    score=item.get('score'),
                    raw_content=item.get('raw_content'),
                    published_date=item.get('published_date')
                ) for item in response_dict.get('results', [])
            ]
            
            image_results = [ImageResult(url=item.get('url'), description=item.get('description')) for item in response_dict.get('images', [])]

            return TavilyResponse(
                query=response_dict.get('query'), answer=response_dict.get('answer'),
                results=search_results, images=image_results,
                response_time=response_dict.get('response_time')
            )
        except Exception as e:
            print(f"Error occurred during search: {str(e)}")
            raise e  # Let the retry mechanism catch and handle it

    # --- Tool methods available to the Agent ---

    def basic_search_news(self, query: str, max_results: int = 7) -> TavilyResponse:
        """
        [Tool] Basic news search: perform a standard, fast news search.
        This is the most commonly used general tool when a specific search type is unclear.
        The Agent provides query and optional max_results.
        """
        print(f"--- TOOL: Basic News Search (query: {query}) ---")
        return self._search_internal(
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_answer=False
        )

    def deep_search_news(self, query: str) -> TavilyResponse:
        """
        [Tool] Deep news analysis: perform the most comprehensive deep search on a topic.
        Returns an AI-generated advanced summary answer and up to 20 most relevant news results.
        Suitable for scenarios requiring full background understanding.
        The Agent only needs to provide query.
        """
        print(f"--- TOOL: Deep News Analysis (query: {query}) ---")
        return self._search_internal(
            query=query, search_depth="advanced", max_results=20, include_answer="advanced"
        )

    def search_news_last_24_hours(self, query: str) -> TavilyResponse:
        """
        [Tool] Search news from last 24 hours: get the latest updates on a topic.
        This tool is specialized for news published in the last 24 hours.
        Suitable for tracking breaking events or latest progress.
        The Agent only needs to provide query.
        """
        print(f"--- TOOL: Search News in Last 24 Hours (query: {query}) ---")
        return self._search_internal(query=query, time_range='d', max_results=10)

    def search_news_last_week(self, query: str) -> TavilyResponse:
        """
        [Tool] Search news from last week: get major reports from the past week on a topic.
        Suitable for weekly public-opinion summaries or reviews.
        The Agent only needs to provide query.
        """
        print(f"--- TOOL: Search News from Last Week (query: {query}) ---")
        return self._search_internal(query=query, time_range='w', max_results=10)

    def search_images_for_news(self, query: str) -> TavilyResponse:
        """
        [Tool] Find news images: search for images related to a news topic.
        Returns image links and descriptions, suitable for reports/articles requiring visuals.
        The Agent only needs to provide query.
        """
        print(f"--- TOOL: Find News Images (query: {query}) ---")
        return self._search_internal(
            query=query, include_images=True, include_image_descriptions=True, max_results=5
        )

    def search_news_by_date(self, query: str, start_date: str, end_date: str) -> TavilyResponse:
        """
        [Tool] Search news by date range: search within a specific historical period.
        This is the only tool requiring detailed time parameters from the Agent.
        Suitable for analyzing specific historical events.
        The Agent must provide query, start_date, and end_date in 'YYYY-MM-DD' format.
        """
        print(f"--- TOOL: Search News by Date Range (query: {query}, from: {start_date}, to: {end_date}) ---")
        return self._search_internal(
            query=query, start_date=start_date, end_date=end_date, max_results=15
        )


# --- 3. Tests and Usage Examples ---

def print_response_summary(response: TavilyResponse):
    """Simplified print helper for showing test results, now includes publish date"""
    if not response or not response.query:
        print("Failed to get a valid response.")
        return
        
    print(f"\nQuery: '{response.query}' | Time: {response.response_time}s")
    if response.answer:
        print(f"AI Summary: {response.answer[:120]}...")
    print(f"Found {len(response.results)} web results, {len(response.images)} images.")
    if response.results:
        first_result = response.results[0]
        date_info = f"(Published: {first_result.published_date})" if first_result.published_date else ""
        print(f"First result: {first_result.title} {date_info}")
    print("-" * 60)


if __name__ == "__main__":
    # Before running, ensure TAVILY_API_KEY is set in your environment
    
    try:
        # Initialize the "news agency" client, which contains all tools internally
        agency = TavilyNewsAgency()

        # Scenario 1: Agent performs a routine, fast search
        response1 = agency.basic_search_news(query="Latest Olympic competition results", max_results=5)
        print_response_summary(response1)

        # Scenario 2: Agent needs a full background on "global chip technology competition"
        response2 = agency.deep_search_news(query="Global chip technology competition")
        print_response_summary(response2)

        # Scenario 3: Agent needs to track the latest updates from "Nvidia GTC"
        response3 = agency.search_news_last_24_hours(query="Nvidia GTC latest announcements")
        print_response_summary(response3)
        
        # Scenario 4: Agent needs materials for a weekly report on "autonomous driving"
        response4 = agency.search_news_last_week(query="Commercial deployment of autonomous driving")
        print_response_summary(response4)
        
        # Scenario 5: Agent needs news images about the "James Webb Space Telescope"
        response5 = agency.search_images_for_news(query="Latest discoveries by the James Webb Space Telescope")
        print_response_summary(response5)

        # Scenario 6: Agent needs to research news about "AI regulation" in Q1 2025
        response6 = agency.search_news_by_date(
            query="AI regulation",
            start_date="2025-01-01",
            end_date="2025-03-31"
        )
        print_response_summary(response6)

    except ValueError as e:
        print(f"Initialization failed: {e}")
        print("Please ensure the TAVILY_API_KEY environment variable is set correctly.")
    except Exception as e:
        print(f"An unknown error occurred during testing: {e}")