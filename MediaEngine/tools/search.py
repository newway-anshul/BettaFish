"""
Multimodal Search Toolkit Designed for AI Agents (Bocha)

Version: 1.1
Last Updated: 2025-08-22

This script breaks down the complex Bocha AI Search capabilities into a series of
independent tools with clear purposes and minimal parameters, specifically designed
for AI Agent use. The agent only needs to choose the appropriate tool based on the
task intent, such as general search, structured data lookup, or time-sensitive news,
without needing to understand complex parameter combinations.

Core Features:
- Powerful multimodal capabilities: can return web pages, images, AI summaries,
    follow-up suggestions, and rich structured "modal card" data at the same time.
- Modal card support: for specific queries such as weather, stocks, exchange rates,
    encyclopedia entries, and healthcare topics, structured data cards can be returned
    directly for easy parsing and use by the agent.

Main Tools:
- comprehensive_search: performs a full search and returns web pages, images,
    AI summaries, and possible modal cards.
- search_for_structured_data: specifically used to query structured information such
    as weather, stocks, and exchange rates that may trigger modal cards.
- web_search_only: performs a web-only search without requesting an AI summary,
    making it faster.
- search_last_24_hours: retrieves the latest information from the last 24 hours.
- search_last_week: retrieves the major reports from the past week.
"""

import os
import json
import sys
import datetime
from typing import List, Dict, Any, Optional, Literal

from loguru import logger
from config import settings

# Make sure the requests library is installed before running: pip install requests
try:
    import requests
except ImportError:
    raise ImportError("The requests library is not installed. Run `pip install requests` to install it.")

# Add the utils directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
utils_dir = os.path.join(root_dir, 'utils')
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

from retry_helper import with_graceful_retry, SEARCH_API_RETRY_CONFIG

# --- 1. Data Structure Definitions ---
from dataclasses import dataclass, field

@dataclass
class WebpageResult:
    """Web search result."""
    name: str
    url: str
    snippet: str
    display_url: Optional[str] = None
    date_last_crawled: Optional[str] = None

@dataclass
class ImageResult:
    """Image search result."""
    name: str
    content_url: str
    host_page_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None

@dataclass
class ModalCardResult:
    """
    Structured modal card result.
    This is a core Bocha Search feature used to return specific kinds of structured information.
    """
    card_type: str  # Example: weather_china, stock, baike_pro, medical_common
    content: Dict[str, Any]  # Parsed JSON content

@dataclass
class BochaResponse:
    """Encapsulates the full Bocha API response so it can be passed between tools."""
    query: str
    conversation_id: Optional[str] = None
    answer: Optional[str] = None  # AI-generated summary answer
    follow_ups: List[str] = field(default_factory=list) # AI-generated follow-up questions
    webpages: List[WebpageResult] = field(default_factory=list)
    images: List[ImageResult] = field(default_factory=list)
    modal_cards: List[ModalCardResult] = field(default_factory=list)

@dataclass
class AnspireResponse:
    """Encapsulates the full Anspire API response so it can be passed between tools."""
    query: str
    conversation_id: Optional[str] = None
    score: Optional[float] = None
    webpages: List[WebpageResult] = field(default_factory=list)


# --- 2. Core Clients and Specialized Toolsets ---

class BochaMultimodalSearch:
    """
    A client that contains multiple specialized multimodal search tools.
    Each public method is designed to be used independently by an AI Agent.
    """

    BOCHA_BASE_URL = settings.BOCHA_BASE_URL or "https://api.bocha.cn/v1/ai-search"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the client.
        Args:
            api_key: Bocha API key. If not provided, it will be read from the BOCHA_API_KEY environment variable.
        """
        if api_key is None:
            api_key = settings.BOCHA_WEB_SEARCH_API_KEY
            if not api_key:
                raise ValueError("Bocha API Key not found. Set the BOCHA_API_KEY environment variable or provide it during initialization")

        self._headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'Accept': '*/*'
        }

    def _parse_search_response(self, response_dict: Dict[str, Any], query: str) -> BochaResponse:
        """Parse a structured BochaResponse object from the raw API dictionary response."""

        final_response = BochaResponse(query=query)
        final_response.conversation_id = response_dict.get('conversation_id')

        messages = response_dict.get('messages', [])
        for msg in messages:
            role = msg.get('role')
            if role != 'assistant':
                continue

            msg_type = msg.get('type')
            content_type = msg.get('content_type')
            content_str = msg.get('content', '{}')

            try:
                content_data = json.loads(content_str)
            except json.JSONDecodeError:
                # If the content is not a valid JSON string (for example, a plain text answer), use it directly.
                content_data = content_str

            if msg_type == 'answer' and content_type == 'text':
                final_response.answer = content_data

            elif msg_type == 'follow_up' and content_type == 'text':
                final_response.follow_ups.append(content_data)

            elif msg_type == 'source':
                if content_type == 'webpage':
                    web_results = content_data.get('value', [])
                    for item in web_results:
                        final_response.webpages.append(WebpageResult(
                            name=item.get('name'),
                            url=item.get('url'),
                            snippet=item.get('snippet'),
                            display_url=item.get('displayUrl'),
                            date_last_crawled=item.get('dateLastCrawled')
                        ))
                elif content_type == 'image':
                    final_response.images.append(ImageResult(
                        name=content_data.get('name'),
                        content_url=content_data.get('contentUrl'),
                        host_page_url=content_data.get('hostPageUrl'),
                        thumbnail_url=content_data.get('thumbnailUrl'),
                        width=content_data.get('width'),
                        height=content_data.get('height')
                    ))
                # Treat all other content_type values as modal cards.
                else:
                    final_response.modal_cards.append(ModalCardResult(
                        card_type=content_type,
                        content=content_data
                    ))

        return final_response


    @with_graceful_retry(SEARCH_API_RETRY_CONFIG, default_return=BochaResponse(query="Search failed"))
    def _search_internal(self, **kwargs) -> BochaResponse:
        """Internal shared search executor used by all tools."""
        query = kwargs.get("query", "Unknown Query")
        payload = {
            "stream": False,  # Agent tools usually use non-streaming mode to get the full result.
        }
        payload.update(kwargs)

        try:

            response = requests.post(self.BOCHA_BASE_URL, headers=self._headers, json=payload, timeout=30)
            response.raise_for_status()  # Raise an exception if the HTTP status code is 4xx or 5xx.

            response_dict = response.json()
            if response_dict.get("code") != 200:
                logger.error(f"API returned an error: {response_dict.get('msg', 'Unknown error')}")
                return BochaResponse(query=query)

            return self._parse_search_response(response_dict, query)

        except requests.exceptions.RequestException as e:
            logger.exception(f"A network error occurred during search: {str(e)}")
            raise e  # Let the retry mechanism catch and handle it.
        except Exception as e:
            logger.exception(f"An unknown error occurred while processing the response: {str(e)}")
            raise e  # Let the retry mechanism catch and handle it.

    # --- Tool Methods Available to the Agent ---

    def comprehensive_search(self, query: str, max_results: int = 10) -> BochaResponse:
        """
        [Tool] Comprehensive search: performs a standard search that includes all types of information.
        Returns web pages, images, an AI summary, follow-up suggestions, and possible modal cards.
        This is the most commonly used general-purpose search tool.
        The agent can provide a search query and an optional maximum number of results.
        """
        logger.info(f"--- TOOL: Comprehensive search (query: {query}) ---")
        return self._search_internal(
            query=query,
            count=max_results,
            answer=True  # Enable AI summary
        )

    def web_search_only(self, query: str, max_results: int = 15) -> BochaResponse:
        """
        [Tool] Web-only search: retrieves only web links and snippets without requesting an AI-generated answer.
        Suitable when raw web information is needed quickly without additional AI analysis.
        Faster and lower cost.
        """
        logger.info(f"--- TOOL: Web-only search (query: {query}) ---")
        return self._search_internal(
            query=query,
            count=max_results,
            answer=False # Disable AI summary
        )

    def search_for_structured_data(self, query: str) -> BochaResponse:
        """
        [Tool] Structured data lookup: specifically used for queries that may trigger modal cards.
        When the agent intends to query structured information such as weather, stocks,
        exchange rates, encyclopedia definitions, train tickets, or car specifications,
        this tool should be preferred.
        It returns all information, but the agent should focus on the modal_cards section.
        """
        logger.info(f"--- TOOL: Structured data lookup (query: {query}) ---")
        # Implementation-wise it is the same as comprehensive_search, but the name and docs guide agent intent.
        return self._search_internal(
            query=query,
            count=5, # Structured queries usually do not need many web results.
            answer=True
        )

    def search_last_24_hours(self, query: str) -> BochaResponse:
        """
        [Tool] Search information from the last 24 hours: gets the latest updates on a topic.
        This tool specifically looks for content published within the past 24 hours.
        Suitable for tracking breaking events or the latest developments.
        """
        logger.info(f"--- TOOL: Search last 24 hours (query: {query}) ---")
        return self._search_internal(query=query, freshness='oneDay', answer=True)

    def search_last_week(self, query: str) -> BochaResponse:
        """
        [Tool] Search information from the last week: gets the main reports on a topic from the past week.
        Suitable for weekly public opinion summaries or reviews.
        """
        logger.info(f"--- TOOL: Search last week (query: {query}) ---")
        return self._search_internal(query=query, freshness='oneWeek', answer=True)

class AnspireAISearch:
    """
    Anspire AI Search client.
    """
    ANSPIRE_BASE_URL = settings.ANSPIRE_BASE_URL or "https://plugin.anspire.cn/api/ntsearch/search"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the client.
        Args:
            api_key: Anspire API key. If not provided, it will be read from the ANSPIRE_API_KEY environment variable.
        """
        if api_key is None:
            api_key = settings.ANSPIRE_API_KEY
            if not api_key:
                raise ValueError("Anspire API Key not found. Set the ANSPIRE_API_KEY environment variable or provide it during initialization")

        self._headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'Connection': 'keep-alive',
            'Accept': '*/*'
        }

    def _parse_search_response(self, response_dict: Dict[str, Any], query: str) -> AnspireResponse:
        final_response = AnspireResponse(query=query)
        final_response.conversation_id = response_dict.get('Uuid')

        messages = response_dict.get("results", [])
        for msg in messages:
            final_response.score = msg.get("score")
            final_response.webpages.append(WebpageResult(
                name = msg.get("title", ""),
                url = msg.get("url", ""),
                snippet = msg.get("content", ""),
                date_last_crawled = msg.get("date", None)
            ))

        return final_response
    
    @with_graceful_retry(SEARCH_API_RETRY_CONFIG, default_return=AnspireResponse(query="Search failed"))
    def _search_internal(self, **kwargs) -> AnspireResponse:
        """Internal shared search executor used by all tools."""
        query = kwargs.get("query", "Unknown Query")
        payload = {
            "query": query,
            "top_k": kwargs.get("top_k", 10),
            "Insite": kwargs.get("Insite", ""),
            "FromTime": kwargs.get("FromTime", ""),
            "ToTime": kwargs.get("ToTime", "")
        }
        
        try:
            response = requests.get(self.ANSPIRE_BASE_URL, headers=self._headers, params=payload, timeout=30)
            response.raise_for_status()  # Raise an exception if the HTTP status code is 4xx or 5xx.

            response_dict = response.json()
            return self._parse_search_response(response_dict, query)
        except requests.exceptions.RequestException as e:
            logger.exception(f"A network error occurred during search: {str(e)}")
            raise e  # Let the retry mechanism catch and handle it.
        except Exception as e:
            logger.exception(f"An unknown error occurred while processing the response: {str(e)}")
            raise e  # Let the retry mechanism catch and handle it.
    
    def comprehensive_search(self, query: str, max_results: int = 10) -> AnspireResponse:
        """
        [Tool] Comprehensive search: gets broad information on a topic, including web pages.
        Suitable for scenarios that need multiple information sources.
        """
        logger.info(f"--- TOOL: Comprehensive search (query: {query}) ---")
        return self._search_internal(
            query=query,
            top_k=max_results
        )

    def search_last_24_hours(self, query: str, max_results: int = 10) -> AnspireResponse:
        """
        [Tool] Search information from the last 24 hours: gets the latest updates on a topic.
        This tool specifically looks for content published within the past 24 hours.
        Suitable for tracking breaking events or the latest developments.
        """
        logger.info(f"--- TOOL: Search last 24 hours (query: {query}) ---")
        to_time = datetime.datetime.now()
        from_time = to_time - datetime.timedelta(days=1)
        return self._search_internal(query=query,
                                     top_k=max_results,
                                     FromTime=from_time.strftime("%Y-%m-%d %H:%M:%S"), 
                                     ToTime=to_time.strftime("%Y-%m-%d %H:%M:%S"))

    def search_last_week(self, query: str, max_results: int = 10) -> AnspireResponse:
        """
        [Tool] Search information from the last week: gets the main reports on a topic from the past week.
        Suitable for weekly public opinion summaries or reviews.
        """
        logger.info(f"--- TOOL: Search last week (query: {query}) ---")
        to_time = datetime.datetime.now()
        from_time = to_time - datetime.timedelta(weeks=1)
        return self._search_internal(query=query,
                                     top_k=max_results,
                                     FromTime=from_time.strftime("%Y-%m-%d %H:%M:%S"),
                                     ToTime=to_time.strftime("%Y-%m-%d %H:%M:%S"))


# --- 3. Tests and Usage Examples ---
def load_agent_from_config():
    """Select and load the search agent based on the configuration."""
    if settings.BOCHA_WEB_SEARCH_API_KEY:
        logger.info("Loading BochaMultimodalSearch Agent")
        return BochaMultimodalSearch()
    elif settings.ANSPIRE_API_KEY:
        logger.info("Loading AnspireAISearch Agent")
        return AnspireAISearch()
    else:
        raise ValueError("No valid search agent is configured")

def print_response_summary(response):
    """Simplified print function used to display test results."""
    if not response or not response.query:
        logger.error("Failed to obtain a valid response.")
        return

    logger.info(f"\nQuery: '{response.query}' | Conversation ID: {response.conversation_id}")
    if hasattr(response, 'answer') and response.answer:
        logger.info(f"AI summary: {response.answer[:150]}...")

    logger.info(f"Found {len(response.webpages)} web pages")
    if hasattr(response, 'images'):
        logger.info(f"Found {len(response.images)} images")
    if hasattr(response, 'modal_cards'):
        logger.info(f"Found {len(response.modal_cards)} modal cards")

    if hasattr(response, 'modal_cards') and response.modal_cards:
        first_card = response.modal_cards[0]
        logger.info(f"First modal card type: {first_card.card_type}")

    if response.webpages:
        first_result = response.webpages[0]
        logger.info(f"First web result: {first_result.name}")

    if hasattr(response, 'follow_ups') and response.follow_ups:
        logger.info(f"Suggested follow-ups: {response.follow_ups}")

    logger.info("-" * 60)


if __name__ == "__main__":
    # Make sure the BOCHA_API_KEY environment variable is set before running.

    try:
        # Initialize the multimodal search client, which contains all tools internally.
        search_client = load_agent_from_config()

        # Scenario 1: The agent performs a routine comprehensive search that requires an AI summary.
        response1 = search_client.comprehensive_search(query="The impact of artificial intelligence on the future of education")
        print_response_summary(response1)

        # Scenario 2: The agent needs to query specific structured information - weather.
        if isinstance(search_client, BochaMultimodalSearch):
            response2 = search_client.search_for_structured_data(query="What will the weather be like in Shanghai tomorrow?")
            print_response_summary(response2)
            # Deeply inspect the first modal card.
            if response2.modal_cards and response2.modal_cards[0].card_type == 'weather_china':
                logger.info("Weather modal card details:", json.dumps(response2.modal_cards[0].content, indent=2, ensure_ascii=False))


        # Scenario 3: The agent needs to query specific structured information - stock data.
        if isinstance(search_client, BochaMultimodalSearch):
            response3 = search_client.search_for_structured_data(query="Eastmoney stock")
            print_response_summary(response3)

        # Scenario 4: The agent needs to track the latest developments of an event.
        response4 = search_client.search_last_24_hours(query="Latest updates on the C929 large aircraft")
        print_response_summary(response4)

        # Scenario 5: The agent only needs to quickly obtain web information without an AI summary.
        if isinstance(search_client, BochaMultimodalSearch):
            response5 = search_client.web_search_only(query="How to use Python dataclasses")
            print_response_summary(response5)

        # Scenario 6: The agent needs to review one week of news about a technology.
        response6 = search_client.search_last_week(query="Commercialization of quantum computing")
        print_response_summary(response6)

        '''Below is example output from the test program:
        --- TOOL: Comprehensive search (query: The impact of artificial intelligence on the future of education) ---

Query: 'The impact of artificial intelligence on the future of education' | Conversation ID: bf43bfe4c7bb4f7b8a3945515d8ab69e
AI summary: Artificial intelligence is influencing the future of education in many ways.

From a positive perspective:
- In terms of teaching resources, artificial intelligence helps promote a more balanced distribution of educational resources [Ref:4]. For example, AI cloud platforms can enable the sharing of high-quality resources, which is especially meaningful for remote areas. Students there can gain access to better educational content, which can partly relieve teacher shortages through AI-driven teaching assistants or virtual...
Found 10 web pages, 1 image, and 1 modal card.
First modal card type: video
First web result: How artificial intelligence is transforming education
Suggested follow-ups: [['How will artificial intelligence change future education models?', 'What challenges will artificial intelligence bring to teachers in future education?', 'How can students use artificial intelligence to improve learning outcomes in future education?']]
------------------------------------------------------------
--- TOOL: Structured data lookup (query: What will the weather be like in Shanghai tomorrow?) ---

Query: 'What will the weather be like in Shanghai tomorrow?' | Conversation ID: e412aa1548cd43a295430e47a62adda2
AI summary: Based on the provided information, it is not possible to determine tomorrow's weather in Shanghai.

First, the available information is all about the weather conditions on August 22, 2025, including temperature, precipitation, wind, humidity, and high-temperature alerts [Ref:1][Ref:2][Ref:3][Ref:5]. However, it does not include a forecast for tomorrow, August 23. Although the subtropical high and continued heat through the end of August are mentioned...
Found 5 web pages, 1 image, and 2 modal cards.
First modal card type: video
First web result: Temperatures may hit 38 today! Shanghai's August hot-day count and summer heat streak may both break records_weather_low pressure_weather station
Suggested follow-ups: [['Can you tell me the temperature range in Shanghai tomorrow?', 'Will it rain in Shanghai tomorrow?', 'Will the weather in Shanghai tomorrow be sunny or cloudy?']]
------------------------------------------------------------
--- TOOL: Structured data lookup (query: Eastmoney stock) ---

Query: 'Eastmoney stock' | Conversation ID: 584d62ed97834473b967127852e1eaa0
AI summary: Based only on the provided context, it is not possible to obtain exact information about Eastmoney stock.

From the available data, there is no direct information specifically describing Eastmoney stock. For example, there are no exact figures for price movement, trading volume, or market capitalization [Ref:1][Ref:3]. There is also no information related to research reports or ratings for Eastmoney stock [Ref:2]. At the same time, the surrounding context discussing stock prices and transactions...
Found 5 web pages, 1 image, and 2 modal cards.
First modal card type: video
First web result: Stock price_intraday transactions_market_quotes_chart trends - Eastmoney
Suggested follow-ups: [['How has Eastmoney stock performed recently?', 'What are the main investment highlights of Eastmoney stock?', 'What were the historical highest and lowest prices of Eastmoney stock?']]
------------------------------------------------------------
--- TOOL: Search last 24 hours (query: Latest updates on the C929 large aircraft) ---

Query: 'Latest updates on the C929 large aircraft' | Conversation ID: 5904021dc29d497e938e04db18d7f2e2
AI summary: Based on the provided context, there is no direct news about the C929 large aircraft, so no exact latest update can be given.

The provided context covers many aviation-related events, but most of them focus on topics such as personnel changes involving Boeing 787 and Airbus A380 experts, the domestic aircraft "C909 cloud journey," Kede Numerical Control revenue, Russian aero-engine supply issues, and other topics unrelated to the C929 large aircraft...
Found 10 web pages, 1 image, and 1 modal card.
First modal card type: video
First web result: Gave up a million-dollar US salary, top Boeing 787 expert returns to China and may help break through C929 challenges
Suggested follow-ups: [['What is the current development progress of the C929 large aircraft?', 'Is there any news about the expected first flight time of the C929 large aircraft?', 'What new technological advances does the C929 large aircraft have?']]
------------------------------------------------------------
--- TOOL: Web-only search (query: How to use Python dataclasses) ---

Query: 'How to use Python dataclasses' | Conversation ID: 74c742759d2e4b17b52d8b735ce24537
Found 15 web pages, 1 image, and 1 modal card.
First modal card type: video
First web result: Essential dataclasses knowledge: Python tips_python dataclasses - CSDN Blog
------------------------------------------------------------
--- TOOL: Search last week (query: Commercialization of quantum computing) ---

AI summary: The commercialization of quantum computing is gradually advancing.

Quantum computing commercialization is being reflected and driven in many ways. Internationally, the US Department of Energy's Oak Ridge National Laboratory selected IQM Radiance as its first locally deployed quantum computer, with delivery planned for the third quarter of 2025 and integration into a high-performance computing system [Ref:4]. Meanwhile, UK quantum computing company Oxford Ionics and its full-stack trapped-ion quantum computing...
Found 10 web pages, 1 image, and 1 modal card.
First modal card type: video
First web result: The commercial potential of quantum computing is accelerating, and WiMi Hologram (WIMI.US) is staking out an innovation ecosystem high ground
Suggested follow-ups: [['What successful cases of quantum computing commercialization exist so far?', 'Which companies are driving the commercialization of quantum computing?', 'What are the main challenges facing the commercialization of quantum computing?']]
------------------------------------------------------------'''

    except ValueError as e:
        logger.exception(f"Initialization failed: {e}")
        logger.error("Make sure the BOCHA_API_KEY environment variable is set correctly.")
    except Exception as e:
        logger.exception(f"An unknown error occurred during testing: {e}")