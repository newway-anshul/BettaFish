"""
Deep Search Agent main class
Integrates all modules to implement a complete deep-search workflow
"""

import json
import os
import re
from datetime import datetime
from typing import Optional, Dict, Any, List
from loguru import logger
from .llms import LLMClient
from .nodes import (
    ReportStructureNode,
    FirstSearchNode, 
    ReflectionNode,
    FirstSummaryNode,
    ReflectionSummaryNode,
    ReportFormattingNode
)
from .state import State
from .tools import BochaMultimodalSearch, BochaResponse, AnspireAISearch, AnspireResponse
from .utils import settings, Settings, format_search_results_for_prompt


class DeepSearchAgent:
    """Deep Search Agent main class"""
    
    def __init__(self, config: Optional[Settings] = None):
        """
        Initialize Deep Search Agent
        
        Args:
            config: Configuration object; auto-loaded if not provided
        """
        self.config = config or settings
        
        # Initialize LLM client
        self.llm_client = self._initialize_llm()
        
        # Initialize search toolset
        self.search_agency = BochaMultimodalSearch(api_key=(self.config.BOCHA_API_KEY or self.config.BOCHA_WEB_SEARCH_API_KEY))
        
        # Initialize nodes
        self._initialize_nodes()
        
        # State
        self.state = State()
        
        # Ensure output directory exists
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        
        logger.info(f"Media Agent initialized")
        logger.info(f"Using LLM: {self.llm_client.get_model_info()}")
        logger.info(f"Search toolset: BochaMultimodalSearch (supports 5 multimodal search tools)")
    
    def _initialize_llm(self) -> LLMClient:
        """Initialize LLM client"""
        return LLMClient(
            api_key=(self.config.MEDIA_ENGINE_API_KEY or self.config.MINDSPIDER_API_KEY),
            model_name=(self.config.MEDIA_ENGINE_MODEL_NAME or self.config.MINDSPIDER_MODEL_NAME),
            base_url=(self.config.MEDIA_ENGINE_BASE_URL or self.config.MINDSPIDER_BASE_URL),
        )
    
    def _initialize_nodes(self):
        """Initialize processing nodes"""
        self.first_search_node = FirstSearchNode(self.llm_client)
        self.reflection_node = ReflectionNode(self.llm_client)
        self.first_summary_node = FirstSummaryNode(self.llm_client)
        self.reflection_summary_node = ReflectionSummaryNode(self.llm_client)
        self.report_formatting_node = ReportFormattingNode(self.llm_client)
    
    def _validate_date_format(self, date_str: str) -> bool:
        """
        Validate whether date format is YYYY-MM-DD
        
        Args:
            date_str: Date string
            
        Returns:
            Whether the format is valid
        """
        if not date_str:
            return False
        
        # Check format
        pattern = r'^\d{4}-\d{2}-\d{2}$'
        if not re.match(pattern, date_str):
            return False
        
        # Check whether date is valid
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False
    
    def execute_search_tool(self, tool_name: str, query: str, **kwargs) -> BochaResponse:
        """
        Execute the specified search tool
        
        Args:
            tool_name: Tool name. Available values:
                - "comprehensive_search": comprehensive search (default)
                - "web_search_only": web-only search
                - "search_for_structured_data": structured data search
                - "search_last_24_hours": latest information from last 24 hours
                - "search_last_week": information from last week
            query: Search query
            **kwargs: Additional parameters (such as max_results)
            
        Returns:
            BochaResponse object
        """
        logger.info(f"  → Executing search tool: {tool_name}")
        
        if tool_name == "comprehensive_search":
            max_results = kwargs.get("max_results", 10)
            return self.search_agency.comprehensive_search(query, max_results)
        elif tool_name == "web_search_only":
            max_results = kwargs.get("max_results", 15)
            return self.search_agency.web_search_only(query, max_results)
        elif tool_name == "search_for_structured_data":
            return self.search_agency.search_for_structured_data(query)
        elif tool_name == "search_last_24_hours":
            return self.search_agency.search_last_24_hours(query)
        elif tool_name == "search_last_week":
            return self.search_agency.search_last_week(query)
        else:
            logger.info(f"  ⚠️  Unknown search tool: {tool_name}, using default comprehensive search")
            return self.search_agency.comprehensive_search(query)
    
    def research(self, query: str, save_report: bool = True) -> str:
        """
        Run deep research
        
        Args:
            query: Research query
            save_report: Whether to save report to file
            
        Returns:
            Final report content
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting deep research: {query}")
        logger.info(f"{'='*60}")
        
        try:
            # Step 1: Generate report structure
            self._generate_report_structure(query)
            
            # Step 2: Process each paragraph
            self._process_paragraphs()
            
            # Step 3: Generate final report
            final_report = self._generate_final_report()
            
            # Step 4: Save report
            if save_report:
                self._save_report(final_report)
            
            logger.info(f"\n{'='*60}")
            logger.info("Deep research completed!")
            logger.info(f"{'='*60}")
            
            return final_report
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"Error occurred during research: {str(e)} \nStack trace: {error_traceback}")
            raise e
    
    def _generate_report_structure(self, query: str):
        """Generate report structure"""
        logger.info(f"\n[Step 1] Generating report structure...")
        
        # Create report structure node
        report_structure_node = ReportStructureNode(self.llm_client, query)
        
        # Generate structure and update state
        self.state = report_structure_node.mutate_state(state=self.state)
        
        _message = f"Report structure generated with {len(self.state.paragraphs)} paragraphs:"
        for i, paragraph in enumerate(self.state.paragraphs, 1):
            _message += f"\n  {i}. {paragraph.title}"
        logger.info(_message)
    
    def _process_paragraphs(self):
        """Process all paragraphs"""
        total_paragraphs = len(self.state.paragraphs)
        
        for i in range(total_paragraphs):
            logger.info(f"\n[Step 2.{i+1}] Processing paragraph: {self.state.paragraphs[i].title}")
            logger.info("-" * 50)
            
            # Initial search and summary
            self._initial_search_and_summary(i)
            
            # Reflection loop
            self._reflection_loop(i)
            
            # Mark paragraph as completed
            self.state.paragraphs[i].research.mark_completed()
            
            progress = (i + 1) / total_paragraphs * 100
            logger.info(f"Paragraph processing completed ({progress:.1f}%)")
    
    def _initial_search_and_summary(self, paragraph_index: int):
        """Execute initial search and summary"""
        paragraph = self.state.paragraphs[paragraph_index]
        
        # Prepare search input
        search_input = {
            "title": paragraph.title,
            "content": paragraph.content
        }
        
        # Generate search query and tool selection
        logger.info("  - Generating search query...")
        search_output = self.first_search_node.run(search_input)
        search_query = search_output["search_query"]
        search_tool = search_output.get("search_tool", "comprehensive_search")  # Default tool
        reasoning = search_output["reasoning"]
        
        logger.info(f"  - Search query: {search_query}")
        logger.info(f"  - Selected tool: {search_tool}")
        logger.info(f"  - Reasoning: {reasoning}")
        
        # Execute search
        logger.info("  - Executing web search...")
        
        # Handle special parameters (new toolset does not require date parameter handling)
        search_kwargs = {}
        if search_tool in ["comprehensive_search", "web_search_only"]:
            # These tools support the max_results parameter
            search_kwargs["max_results"] = 10
        
        search_response = self.execute_search_tool(search_tool, search_query, **search_kwargs)
        
        # Convert to compatible format
        search_results = []
        if search_response and search_response.webpages:
            # Each tool can return a different number of results; cap at top 10
            max_results = min(len(search_response.webpages), 10)
            for result in search_response.webpages[:max_results]:
                search_results.append({
                    'title': result.name,
                    'url': result.url,
                    'content': result.snippet,
                    'score': None,  # Bocha API does not provide score
                    'raw_content': result.snippet,
                    'published_date': result.date_last_crawled  # Use crawl date
                })
        
        if search_results:
            _message = f"  - Found {len(search_results)} search results" 
            for j, result in enumerate(search_results, 1):
                date_info = f" (Published: {result.get('published_date', 'N/A')})" if result.get('published_date') else ""
                _message += f"\n    {j}. {result['title'][:50]}...{date_info}"
            logger.info(_message)
        else:
            logger.info("  - No search results found")
        
        # Update search history in state
        paragraph.research.add_search_results(
            search_query,
            search_results,
            search_tool=search_tool,
            paragraph_title=paragraph.title,
        )
        
        # Generate initial summary
        logger.info("  - Generating initial summary...")
        summary_input = {
            "title": paragraph.title,
            "content": paragraph.content,
            "search_query": search_query,
            "search_results": format_search_results_for_prompt(
                search_results, self.config.SEARCH_CONTENT_MAX_LENGTH
            )
        }
        
        # Update state
        self.state = self.first_summary_node.mutate_state(
            summary_input, self.state, paragraph_index
        )
        
        logger.info("  - Initial summary completed")
    
    def _reflection_loop(self, paragraph_index: int):
        """Execute reflection loop"""
        paragraph = self.state.paragraphs[paragraph_index]
        
        for reflection_i in range(self.config.MAX_REFLECTIONS):
            logger.info(f"  - Reflection {reflection_i + 1}/{self.config.MAX_REFLECTIONS}...")
            
            # Prepare reflection input
            reflection_input = {
                "title": paragraph.title,
                "content": paragraph.content,
                "paragraph_latest_state": paragraph.research.latest_summary
            }
            
            # Generate reflection search query
            reflection_output = self.reflection_node.run(reflection_input)
            search_query = reflection_output["search_query"]
            search_tool = reflection_output.get("search_tool", "comprehensive_search")  # Default tool
            reasoning = reflection_output["reasoning"]
            
            logger.info(f"    Reflection query: {search_query}")
            logger.info(f"    Selected tool: {search_tool}")
            logger.info(f"    Reflection reasoning: {reasoning}")
            
            # Execute reflection search
            # Handle special parameters
            search_kwargs = {}
            if search_tool in ["comprehensive_search", "web_search_only"]:
                # These tools support the max_results parameter
                search_kwargs["max_results"] = 10
            
            search_response = self.execute_search_tool(search_tool, search_query, **search_kwargs)
            
            # Convert to compatible format
            search_results = []
            if search_response and search_response.webpages:
                # Each tool can return a different number of results; cap at top 10
                max_results = min(len(search_response.webpages), 10)
                for result in search_response.webpages[:max_results]:
                    search_results.append({
                        'title': result.name,
                        'url': result.url,
                        'content': result.snippet,
                        'score': None,  # Bocha API does not provide score
                        'raw_content': result.snippet,
                        'published_date': result.date_last_crawled
                    })
            
            if search_results:
                _message = f"    Found {len(search_results)} reflection search results"
                for j, result in enumerate(search_results, 1):
                    date_info = f" (Published: {result.get('published_date', 'N/A')})" if result.get('published_date') else ""
                    _message += f"\n      {j}. {result['title'][:50]}...{date_info}"
                logger.info(_message)
            else:
                logger.info("    No reflection search results found")
            
            # Update search history
            paragraph.research.add_search_results(
                search_query,
                search_results,
                search_tool=search_tool,
                paragraph_title=paragraph.title,
            )
            
            # Generate reflection summary
            reflection_summary_input = {
                "title": paragraph.title,
                "content": paragraph.content,
                "search_query": search_query,
                "search_results": format_search_results_for_prompt(
                    search_results, self.config.SEARCH_CONTENT_MAX_LENGTH
                ),
                "paragraph_latest_state": paragraph.research.latest_summary
            }
            
            # Update state
            self.state = self.reflection_summary_node.mutate_state(
                reflection_summary_input, self.state, paragraph_index
            )
            
            logger.info(f"    Reflection {reflection_i + 1} completed")
    
    def _generate_final_report(self) -> str:
        """Generate final report"""
        logger.info(f"\n[Step 3] Generating final report...")
        
        # Prepare report data
        report_data = []
        for paragraph in self.state.paragraphs:
            report_data.append({
                "title": paragraph.title,
                "paragraph_latest_state": paragraph.research.latest_summary
            })
        
        # Format report
        try:
            final_report = self.report_formatting_node.run(report_data)
        except Exception as e:
            logger.info(f"LLM formatting failed, using fallback method: {str(e)}")
            final_report = self.report_formatting_node.format_report_manually(
                report_data, self.state.report_title
            )
        
        # Update state
        self.state.final_report = final_report
        self.state.mark_completed()
        
        logger.info("Final report generation completed")
        return final_report
    
    def _save_report(self, report_content: str):
        """Save report to file"""
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_safe = "".join(c for c in self.state.query if c.isalnum() or c in (' ', '-', '_')).rstrip()
        query_safe = query_safe.replace(' ', '_')[:30]
        
        filename = f"deep_search_report_{query_safe}_{timestamp}.md"
        filepath = os.path.join(self.config.OUTPUT_DIR, filename)
        
        # Save report
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Report saved to: {filepath}")
        
        # Save state (if enabled by configuration)
        if self.config.SAVE_INTERMEDIATE_STATES:
            state_filename = f"state_{query_safe}_{timestamp}.json"
            state_filepath = os.path.join(self.config.OUTPUT_DIR, state_filename)
            self.state.save_to_file(state_filepath)
            logger.info(f"State saved to: {state_filepath}")
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get progress summary"""
        return self.state.get_progress_summary()
    
    def load_state(self, filepath: str):
        """Load state from file"""
        self.state = State.load_from_file(filepath)
        logger.info(f"State loaded from {filepath}")
    
    def save_state(self, filepath: str):
        """Save state to file"""
        self.state.save_to_file(filepath)
        logger.info(f"State saved to {filepath}")

class AnspireSearchAgent(DeepSearchAgent):
    """Deep Search Agent using the Anspire search engine"""
    
    def __init__(self, config: Settings | None = None):
        self.config = config or settings
        
        # Initialize LLM client
        self.llm_client = self._initialize_llm()
        
        # Initialize search toolset
        self.search_agency = AnspireAISearch(api_key=self.config.ANSPIRE_API_KEY)

        # Initialize nodes
        self._initialize_nodes()
        
        # State
        self.state = State()
        
        # Ensure output directory exists
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        
        logger.info(f"Media Agent initialized")
        logger.info(f"Using LLM: {self.llm_client.get_model_info()}")
        logger.info(f"Search toolset: AnspireSearch")

    def execute_search_tool(self, tool_name: str, query: str, **kwargs) -> AnspireResponse:
        # TODO: Execute search using Anspire search tools
        """
        Execute the specified search tool
        
        Args:
            tool_name: Tool name. Available values:
                - "comprehensive_search": comprehensive search (default)
                - "search_last_24_hours": latest information from last 24 hours
                - "search_last_week": information from last week
            query: Search query
            **kwargs: Additional parameters (such as max_results)
            
        Returns:
            AnspireResponse object
        """
        logger.info(f"  → Executing search tool: {tool_name}")
        
        if tool_name == "comprehensive_search":
            max_results = kwargs.get("max_results", 10)
            return self.search_agency.comprehensive_search(query, max_results)
        elif tool_name == "search_last_24_hours":
            return self.search_agency.search_last_24_hours(query)
        elif tool_name == "search_last_week":
            return self.search_agency.search_last_week(query)
        else:
            logger.info(f"  ⚠️  Unknown search tool: {tool_name}, using default comprehensive search")
            return self.search_agency.comprehensive_search(query)


def create_agent(config_file: Optional[str] = None) -> DeepSearchAgent:
    """
    Convenience function to create a Deep Search Agent instance
    
    Args:
        config_file: Configuration file path
        
    Returns:
        DeepSearchAgent instance
    """
    settings = Settings()
    if settings.SEARCH_TOOL_TYPE == "AnspireAPI":
        return AnspireSearchAgent(settings)
    return DeepSearchAgent(settings)
