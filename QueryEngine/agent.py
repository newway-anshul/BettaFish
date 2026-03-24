"""
Deep Search Agent main class
Integrates all modules to implement a complete deep-search workflow
"""

import json
import os
import re
from datetime import datetime
from typing import Optional, Dict, Any, List

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
from .tools import TavilyNewsAgency, TavilyResponse
from .utils import Settings, format_search_results_for_prompt
from loguru import logger

class DeepSearchAgent:
    """Deep Search Agent main class"""
    
    def __init__(self, config: Optional[Settings] = None):
        """
        Initialize Deep Search Agent
        
        Args:
            config: Configuration object; auto-load if not provided
        """
        # Load configuration
        from .utils.config import settings
        self.config = config or settings
        
        # Initialize LLM client
        self.llm_client = self._initialize_llm()
        
        # Initialize search toolset
        self.search_agency = TavilyNewsAgency(api_key=self.config.TAVILY_API_KEY)
        
        # Initialize nodes
        self._initialize_nodes()
        
        # State
        self.state = State()
        
        # Ensure output directory exists
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        
        logger.info(f"Query Agent initialized")
        logger.info(f"Using LLM: {self.llm_client.get_model_info()}")
        logger.info(f"Search toolkit: TavilyNewsAgency (supports 6 search tools)")
    
    def _initialize_llm(self) -> LLMClient:
        """Initialize the LLM client"""
        return LLMClient(
            api_key=self.config.QUERY_ENGINE_API_KEY,
            model_name=self.config.QUERY_ENGINE_MODEL_NAME,
            base_url=self.config.QUERY_ENGINE_BASE_URL,
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
        Validate whether the date format is YYYY-MM-DD
        
        Args:
            date_str: Date string
            
        Returns:
            Whether the format is valid
        """
        if not date_str:
            return False
        
        # Validate pattern
        pattern = r'^\d{4}-\d{2}-\d{2}$'
        if not re.match(pattern, date_str):
            return False
        
        # Validate whether the date itself is valid
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False
    
    def execute_search_tool(self, tool_name: str, query: str, **kwargs) -> TavilyResponse:
        """
        Execute the specified search tool
        
        Args:
            tool_name: Tool name, options:
                - "basic_search_news": Basic news search (fast, general)
                - "deep_search_news": In-depth news analysis
                - "search_news_last_24_hours": Latest news in the last 24 hours
                - "search_news_last_week": News from the past week
                - "search_images_for_news": News image search
                - "search_news_by_date": Search news by date range
            query: Search query
            **kwargs: Additional parameters (e.g., start_date, end_date, max_results)
            
        Returns:
            TavilyResponse object
        """
        logger.info(f"  -> Executing search tool: {tool_name}")
        
        if tool_name == "basic_search_news":
            max_results = kwargs.get("max_results", 7)
            return self.search_agency.basic_search_news(query, max_results)
        elif tool_name == "deep_search_news":
            return self.search_agency.deep_search_news(query)
        elif tool_name == "search_news_last_24_hours":
            return self.search_agency.search_news_last_24_hours(query)
        elif tool_name == "search_news_last_week":
            return self.search_agency.search_news_last_week(query)
        elif tool_name == "search_images_for_news":
            return self.search_agency.search_images_for_news(query)
        elif tool_name == "search_news_by_date":
            start_date = kwargs.get("start_date")
            end_date = kwargs.get("end_date")
            if not start_date or not end_date:
                raise ValueError("search_news_by_date requires start_date and end_date parameters")
            return self.search_agency.search_news_by_date(query, start_date, end_date)
        else:
            logger.warning(f"  ⚠️  Unknown search tool: {tool_name}, using default basic search")
            return self.search_agency.basic_search_news(query)
    
    def research(self, query: str, save_report: bool = True) -> str:
        """
        Execute deep research
        
        Args:
            query: Research query
            save_report: Whether to save the report to file
            
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
            logger.error(f"An error occurred during research: {str(e)} \nError traceback: {error_traceback}")
            raise e
    
    def _generate_report_structure(self, query: str):
        """Generate report structure"""
        logger.info(f"\n[Step 1] Generating report structure...")
        
        # Create report structure node
        report_structure_node = ReportStructureNode(self.llm_client, query)
        
        # Generate structure and update state
        self.state = report_structure_node.mutate_state(state=self.state)
        
        _message = f"Report structure generated, total paragraphs: {len(self.state.paragraphs)}"
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
        search_tool = search_output.get("search_tool", "basic_search_news")  # Default tool
        reasoning = search_output["reasoning"]
        
        logger.info(f"  - Search query: {search_query}")
        logger.info(f"  - Selected tool: {search_tool}")
        logger.info(f"  - Reasoning: {reasoning}")
        
        # Execute search
        logger.info("  - Executing web search...")
        
        # Handle special parameters for search_news_by_date
        search_kwargs = {}
        if search_tool == "search_news_by_date":
            start_date = search_output.get("start_date")
            end_date = search_output.get("end_date")
            
            if start_date and end_date:
                # Validate date format
                if self._validate_date_format(start_date) and self._validate_date_format(end_date):
                    search_kwargs["start_date"] = start_date
                    search_kwargs["end_date"] = end_date
                    logger.info(f"  - Date range: {start_date} to {end_date}")
                else:
                    logger.info(f"  ⚠️  Invalid date format (expected YYYY-MM-DD), falling back to basic search")
                    logger.info(f"      Provided dates: start_date={start_date}, end_date={end_date}")
                    search_tool = "basic_search_news"
            else:
                logger.info(f"  ⚠️  search_news_by_date is missing date parameters, falling back to basic search")
                search_tool = "basic_search_news"
        
        search_response = self.execute_search_tool(search_tool, search_query, **search_kwargs)
        
        # Convert to a compatible format
        search_results = []
        if search_response and search_response.results:
            # Each search tool has tool-specific result counts; cap at top 10
            max_results = min(len(search_response.results), 10)
            for result in search_response.results[:max_results]:
                search_results.append({
                    'title': result.title,
                    'url': result.url,
                    'content': result.content,
                    'score': result.score,
                    'raw_content': result.raw_content,
                    'published_date': result.published_date  # Added field
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
        paragraph.research.add_search_results(search_query, search_results)
        
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
            search_tool = reflection_output.get("search_tool", "basic_search_news")  # Default tool
            reasoning = reflection_output["reasoning"]
            
            logger.info(f"    Reflection query: {search_query}")
            logger.info(f"    Selected tool: {search_tool}")
            logger.info(f"    Reflection reasoning: {reasoning}")
            
            # Execute reflection search
            # Handle special parameters for search_news_by_date
            search_kwargs = {}
            if search_tool == "search_news_by_date":
                start_date = reflection_output.get("start_date")
                end_date = reflection_output.get("end_date")
                
                if start_date and end_date:
                    # Validate date format
                    if self._validate_date_format(start_date) and self._validate_date_format(end_date):
                        search_kwargs["start_date"] = start_date
                        search_kwargs["end_date"] = end_date
                        logger.info(f"    Date range: {start_date} to {end_date}")
                    else:
                        logger.info(f"    ⚠️  Invalid date format (expected YYYY-MM-DD), falling back to basic search")
                        logger.info(f"        Provided dates: start_date={start_date}, end_date={end_date}")
                        search_tool = "basic_search_news"
                else:
                    logger.info(f"    ⚠️  search_news_by_date is missing date parameters, falling back to basic search")
                    search_tool = "basic_search_news"
            
            search_response = self.execute_search_tool(search_tool, search_query, **search_kwargs)
            
            # Convert to a compatible format
            search_results = []
            if search_response and search_response.results:
                # Each search tool has tool-specific result counts; cap at top 10
                max_results = min(len(search_response.results), 10)
                for result in search_response.results[:max_results]:
                    search_results.append({
                        'title': result.title,
                        'url': result.url,
                        'content': result.content,
                        'score': result.score,
                        'raw_content': result.raw_content,
                        'published_date': result.published_date
                    })
            
            if search_results:
                logger.info(f"    Found {len(search_results)} reflection search results")
                for j, result in enumerate(search_results, 1):
                    date_info = f" (Published: {result.get('published_date', 'N/A')})" if result.get('published_date') else ""
                    logger.info(f"      {j}. {result['title'][:50]}...{date_info}")
            else:
                logger.info("    No reflection search results found")
            
            # Update search history
            paragraph.research.add_search_results(search_query, search_results)
            
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
            logger.error(f"LLM formatting failed, using fallback method: {str(e)}")
            final_report = self.report_formatting_node.format_report_manually(
                report_data, self.state.report_title
            )
        
        # Update state
        self.state.final_report = final_report
        self.state.mark_completed()
        
        logger.info("Final report generated")
        return final_report
    
    def _save_report(self, report_content: str):
        """Save report to file"""
        # Generate file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_safe = "".join(c for c in self.state.query if c.isalnum() or c in (' ', '-', '_')).rstrip()
        query_safe = query_safe.replace(' ', '_')[:30]
        
        filename = f"deep_search_report_{query_safe}_{timestamp}.md"
        filepath = os.path.join(self.config.OUTPUT_DIR, filename)
        
        # Save report
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Report saved to: {filepath}")
        
        # Save state (if enabled in config)
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


def create_agent() -> DeepSearchAgent:
    """
    Convenience function to create a Deep Search Agent instance
    
    Returns:
        DeepSearchAgent instance
    """
    from .utils.config import Settings
    config = Settings()
    return DeepSearchAgent(config)
