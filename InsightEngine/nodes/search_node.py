"""
Search node implementations.
Responsible for generating initial and reflection search queries.
"""

import json
from typing import Dict, Any
from json.decoder import JSONDecodeError
from loguru import logger

from .base_node import BaseNode
from ..prompts import SYSTEM_PROMPT_FIRST_SEARCH, SYSTEM_PROMPT_REFLECTION
from ..utils.text_processing import (
    remove_reasoning_from_output,
    clean_json_tags,
    extract_clean_response,
    fix_incomplete_json
)


class FirstSearchNode(BaseNode):
    """Node for generating the first search query for a paragraph."""
    
    def __init__(self, llm_client):
        """
        Initialize first search node.
        
        Args:
            llm_client: LLM client
        """
        super().__init__(llm_client, "FirstSearchNode")
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        if isinstance(input_data, str):
            try:
                data = json.loads(input_data)
                return "title" in data and "content" in data
            except JSONDecodeError:
                return False
        elif isinstance(input_data, dict):
            return "title" in input_data and "content" in input_data
        return False
    
    def run(self, input_data: Any, **kwargs) -> Dict[str, str]:
        """
        Call the LLM to generate a search query and reasoning.
        
        Args:
            input_data: String or dict containing title and content
            **kwargs: Extra parameters
            
        Returns:
            Dict containing search_query and reasoning
        """
        try:
            if not self.validate_input(input_data):
                raise ValueError("Invalid input format. title and content are required.")
            
            # Prepare input data.
            if isinstance(input_data, str):
                message = input_data
            else:
                message = json.dumps(input_data, ensure_ascii=False)
            
            logger.info("Generating first search query")
            
            # Call LLM (streaming with safe UTF-8 concatenation).
            response = self.llm_client.stream_invoke_to_string(SYSTEM_PROMPT_FIRST_SEARCH, message)
            
            # Process response.
            processed_response = self.process_output(response)
            
            logger.info(f"Generated search query: {processed_response.get('search_query', 'N/A')}")
            return processed_response
            
        except Exception as e:
            logger.exception(f"Failed to generate first search query: {str(e)}")
            raise e
    
    def process_output(self, output: str) -> Dict[str, str]:
        """
        Process LLM output and extract search query and reasoning.
        
        Args:
            output: Raw LLM output
            
        Returns:
            Dict containing search_query and reasoning
        """
        try:
            # Clean response text.
            cleaned_output = remove_reasoning_from_output(output)
            cleaned_output = clean_json_tags(cleaned_output)
            
            # Log cleaned output for debugging.
            logger.info(f"Cleaned output: {cleaned_output}")
            
            # Parse JSON.
            try:
                result = json.loads(cleaned_output)
                logger.info("JSON parsed successfully")
            except JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {str(e)}")
                # Use stronger extraction.
                result = extract_clean_response(cleaned_output)
                if "error" in result:
                    logger.error("JSON parsing failed, trying to repair...")
                    # Try to repair JSON.
                    fixed_json = fix_incomplete_json(cleaned_output)
                    if fixed_json:
                        try:
                            result = json.loads(fixed_json)
                            logger.info("JSON repaired successfully")
                        except JSONDecodeError:
                            logger.error("JSON repair failed")
                            # Return default query.
                            return self._get_default_search_query()
                    else:
                        logger.error("Unable to repair JSON, using default query")
                        return self._get_default_search_query()
            
            # Validate and clean result.
            search_query = result.get("search_query", "")
            reasoning = result.get("reasoning", "")
            
            if not search_query:
                logger.warning("No search query found, using default query")
                return self._get_default_search_query()
            
            return {
                "search_query": search_query,
                "reasoning": reasoning
            }
            
        except Exception as e:
            self.log_error(f"Output processing failed: {str(e)}")
            # Return default query.
            return self._get_default_search_query()
    
    def _get_default_search_query(self) -> Dict[str, str]:
        """
        Get default search query.
        
        Returns:
            Default search query dictionary
        """
        return {
            "search_query": "related topic research",
            "reasoning": "Using default search query because parsing failed"
        }


class ReflectionNode(BaseNode):
    """Node that reflects on a paragraph and generates a new search query."""
    
    def __init__(self, llm_client):
        """
        Initialize reflection node.
        
        Args:
            llm_client: LLM client
        """
        super().__init__(llm_client, "ReflectionNode")
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        if isinstance(input_data, str):
            try:
                data = json.loads(input_data)
                required_fields = ["title", "content", "paragraph_latest_state"]
                return all(field in data for field in required_fields)
            except JSONDecodeError:
                return False
        elif isinstance(input_data, dict):
            required_fields = ["title", "content", "paragraph_latest_state"]
            return all(field in input_data for field in required_fields)
        return False
    
    def run(self, input_data: Any, **kwargs) -> Dict[str, str]:
        """
        Call the LLM to reflect and generate a search query.
        
        Args:
            input_data: String or dict containing title, content, and paragraph_latest_state
            **kwargs: Extra parameters
            
        Returns:
            Dict containing search_query and reasoning
        """
        try:
            if not self.validate_input(input_data):
                raise ValueError("Invalid input format. title, content, and paragraph_latest_state are required.")
            
            # Prepare input data.
            if isinstance(input_data, str):
                message = input_data
            else:
                message = json.dumps(input_data, ensure_ascii=False)
            
            logger.info("Reflecting and generating a new search query")
            
            # Call LLM (streaming with safe UTF-8 concatenation).
            response = self.llm_client.stream_invoke_to_string(SYSTEM_PROMPT_REFLECTION, message)
            
            # Process response.
            processed_response = self.process_output(response)
            
            logger.info(f"Reflection generated search query: {processed_response.get('search_query', 'N/A')}")
            return processed_response
            
        except Exception as e:
            logger.exception(f"Failed to generate reflection search query: {str(e)}")
            raise e
    
    def process_output(self, output: str) -> Dict[str, str]:
        """
        Process LLM output and extract search query and reasoning.
        
        Args:
            output: Raw LLM output
            
        Returns:
            Dict containing search_query and reasoning
        """
        try:
            # Clean response text.
            cleaned_output = remove_reasoning_from_output(output)
            cleaned_output = clean_json_tags(cleaned_output)
            
            # Log cleaned output for debugging.
            logger.info(f"Cleaned output: {cleaned_output}")
            
            # Parse JSON.
            try:
                result = json.loads(cleaned_output)
                logger.info("JSON parsed successfully")
            except JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {str(e)}")
                # Use stronger extraction.
                result = extract_clean_response(cleaned_output)
                if "error" in result:
                    logger.error("JSON parsing failed, trying to repair...")
                    # Try to repair JSON.
                    fixed_json = fix_incomplete_json(cleaned_output)
                    if fixed_json:
                        try:
                            result = json.loads(fixed_json)
                            logger.info("JSON repaired successfully")
                        except JSONDecodeError:
                            logger.error("JSON repair failed")
                            # Return default query.
                            return self._get_default_reflection_query()
                    else:
                        logger.error("Unable to repair JSON, using default query")
                        return self._get_default_reflection_query()
            
            # Validate and clean result.
            search_query = result.get("search_query", "")
            reasoning = result.get("reasoning", "")
            
            if not search_query:
                logger.warning("No search query found, using default query")
                return self._get_default_reflection_query()
            
            return {
                "search_query": search_query,
                "reasoning": reasoning
            }
            
        except Exception as e:
            logger.exception(f"Output processing failed: {str(e)}")
            # Return default query.
            return self._get_default_reflection_query()
    
    def _get_default_reflection_query(self) -> Dict[str, str]:
        """
        Get default reflection search query.
        
        Returns:
            Default reflection search query dictionary
        """
        return {
            "search_query": "deep research supplementary information",
            "reasoning": "Using default reflection search query because parsing failed"
        }
