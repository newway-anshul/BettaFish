"""
Summary node implementations.
Responsible for generating and updating paragraph content from search results.
"""

import json
from typing import Dict, Any, List
from json.decoder import JSONDecodeError
from loguru import logger

from .base_node import StateMutationNode
from ..state.state import State
from ..prompts import SYSTEM_PROMPT_FIRST_SUMMARY, SYSTEM_PROMPT_REFLECTION_SUMMARY
from ..utils.text_processing import (
    remove_reasoning_from_output,
    clean_json_tags,
    extract_clean_response,
    fix_incomplete_json,
    format_search_results_for_prompt
)

# Import forum reading utilities.
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from utils.forum_reader import get_latest_host_speech, format_host_speech_for_prompt
    FORUM_READER_AVAILABLE = True
except ImportError:
    FORUM_READER_AVAILABLE = False
    logger.warning("Unable to import forum_reader module; HOST speech reading will be skipped")


class FirstSummaryNode(StateMutationNode):
    """Node that generates the first paragraph summary from search results."""
    
    def __init__(self, llm_client):
        """
        Initialize first summary node.
        
        Args:
            llm_client: LLM client
        """
        super().__init__(llm_client, "FirstSummaryNode")
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        if isinstance(input_data, str):
            try:
                data = json.loads(input_data)
                required_fields = ["title", "content", "search_query", "search_results"]
                return all(field in data for field in required_fields)
            except JSONDecodeError:
                return False
        elif isinstance(input_data, dict):
            required_fields = ["title", "content", "search_query", "search_results"]
            return all(field in input_data for field in required_fields)
        return False
    
    def run(self, input_data: Any, **kwargs) -> str:
        """
        Call the LLM to generate paragraph summary.
        
        Args:
            input_data: Data containing title, content, search_query, and search_results
            **kwargs: Extra parameters
            
        Returns:
            Paragraph summary content
        """
        try:
            if not self.validate_input(input_data):
                raise ValueError("Invalid input format")
            
            # Prepare input data.
            if isinstance(input_data, str):
                data = json.loads(input_data)
            else:
                data = input_data.copy() if isinstance(input_data, dict) else input_data
            
            # Read latest HOST speech (if available).
            if FORUM_READER_AVAILABLE:
                try:
                    host_speech = get_latest_host_speech()
                    if host_speech:
                        # Add HOST speech into input data.
                        data['host_speech'] = host_speech
                        logger.info(f"Read HOST speech, length: {len(host_speech)} characters")
                except Exception as e:
                    logger.exception(f"Failed to read HOST speech: {str(e)}")
            
            # Convert to JSON string.
            message = json.dumps(data, ensure_ascii=False)
            
            # If HOST speech exists, prepend as context.
            if FORUM_READER_AVAILABLE and 'host_speech' in data and data['host_speech']:
                formatted_host = format_host_speech_for_prompt(data['host_speech'])
                message = formatted_host + "\n" + message
            
            logger.info("Generating first paragraph summary")
            
            # Call LLM (streaming with safe UTF-8 concatenation).
            response = self.llm_client.stream_invoke_to_string(SYSTEM_PROMPT_FIRST_SUMMARY, message)
            
            # Process response.
            processed_response = self.process_output(response)
            
            logger.info("First paragraph summary generated successfully")
            return processed_response
            
        except Exception as e:
            logger.exception(f"Failed to generate first summary: {str(e)}")
            raise e
    
    def process_output(self, output: str) -> str:
        """
        Process LLM output and extract paragraph content.
        
        Args:
            output: Raw LLM output
            
        Returns:
            Paragraph content
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
                # Try to repair JSON.
                fixed_json = fix_incomplete_json(cleaned_output)
                if fixed_json:
                    try:
                        result = json.loads(fixed_json)
                        logger.info("JSON repaired successfully")
                    except JSONDecodeError:
                        logger.exception("JSON repair failed, using cleaned text directly")
                        # If not JSON, return cleaned text directly.
                        return cleaned_output
                else:
                    logger.exception("Unable to repair JSON, using cleaned text directly")
                    # If not JSON, return cleaned text directly.
                    return cleaned_output
            
            # Extract paragraph content.
            if isinstance(result, dict):
                paragraph_content = result.get("paragraph_latest_state", "")
                if paragraph_content:
                    return paragraph_content
            
            # Fallback to cleaned text.
            return cleaned_output
            
        except Exception as e:
            logger.exception(f"Output processing failed: {str(e)}")
            return "Paragraph summary generation failed"
    
    def mutate_state(self, input_data: Any, state: State, paragraph_index: int, **kwargs) -> State:
        """
        Update the paragraph's latest summary in state.
        
        Args:
            input_data: Input data
            state: Current state
            paragraph_index: Paragraph index
            **kwargs: Extra parameters
            
        Returns:
            Updated state
        """
        try:
            # Generate summary.
            summary = self.run(input_data, **kwargs)
            
            # Update state.
            if 0 <= paragraph_index < len(state.paragraphs):
                state.paragraphs[paragraph_index].research.latest_summary = summary
                logger.info(f"Updated first summary for paragraph {paragraph_index}")
            else:
                raise ValueError(f"Paragraph index {paragraph_index} is out of range")
            
            state.update_timestamp()
            return state
            
        except Exception as e:
            logger.exception(f"State update failed: {str(e)}")
            raise e


class ReflectionSummaryNode(StateMutationNode):
    """Node that updates paragraph summary using reflection search results."""
    
    def __init__(self, llm_client):
        """
        Initialize reflection summary node.
        
        Args:
            llm_client: LLM client
        """
        super().__init__(llm_client, "ReflectionSummaryNode")
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        if isinstance(input_data, str):
            try:
                data = json.loads(input_data)
                required_fields = ["title", "content", "search_query", "search_results", "paragraph_latest_state"]
                return all(field in data for field in required_fields)
            except JSONDecodeError:
                return False
        elif isinstance(input_data, dict):
            required_fields = ["title", "content", "search_query", "search_results", "paragraph_latest_state"]
            return all(field in input_data for field in required_fields)
        return False
    
    def run(self, input_data: Any, **kwargs) -> str:
        """
        Call the LLM to update paragraph content.
        
        Args:
            input_data: Data containing full reflection information
            **kwargs: Extra parameters
            
        Returns:
            Updated paragraph content
        """
        try:
            if not self.validate_input(input_data):
                raise ValueError("Invalid input format")
            
            # Prepare input data.
            if isinstance(input_data, str):
                data = json.loads(input_data)
            else:
                data = input_data.copy() if isinstance(input_data, dict) else input_data
            
            # Read latest HOST speech (if available).
            if FORUM_READER_AVAILABLE:
                try:
                    host_speech = get_latest_host_speech()
                    if host_speech:
                        # Add HOST speech into input data.
                        data['host_speech'] = host_speech
                        logger.info(f"Read HOST speech, length: {len(host_speech)} characters")
                except Exception as e:
                    logger.exception(f"Failed to read HOST speech: {str(e)}")
            
            # Convert to JSON string.
            message = json.dumps(data, ensure_ascii=False)
            
            # If HOST speech exists, prepend as context.
            if FORUM_READER_AVAILABLE and 'host_speech' in data and data['host_speech']:
                formatted_host = format_host_speech_for_prompt(data['host_speech'])
                message = formatted_host + "\n" + message
            
            logger.info("Generating reflection summary")
            
            # Call LLM (streaming with safe UTF-8 concatenation).
            response = self.llm_client.stream_invoke_to_string(SYSTEM_PROMPT_REFLECTION_SUMMARY, message)
            
            # Process response.
            processed_response = self.process_output(response)
            
            logger.info("Reflection summary generated successfully")
            return processed_response
            
        except Exception as e:
            logger.exception(f"Failed to generate reflection summary: {str(e)}")
            raise e
    
    def process_output(self, output: str) -> str:
        """
        Process LLM output and extract updated paragraph content.
        
        Args:
            output: Raw LLM output
            
        Returns:
            Updated paragraph content
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
                # Try to repair JSON.
                fixed_json = fix_incomplete_json(cleaned_output)
                if fixed_json:
                    try:
                        result = json.loads(fixed_json)
                        logger.info("JSON repaired successfully")
                    except JSONDecodeError:
                        logger.error("JSON repair failed, using cleaned text directly")
                        # If not JSON, return cleaned text directly.
                        return cleaned_output
                else:
                    logger.error("Unable to repair JSON, using cleaned text directly")
                    # If not JSON, return cleaned text directly.
                    return cleaned_output
            
            # Extract updated paragraph content.
            if isinstance(result, dict):
                updated_content = result.get("updated_paragraph_latest_state", "")
                if updated_content:
                    return updated_content
            
            # Fallback to cleaned text.
            return cleaned_output
            
        except Exception as e:
            logger.exception(f"Output processing failed: {str(e)}")
            return "Reflection summary generation failed"
    
    def mutate_state(self, input_data: Any, state: State, paragraph_index: int, **kwargs) -> State:
        """
        Write updated summary into state.
        
        Args:
            input_data: Input data
            state: Current state
            paragraph_index: Paragraph index
            **kwargs: Extra parameters
            
        Returns:
            Updated state
        """
        try:
            # Generate updated summary.
            updated_summary = self.run(input_data, **kwargs)
            
            # Update state.
            if 0 <= paragraph_index < len(state.paragraphs):
                state.paragraphs[paragraph_index].research.latest_summary = updated_summary
                state.paragraphs[paragraph_index].research.increment_reflection()
                logger.info(f"Updated reflection summary for paragraph {paragraph_index}")
            else:
                raise ValueError(f"Paragraph index {paragraph_index} is out of range")
            
            state.update_timestamp()
            return state
            
        except Exception as e:
            logger.exception(f"State update failed: {str(e)}")
            raise e
