"""
Text processing utility functions
Used to clean LLM output, parse JSON, and related tasks
"""

import re
import json
from typing import Dict, Any, List
from json.decoder import JSONDecodeError


def clean_json_tags(text: str) -> str:
    """
    Clean JSON code block tags from text
    
    Args:
        text: Original text
        
    Returns:
        Cleaned text
    """
    # Remove ```json and ``` tags
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    text = re.sub(r'```', '', text)
    
    return text.strip()


def clean_markdown_tags(text: str) -> str:
    """
    Clean Markdown code block tags from text
    
    Args:
        text: Original text
        
    Returns:
        Cleaned text
    """
    # Remove ```markdown and ``` tags
    text = re.sub(r'```markdown\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    text = re.sub(r'```', '', text)
    
    return text.strip()


def remove_reasoning_from_output(text: str) -> str:
    """
    Remove reasoning text from output
    
    Args:
        text: Original text
        
    Returns:
        Cleaned text
    """
    # Find JSON start position
    json_start = -1
    
    # Try to find the first { or [
    for i, char in enumerate(text):
        if char in '{[':
            json_start = i
            break
    
    if json_start != -1:
        # Slice from JSON start position
        return text[json_start:].strip()
    
    # If no JSON marker is found, try other methods
    # Remove common reasoning markers
    patterns = [
        r'(?:reasoning|thinking|analysis)[:：]\s*.*?(?=\{|\[)',  # Remove reasoning section
        r'(?:explanation|description|note)[:：]\s*.*?(?=\{|\[)',   # Remove explanation section
        r'^.*?(?=\{|\[)',  # Remove all text before JSON
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    return text.strip()


def extract_clean_response(text: str) -> Dict[str, Any]:
    """
    Extract and clean JSON content from a response
    
    Args:
        text: Original response text
        
    Returns:
        Parsed JSON dictionary
    """
    # Clean text
    cleaned_text = clean_json_tags(text)
    cleaned_text = remove_reasoning_from_output(cleaned_text)
    
    # Try direct parsing
    try:
        return json.loads(cleaned_text)
    except JSONDecodeError:
        pass
    
    # Try fixing incomplete JSON
    fixed_text = fix_incomplete_json(cleaned_text)
    if fixed_text:
        try:
            return json.loads(fixed_text)
        except JSONDecodeError:
            pass
    
    # Try to find a JSON object
    json_pattern = r'\{.*\}'
    match = re.search(json_pattern, cleaned_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except JSONDecodeError:
            pass
    
    # Try to find a JSON array
    array_pattern = r'\[.*\]'
    match = re.search(array_pattern, cleaned_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except JSONDecodeError:
            pass
    
    # If all methods fail, return error information
    print(f"Failed to parse JSON response: {cleaned_text[:200]}...")
    return {"error": "JSON parse failed", "raw_text": cleaned_text}


def fix_incomplete_json(text: str) -> str:
    """
    Fix incomplete JSON responses
    
    Args:
        text: Original text
        
    Returns:
        Repaired JSON text, or an empty string if repair fails
    """
    # Remove extra commas and whitespace
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    
    # Check whether it's already valid JSON
    try:
        json.loads(text)
        return text
    except JSONDecodeError:
        pass
    
    # Check whether opening array bracket is missing
    if text.strip().startswith('{') and not text.strip().startswith('['):
        # If it starts with an object, try wrapping as an array
        if text.count('{') > 1:
            # Multiple objects: wrap as array
            text = '[' + text + ']'
        else:
            # Single object: wrap as array
            text = '[' + text + ']'
    
    # Check whether closing array bracket is missing
    if text.strip().endswith('}') and not text.strip().endswith(']'):
        # If it ends with an object, try wrapping as an array
        if text.count('}') > 1:
            # Multiple objects: wrap as array
            text = '[' + text + ']'
        else:
            # Single object: wrap as array
            text = '[' + text + ']'
    
    # Check whether brackets are balanced
    open_braces = text.count('{')
    close_braces = text.count('}')
    open_brackets = text.count('[')
    close_brackets = text.count(']')
    
    # Fix unbalanced brackets
    if open_braces > close_braces:
        text += '}' * (open_braces - close_braces)
    if open_brackets > close_brackets:
        text += ']' * (open_brackets - close_brackets)
    
    # Validate repaired JSON
    try:
        json.loads(text)
        return text
    except JSONDecodeError:
        # If still invalid, try a more aggressive repair
        return fix_aggressive_json(text)


def fix_aggressive_json(text: str) -> str:
    """
    More aggressive JSON repair strategy
    
    Args:
        text: Original text
        
    Returns:
        Repaired JSON text
    """
    # Find all possible JSON objects
    objects = re.findall(r'\{[^{}]*\}', text)
    
    if len(objects) >= 2:
        # If there are multiple objects, wrap them into an array
        return '[' + ','.join(objects) + ']'
    elif len(objects) == 1:
        # If there is only one object, wrap it into an array
        return '[' + objects[0] + ']'
    else:
        # If no objects are found, return an empty array
        return '[]'


def update_state_with_search_results(search_results: List[Dict[str, Any]], 
                                   paragraph_index: int, state: Any) -> Any:
    """
    Update state with search results
    
    Args:
        search_results: Search result list
        paragraph_index: Paragraph index
        state: State object
        
    Returns:
        Updated state object
    """
    if 0 <= paragraph_index < len(state.paragraphs):
        # Get the query from the last search (assumed current query)
        current_query = ""
        if search_results:
            # Infer query from search results (needs improvement to get actual query)
            current_query = "search query"
        
        # Add search results to state
        state.paragraphs[paragraph_index].research.add_search_results(
            current_query, search_results
        )
    
    return state


def validate_json_schema(data: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    Validate whether JSON data contains required fields
    
    Args:
        data: Data to validate
        required_fields: List of required fields
        
    Returns:
        Whether validation passes
    """
    return all(field in data for field in required_fields)


def truncate_content(content: str, max_length: int = 20000) -> str:
    """
    Truncate content to a specified length
    
    Args:
        content: Original content
        max_length: Maximum length
        
    Returns:
        Truncated content
    """
    if len(content) <= max_length:
        return content
    
    # Try truncating at a word boundary
    truncated = content[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If the last space position is reasonable
        return truncated[:last_space] + "..."
    else:
        return truncated + "..."


def format_search_results_for_prompt(search_results: List[Dict[str, Any]], 
                                   max_length: int = 20000) -> List[str]:
    """
    Format search results for prompt input
    
    Args:
        search_results: Search result list
        max_length: Maximum length per result
        
    Returns:
        Formatted content list
    """
    formatted_results = []
    
    for result in search_results:
        content = result.get('content', '')
        if content:
            truncated_content = truncate_content(content, max_length)
            formatted_results.append(truncated_content)
    
    return formatted_results
