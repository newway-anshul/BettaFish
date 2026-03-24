"""
Text processing utility functions
Used for cleaning LLM output, parsing JSON, etc.
"""

import re
import json
from typing import Dict, Any, List
from json.decoder import JSONDecodeError


def clean_json_tags(text: str) -> str:
    """
    Remove JSON tags from text
    
    Args:
        text: Raw text
        
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
    Remove Markdown tags from text
    
    Args:
        text: Raw text
        
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
    Remove reasoning process text from output
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
    """
    # Find the JSON start position
    json_start = -1
    
    # Try to find the first { or [
    for i, char in enumerate(text):
        if char in '{[':
            json_start = i
            break
    
    if json_start != -1:
        # Slice from the JSON start position
        return text[json_start:].strip()
    
    # If no JSON marker found, try other methods
    # Remove common reasoning identifiers
    patterns = [
        r'(?:reasoning|推理|思考|分析)[:：]\s*.*?(?=\{|\[)',  # Remove reasoning section
        r'(?:explanation|解释|说明)[:：]\s*.*?(?=\{|\[)',   # Remove explanation section
        r'^.*?(?=\{|\[)',  # Remove all text before JSON
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    return text.strip()


def extract_clean_response(text: str) -> Dict[str, Any]:
    """
    Extract and clean JSON content from a response
    
    Args:
        text: Raw response text
        
    Returns:
        Parsed JSON dictionary
    """
    # 清理文本
    cleaned_text = clean_json_tags(text)
    cleaned_text = remove_reasoning_from_output(cleaned_text)
    
    # 尝试直接解析
    try:
        return json.loads(cleaned_text)
    except JSONDecodeError:
        pass
    
    # 尝试修复不完整的JSON
    fixed_text = fix_incomplete_json(cleaned_text)
    if fixed_text:
        try:
            return json.loads(fixed_text)
        except JSONDecodeError:
            pass
    
    # 尝试查找JSON对象
    json_pattern = r'\{.*\}'
    match = re.search(json_pattern, cleaned_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except JSONDecodeError:
            pass
    
    # 尝试查找JSON数组
    array_pattern = r'\[.*\]'
    match = re.search(array_pattern, cleaned_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except JSONDecodeError:
            pass
    
    # If all methods fail, return an error message
    print(f"Failed to parse JSON response: {cleaned_text[:200]}...")
    return {"error": "JSON parsing failed", "raw_text": cleaned_text}


def fix_incomplete_json(text: str) -> str:
    """
    Repair incomplete JSON responses
    
    Args:
        text: Raw text
        
    Returns:
        Repaired JSON text, or empty string if repair is not possible
    """
    # Remove trailing commas and extra whitespace
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    
    # Check if already valid JSON
    try:
        json.loads(text)
        return text
    except JSONDecodeError:
        pass
    
    # Check if the opening array bracket is missing
    if text.strip().startswith('{') and not text.strip().startswith('['):
        # If it starts with an object, try wrapping in an array
        if text.count('{') > 1:
            # Multiple objects, wrap in array
            text = '[' + text + ']'
        else:
            # Single object, wrap in array
            text = '[' + text + ']'
    
    # Check if the closing array bracket is missing
    if text.strip().endswith('}') and not text.strip().endswith(']'):
        # If it ends with an object, try wrapping in an array
        if text.count('}') > 1:
            # Multiple objects, wrap in array
            text = '[' + text + ']'
        else:
            # Single object, wrap in array
            text = '[' + text + ']'
    
    # Check if brackets are balanced
    open_braces = text.count('{')
    close_braces = text.count('}')
    open_brackets = text.count('[')
    close_brackets = text.count(']')
    
    # Fix unbalanced brackets
    if open_braces > close_braces:
        text += '}' * (open_braces - close_braces)
    if open_brackets > close_brackets:
        text += ']' * (open_brackets - close_brackets)
    
    # Validate the repaired JSON
    try:
        json.loads(text)
        return text
    except JSONDecodeError:
        # If still invalid, try a more aggressive repair
        return fix_aggressive_json(text)


def fix_aggressive_json(text: str) -> str:
    """
    More aggressive JSON repair method
    
    Args:
        text: Raw text
        
    Returns:
        Repaired JSON text
    """
    # Find all possible JSON objects
    objects = re.findall(r'\{[^{}]*\}', text)
    
    if len(objects) >= 2:
        # Multiple objects, wrap in array
        return '[' + ','.join(objects) + ']'
    elif len(objects) == 1:
        # Single object, wrap in array
        return '[' + objects[0] + ']'
    else:
        # No objects found, return empty array
        return '[]'


def update_state_with_search_results(search_results: List[Dict[str, Any]], 
                                   paragraph_index: int, state: Any) -> Any:
    """
    Update state with search results
    
    Args:
        search_results: List of search results
        paragraph_index: Paragraph index
        state: State object
        
    Returns:
        Updated state object
    """
    if 0 <= paragraph_index < len(state.paragraphs):
        # Get the query from the last search (assumed to be the current query)
        current_query = ""
        if search_results:
            # Infer query from search results (needs improvement to get the actual query)
            current_query = "search query"
        
        # Add search results to state
        state.paragraphs[paragraph_index].research.add_search_results(
            current_query, search_results
        )
    
    return state


def validate_json_schema(data: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    验证JSON数据是否包含必需字段
    
    Args:
        data: 要验证的数据
        required_fields: 必需字段列表
        
    Returns:
        验证是否通过
    """
    return all(field in data for field in required_fields)


def truncate_content(content: str, max_length: int = 20000) -> str:
    """
    截断内容到指定长度
    
    Args:
        content: 原始内容
        max_length: 最大长度
        
    Returns:
        截断后的内容
    """
    if len(content) <= max_length:
        return content
    
    # 尝试在单词边界截断
    truncated = content[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # 如果最后一个空格位置合理
        return truncated[:last_space] + "..."
    else:
        return truncated + "..."


def format_search_results_for_prompt(search_results: List[Dict[str, Any]], 
                                   max_length: int = 20000) -> List[str]:
    """
    格式化搜索结果用于提示词
    
    Args:
        search_results: 搜索结果列表
        max_length: 每个结果的最大长度
        
    Returns:
        格式化后的内容列表
    """
    formatted_results = []
    
    for result in search_results:
        content = result.get('content', '')
        if content:
            truncated_content = truncate_content(content, max_length)
            formatted_results.append(truncated_content)
    
    return formatted_results
