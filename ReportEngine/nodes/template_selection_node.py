"""
Template selection node.

Combines the user query, three-engine reports, forum logs, and the local
template library to let the LLM choose the most suitable report skeleton.
"""

import os
import json
from typing import Dict, Any, List, Optional
from loguru import logger

from .base_node import BaseNode
from ..prompts import SYSTEM_PROMPT_TEMPLATE_SELECTION
from ..utils.json_parser import RobustJSONParser, JSONParseError


TEMPLATE_KEYWORDS = {
    "brand": ["\u4f01\u4e1a\u54c1\u724c"],
    "competition": ["\u5e02\u573a\u7ade\u4e89"],
    "routine": ["\u65e5\u5e38", "\u5b9a\u671f"],
    "policy": ["\u653f\u7b56", "\u884c\u4e1a"],
    "hotspot": ["\u70ed\u70b9", "\u793e\u4f1a"],
    "crisis": ["\u7a81\u53d1", "\u5371\u673a"],
}
CN_TEMPLATE_SUFFIX = "\u6a21\u677f"


class TemplateSelectionNode(BaseNode):
    """
    Template selection processing node.

    Prepares candidate templates, builds prompts, parses the LLM result, and
    falls back to a built-in default when selection fails.
    """
    
    def __init__(self, llm_client, template_dir: str = "ReportEngine/report_template"):
        """
        Initialize the template selection node.

        Args:
            llm_client: LLM client instance.
            template_dir: Template directory path.
        """
        super().__init__(llm_client, "TemplateSelectionNode")
        self.template_dir = template_dir
        # Initialize the robust JSON parser with all repair strategies enabled.
        self.json_parser = RobustJSONParser(
            enable_json_repair=True,
            enable_llm_repair=False,
            max_repair_attempts=3,
        )
        
    def run(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute template selection.
        
        Args:
            input_data: Dictionary containing the query and report content.
                - query: Original query.
                - reports: Report list from the three sub-agents.
                - forum_logs: Forum log content.
                
        Returns:
            Selected template metadata including name, content, and selection
            reason.
        """
        logger.info("Starting template selection...")
        
        query = input_data.get('query', '')
        reports = input_data.get('reports', [])
        forum_logs = input_data.get('forum_logs', '')
        
        # Collect available templates.
        available_templates = self._get_available_templates()
        
        if not available_templates:
            logger.info("No preset templates found; using the built-in default template")
            return self._get_fallback_template()
        
        # Use the LLM to select a template.
        try:
            llm_result = self._llm_template_selection(query, reports, forum_logs, available_templates)
            if llm_result:
                return llm_result
        except Exception as e:
            logger.exception(f"LLM template selection failed: {str(e)}")
        
        # Fall back if LLM-based selection fails.
        return self._get_fallback_template()
    

    
    def _llm_template_selection(self, query: str, reports: List[Any], forum_logs: str, 
                              available_templates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Use the LLM to select the most suitable template.

        Build the template list and report summary, call the LLM, parse the JSON
        response, then verify the selected template exists before returning a
        normalized result.

        Args:
            query: User topic.
            reports: Report content from multiple analysis engines.
            forum_logs: Forum logs, possibly empty.
            available_templates: Local template catalog.

        Returns:
            dict | None: Template information when the LLM returns a valid
            result, otherwise None.
        """
        logger.info("Attempting LLM-based template selection...")
        
        # Build the template list.
        template_list = "\n".join([f"- {t['name']}: {t['description']}" for t in available_templates])
        
        # Build the report content summary.
        reports_summary = ""
        if reports:
            reports_summary = "\n\n=== Analysis Engine Report Content ===\n"
            for i, report in enumerate(reports, 1):
                # Extract report content from supported data shapes.
                if isinstance(report, dict):
                    content = report.get('content', str(report))
                elif hasattr(report, 'content'):
                    content = report.content
                else:
                    content = str(report)
                
                # Truncate long content and keep the first 1000 characters.
                if len(content) > 1000:
                    content = content[:1000] + "...(content truncated)"
                
                reports_summary += f"\nReport {i} Content:\n{content}\n"
        
        # Build the forum log summary.
        forum_summary = ""
        if forum_logs and forum_logs.strip():
            forum_summary = "\n\n=== Discussion Across the Three Engines ===\n"
            # Truncate long log content and keep the first 800 characters.
            if len(forum_logs) > 800:
                forum_content = forum_logs[:800] + "...(discussion truncated)"
            else:
                forum_content = forum_logs
            forum_summary += forum_content
        
        user_message = f"""Query: {query}

Report count: {len(reports)} analysis-engine reports
Forum logs: {'available' if forum_logs else 'not available'}
{reports_summary}{forum_summary}

Available templates:
{template_list}

Please choose the most suitable template based on the query, report content,
and forum logs."""
        
        # Call the LLM.
        response = self.llm_client.stream_invoke_to_string(SYSTEM_PROMPT_TEMPLATE_SELECTION, user_message)

        # Check whether the response is empty.
        if not response or not response.strip():
            logger.error("LLM returned an empty response")
            return None

        logger.info(f"Raw LLM response: {response}")

        # Parse the JSON response with the robust parser.
        try:
            result = self.json_parser.parse(
                response,
                context_name="template selection",
                expected_keys=["template_name", "selection_reason"],
            )

            # Verify that the selected template exists.
            selected_template_name = result.get('template_name', '')
            for template in available_templates:
                if template['name'] == selected_template_name or selected_template_name in template['name']:
                    logger.info(f"LLM selected template: {selected_template_name}")
                    return {
                        'template_name': template['name'],
                        'template_content': template['content'],
                        'selection_reason': result.get('selection_reason', 'Selected by the LLM')
                    }

            logger.error(f"LLM selected a template that does not exist: {selected_template_name}")
            return None

        except JSONParseError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            # Try to extract the template from free-form text.
            return self._extract_template_from_text(response, available_templates)
    

    def _extract_template_from_text(self, response: str, available_templates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Extract template information from a text response.

        When the LLM does not return valid JSON, fall back to matching template
        name variants in the raw text.

        Args:
            response: Unstructured LLM text.
            available_templates: Optional template list.

        Returns:
            dict | None: Template details on success, otherwise None.
        """
        logger.info("Attempting to extract template information from text")
        
        # Search the response for template-name variants.
        for template in available_templates:
            template_name_variants = [
                template['name'],
                template['name'].replace('.md', ''),
                template['name'].replace(CN_TEMPLATE_SUFFIX, ''),
            ]
            
            for variant in template_name_variants:
                if variant in response:
                    logger.info(f"Found template in response: {template['name']}")
                    return {
                        'template_name': template['name'],
                        'template_content': template['content'],
                        'selection_reason': 'Extracted from the text response'
                    }
        
        return None
    
    def _get_available_templates(self) -> List[Dict[str, Any]]:
        """
        Get the list of available templates.

        Enumerate the `.md` files in the template directory and read their
        content and derived descriptions.

        Returns:
            list[dict]: Each item contains name, path, content, and
            description.
        """
        templates = []
        
        if not os.path.exists(self.template_dir):
            logger.error(f"Template directory does not exist: {self.template_dir}")
            return templates
        
        # Find all markdown template files.
        for filename in os.listdir(self.template_dir):
            if filename.endswith('.md'):
                template_path = os.path.join(self.template_dir, filename)
                try:
                    with open(template_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    template_name = filename.replace('.md', '')
                    description = self._extract_template_description(template_name)
                    
                    templates.append({
                        'name': template_name,
                        'path': template_path,
                        'content': content,
                        'description': description
                    })
                except Exception as e:
                    logger.exception(f"Failed to read template file {filename}: {str(e)}")
        
        return templates
    
    def _extract_template_description(self, template_name: str) -> str:
        """Generate a readable description from the template name."""
        if any(keyword in template_name for keyword in TEMPLATE_KEYWORDS["brand"]):
            return "Suitable for corporate brand reputation and image analysis"
        elif any(keyword in template_name for keyword in TEMPLATE_KEYWORDS["competition"]):
            return "Suitable for market competition landscape and competitor analysis"
        elif any(keyword in template_name for keyword in TEMPLATE_KEYWORDS["routine"]):
            return "Suitable for routine monitoring and periodic reporting"
        elif any(keyword in template_name for keyword in TEMPLATE_KEYWORDS["policy"]):
            return "Suitable for policy impact and industry dynamics analysis"
        elif any(keyword in template_name for keyword in TEMPLATE_KEYWORDS["hotspot"]):
            return "Suitable for social hot topics and public event analysis"
        elif any(keyword in template_name for keyword in TEMPLATE_KEYWORDS["crisis"]):
            return "Suitable for breaking incidents and crisis communication"
        
        return "General report template"
    

    
    def _get_fallback_template(self) -> Dict[str, Any]:
        """
        Get the fallback default template.

        The fallback uses an empty template so the LLM can design the structure
        more freely.

        Returns:
            dict: A structure compatible with the normal LLM return shape.
        """
        logger.info("No suitable template found; using an empty template as fallback")
        
        return {
            'template_name': 'Freeform Template',
            'template_content': '',
            'selection_reason': 'No suitable preset template was found, so the LLM will design the report structure from the content'
        }
