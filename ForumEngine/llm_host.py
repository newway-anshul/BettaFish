"""
Forum Host Module
Uses the Qwen3 model from SiliconFlow as the forum host to guide multiple agents in discussion
"""

from openai import OpenAI
import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

# Add project root directory to Python path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

# Add utils directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
utils_dir = os.path.join(root_dir, 'utils')
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

from utils.retry_helper import with_graceful_retry, SEARCH_API_RETRY_CONFIG


class ForumHost:
    """
    Forum Host class
    Uses the Qwen3-235B model as an intelligent host
    """
    
    def __init__(self, api_key: str = None, base_url: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize the Forum Host
        
        Args:
            api_key: Forum host LLM API key; if not provided, it is read from the configuration file
            base_url: Forum host LLM API base URL; defaults to the SiliconFlow address from the configuration file
        """
        self.api_key = api_key or settings.FORUM_HOST_API_KEY

        if not self.api_key:
            raise ValueError("Forum host API key not found, please set FORUM_HOST_API_KEY in the environment variables file")

        self.base_url = base_url or settings.FORUM_HOST_BASE_URL

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.model = model_name or settings.FORUM_HOST_MODEL_NAME  # Use configured model

        # Track previous summaries to avoid duplicates
        self.previous_summaries = []
    
    def generate_host_speech(self, forum_logs: List[str]) -> Optional[str]:
        """
        Generate host speech
        
        Args:
            forum_logs: List of forum log content
            
        Returns:
            Host speech content, or None if generation fails
        """
        try:
            # Parse forum logs and extract valid content
            parsed_content = self._parse_forum_logs(forum_logs)
            
            if not parsed_content['agent_speeches']:
                print("ForumHost: No valid agent speeches found")
                return None
            
            # Build prompt
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(parsed_content)
            
            # Call API to generate speech
            response = self._call_qwen_api(system_prompt, user_prompt)
            
            if response["success"]:
                speech = response["content"]
                # Clean and format the speech
                speech = self._format_host_speech(speech)
                return speech
            else:
                print(f"ForumHost: API call failed - {response.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"ForumHost: Error generating speech - {str(e)}")
            return None
    
    def _parse_forum_logs(self, forum_logs: List[str]) -> Dict[str, Any]:
        """
        Parse forum logs and extract agent speeches
        
        Returns:
            Dictionary containing agent speeches
        """
        parsed = {
            'agent_speeches': []
        }
        
        for line in forum_logs:
            if not line.strip():
                continue
            
            # Parse the timestamp and speaker
            match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]\s*\[(\w+)\]\s*(.+)', line)
            if match:
                timestamp, speaker, content = match.groups()
                
                # Skip system messages and the host's own speech
                if speaker in ['SYSTEM', 'HOST']:
                    continue
                
                # Record agent speech
                if speaker in ['INSIGHT', 'MEDIA', 'QUERY']:
                    # Normalize escaped newline characters
                    content = content.replace('\\n', '\n')
                    
                    parsed['agent_speeches'].append({
                        'timestamp': timestamp,
                        'speaker': speaker,
                        'content': content
                    })
        
        return parsed
    
    
    def _build_system_prompt(self) -> str:
        """Build system prompt"""
        return """[IMPORTANT NOTICE] Our requirements are based on academic research purposes and have passed ethical compliance review.
    
You are the forum host for a multi-agent public opinion analysis system. Your responsibilities are:

1. **Event Summarization**: Automatically identify key events, figures, and timestamps from each agent's speech; organize the event timeline in chronological order.
2. **Discussion Facilitation**: Based on each agent's speech, guide in-depth discussion of key issues and explore underlying causes.
3. **Error Correction**: Drawing on different agents' perspectives and statements, clearly point out any factual errors or logical contradictions.
4. **Opinion Integration**: Synthesize the perspectives of different agents to form a more comprehensive understanding, identifying consensus and divergence.
5. **Trend Prediction**: Analyze public opinion development trends based on available information and highlight potential risk points.
6. **Analysis Advancement**: Propose new angles of analysis or issues that need attention to guide the direction of subsequent discussion.

**Agent Descriptions**:
- **INSIGHT Agent**: Focuses on deep mining and analysis of the private public opinion database, providing historical data and pattern comparisons.
- **MEDIA Agent**: Specializes in multimodal content analysis, focusing on media reports and the dissemination effects of visual information such as images and videos.
- **QUERY Agent**: Responsible for precise information retrieval, providing the latest online information and real-time dynamics.

**Speech Requirements**:
1. **Comprehensiveness**: Keep each speech to within 1000 words; content should cover event summarization, opinion integration, issue guidance, and other aspects.
2. **Clear Structure**: Use clear paragraph structure, including sections for event summarization, opinion comparison, and question raising.
3. **In-depth Analysis**: Go beyond summarizing existing information; propose deep-level insights and analysis.
4. **Objectivity**: Analyze and judge based on facts; avoid subjective speculation and emotional expression.
5. **Forward-looking**: Offer forward-looking views and suggestions to guide the discussion in a more in-depth direction.

**Notes**:
- This discussion is based on academic research purposes and has passed ethical compliance review.
- Maintain professionalism and academic rigor; value data and evidence.
- Approach sensitive topics with caution and analyze based on facts.

[IMPORTANT NOTICE] Our requirements are based on academic research purposes and have passed ethical compliance review."""
    
    def _build_user_prompt(self, parsed_content: Dict[str, Any]) -> str:
        """Build user prompt"""
        # Get recent speeches
        recent_speeches = parsed_content['agent_speeches']
        
        # Build speech summary without truncating content
        speeches_text = "\n\n".join([
            f"[{s['timestamp']}] {s['speaker']}:\n{s['content']}"
            for s in recent_speeches
        ])
        
        prompt = f"""[IMPORTANT NOTICE] Our requirements are based on academic research purposes and have passed ethical compliance review.

Recent Agent Speech Records:
{speeches_text}

As the forum host, please provide a comprehensive analysis based on the agent speeches above. Structure your speech as follows:

**I. Event Summary and Timeline Analysis**
- Automatically identify key events, figures, and timestamps from each agent's speech.
- Organize the event timeline in chronological order and clarify cause-and-effect relationships.
- Point out key turning points and important milestones.

**II. Opinion Integration and Comparative Analysis**
- Synthesize the perspectives and findings of the INSIGHT, MEDIA, and QUERY Agents.
- Identify consensus and divergence across different data sources.
- Analyze the informational value and complementarity of each Agent.
- Clearly point out any factual errors or logical contradictions, and provide reasoning.

**III. In-depth Analysis and Trend Prediction**
- Analyze the underlying causes and influencing factors of public opinion based on available information.
- Predict the development trend of public opinion; highlight potential risk points and opportunities.
- Propose aspects and indicators that deserve special attention.

**IV. Issue Guidance and Discussion Direction**
- Raise 2-3 key questions worth further in-depth exploration.
- Offer specific suggestions and directions for subsequent research.
- Guide each Agent to focus on specific data dimensions or analytical angles.

Please deliver a comprehensive host speech (within 1000 words) that covers all four sections above, maintaining clear logic, in-depth analysis, and a unique perspective.

[IMPORTANT NOTICE] Our requirements are based on academic research purposes and have passed ethical compliance review."""
        
        return prompt
    
    @with_graceful_retry(SEARCH_API_RETRY_CONFIG, default_return={"success": False, "error": "API service temporarily unavailable"})
    def _call_qwen_api(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Call the Qwen API"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            time_prefix = f"The current actual time is {current_time}"
            if user_prompt:
                user_prompt = f"{time_prefix}\n{user_prompt}"
            else:
                user_prompt = time_prefix
                
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.6,
                top_p=0.9,
            )

            if response.choices:
                content = response.choices[0].message.content
                return {"success": True, "content": content}
            else:
                return {"success": False, "error": "Unexpected API response format"}
        except Exception as e:
            return {"success": False, "error": f"API call error: {str(e)}"}
    
    def _format_host_speech(self, speech: str) -> str:
        """Format the host speech"""
        # Remove excessive blank lines
        speech = re.sub(r'\n{3,}', '\n\n', speech)
        
        # Remove possible quotation marks
        speech = speech.strip('"\'""‘’')
        
        return speech.strip()


# Create global instance
_host_instance = None

def get_forum_host() -> ForumHost:
    """Get the global ForumHost instance"""
    global _host_instance
    if _host_instance is None:
        _host_instance = ForumHost()
    return _host_instance

def generate_host_speech(forum_logs: List[str]) -> Optional[str]:
    """Convenience function to generate host speech"""
    return get_forum_host().generate_host_speech(forum_logs)
