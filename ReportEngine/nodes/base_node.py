"""
Base classes for Report Engine nodes.

All higher-level reasoning nodes inherit from this module so they can share
logging, input validation, and state-mutation interfaces.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from ..llms.base import LLMClient
from ..state.state import ReportState
from loguru import logger

class BaseNode(ABC):
    """
    Base class for all nodes.

    Provides shared logging helpers, input/output hooks, and LLM client
    injection so each node can focus on its domain logic.
    """
    
    def __init__(self, llm_client: LLMClient, node_name: str = ""):
        """
        Initialize the node.
        
        Args:
            llm_client: LLM client instance.
            node_name: Node name.

        BaseNode stores the node name so log messages share a consistent prefix.
        """
        self.llm_client = llm_client
        self.node_name = node_name or self.__class__.__name__
    
    @abstractmethod
    def run(self, input_data: Any, **kwargs) -> Any:
        """
        Execute the node logic.
        
        Args:
            input_data: Input payload.
            **kwargs: Additional arguments.
            
        Returns:
            Processing result.
        """
        pass
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data.
        The default implementation accepts all input. Subclasses can override it
        to enforce field-level validation.
        
        Args:
            input_data: Input payload.
            
        Returns:
            Whether validation passed.
        """
        return True
    
    def process_output(self, output: Any) -> Any:
        """
        Process output data.
        Subclasses can override this to transform or validate the output.
        
        Args:
            output: Raw output.
            
        Returns:
            Processed output.
        """
        return output
    
    def log_info(self, message: str):
        """Write an info log entry with the node name prefix."""
        formatted_message = f"[{self.node_name}] {message}"
        logger.info(formatted_message)
    
    def log_error(self, message: str):
        """Write an error log entry for troubleshooting."""
        formatted_message = f"[{self.node_name}] {message}"
        logger.error(formatted_message)


class StateMutationNode(BaseNode):
    """
    Base class for nodes that mutate state.

    Intended for nodes that need to update ReportState directly.
    """
    
    @abstractmethod
    def mutate_state(self, input_data: Any, state: ReportState, **kwargs) -> ReportState:
        """
        Mutate the state.

        Subclasses should return the new state object, or mutate the existing
        state in place and return it so the pipeline can record the update.
        
        Args:
            input_data: Input payload.
            state: Current state.
            **kwargs: Additional arguments.
            
        Returns:
            Updated state.
        """
        pass
