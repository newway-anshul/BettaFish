"""
Base node classes.
Defines the base interfaces for all processing nodes.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from loguru import logger
from ..llms.base import LLMClient
from ..state.state import State


class BaseNode(ABC):
    """Base class for all processing nodes."""
    
    def __init__(self, llm_client: LLMClient, node_name: str = ""):
        """
        Initialize the node.
        
        Args:
            llm_client: LLM client
            node_name: Node name
        """
        self.llm_client = llm_client
        self.node_name = node_name or self.__class__.__name__
    
    @abstractmethod
    def run(self, input_data: Any, **kwargs) -> Any:
        """
        Execute node processing logic.
        
        Args:
            input_data: Input data
            **kwargs: Extra parameters
            
        Returns:
            Processing result
        """
        pass
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data.
        
        Args:
            input_data: Input data
            
        Returns:
            Whether validation passes
        """
        return True
    
    def process_output(self, output: Any) -> Any:
        """
        Process output data.
        
        Args:
            output: Raw output
            
        Returns:
            Processed output
        """
        return output
    
    def log_info(self, message: str):
        """Record an info log."""
        logger.info(f"[{self.node_name}] {message}")
    
    def log_warning(self, message: str):
        """Record a warning log."""
        logger.warning(f"[{self.node_name}] Warning: {message}")
    
    def log_error(self, message: str):
        """Record an error log."""
        logger.error(f"[{self.node_name}] Error: {message}")


class StateMutationNode(BaseNode):
    """Base class for nodes that mutate state."""
    
    @abstractmethod
    def mutate_state(self, input_data: Any, state: State, **kwargs) -> State:
        """
        Mutate state.
        
        Args:
            input_data: Input data
            state: Current state
            **kwargs: Extra parameters
            
        Returns:
            Mutated state
        """
        pass
