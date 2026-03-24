"""
Retry utility module.
Provides generic retry support for network requests to improve system robustness.
"""

import time
from functools import wraps
from typing import Callable, Any
import requests
from loguru import logger

class RetryConfig:
    """Retry configuration class."""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        retry_on_exceptions: tuple = None
    ):
        """
        Initialize retry configuration.

        Args:
            max_retries: Maximum number of retries.
            initial_delay: Initial delay in seconds.
            backoff_factor: Backoff multiplier for each retry.
            max_delay: Maximum delay in seconds.
            retry_on_exceptions: Tuple of exception types that should be retried.
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        
        # Default exception types to retry.
        if retry_on_exceptions is None:
            self.retry_on_exceptions = (
                requests.exceptions.RequestException,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
                requests.exceptions.Timeout,
                requests.exceptions.TooManyRedirects,
                ConnectionError,
                TimeoutError,
                Exception  # Generic exception sometimes raised by OpenAI and other APIs.
            )
        else:
            self.retry_on_exceptions = retry_on_exceptions

# Default configuration.
DEFAULT_RETRY_CONFIG = RetryConfig()

def with_retry(config: RetryConfig = None):
    """Retry decorator.

    Args:
        config: Retry configuration. Uses default config when omitted.

    Returns:
        Decorator function.
    """
    if config is None:
        config = DEFAULT_RETRY_CONFIG
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(config.max_retries + 1):  # +1 because the first call is not a retry.
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"Function {func.__name__} succeeded on attempt {attempt + 1}")
                    return result
                    
                except config.retry_on_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_retries:
                        # Final attempt failed.
                        logger.error(f"Function {func.__name__} still failed after {config.max_retries + 1} attempts")
                        logger.error(f"Final error: {str(e)}")
                        raise e
                    
                    # Compute retry delay.
                    delay = min(
                        config.initial_delay * (config.backoff_factor ** attempt),
                        config.max_delay
                    )
                    
                    logger.warning(f"Function {func.__name__} failed on attempt {attempt + 1}: {str(e)}")
                    logger.info(f"Retrying in {delay:.1f} seconds (attempt {attempt + 2})...")
                    
                    time.sleep(delay)
                
                except Exception as e:
                    # Exception is not retryable; raise immediately.
                    logger.error(f"Function {func.__name__} hit a non-retryable exception: {str(e)}")
                    raise e
            
            # Safety fallback (should not be reached).
            if last_exception:
                raise last_exception
            
        return wrapper
    return decorator

def retry_on_network_error(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
):
    """Retry decorator specialized for network errors (simplified).

    Args:
        max_retries: Maximum number of retries.
        initial_delay: Initial delay in seconds.
        backoff_factor: Backoff multiplier.

    Returns:
        Decorator function.
    """
    config = RetryConfig(
        max_retries=max_retries,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor
    )
    return with_retry(config)

class RetryableError(Exception):
    """Custom retryable exception."""
    pass

def with_graceful_retry(config: RetryConfig = None, default_return=None):
    """Graceful retry decorator for non-critical API calls.

    If all retries fail, it returns a default value instead of raising,
    allowing the system to keep running.

    Args:
        config: Retry configuration. Uses default search API retry config if omitted.
        default_return: Value returned after all retries fail.

    Returns:
        Decorator function.
    """
    if config is None:
        config = SEARCH_API_RETRY_CONFIG
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(config.max_retries + 1):  # +1 because the first call is not a retry.
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"Non-critical API {func.__name__} succeeded on attempt {attempt + 1}")
                    return result
                    
                except config.retry_on_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_retries:
                        # Final attempt failed; return default value instead of raising.
                        logger.warning(f"Non-critical API {func.__name__} still failed after {config.max_retries + 1} attempts")
                        logger.warning(f"Final error: {str(e)}")
                        logger.info(f"Returning default value to keep system running: {default_return}")
                        return default_return
                    
                    # Compute retry delay.
                    delay = min(
                        config.initial_delay * (config.backoff_factor ** attempt),
                        config.max_delay
                    )
                    
                    logger.warning(f"Non-critical API {func.__name__} failed on attempt {attempt + 1}: {str(e)}")
                    logger.info(f"Retrying in {delay:.1f} seconds (attempt {attempt + 2})...")
                    
                    time.sleep(delay)
                
                except Exception as e:
                    # Exception is not retryable; return default value.
                    logger.warning(f"Non-critical API {func.__name__} hit a non-retryable exception: {str(e)}")
                    logger.info(f"Returning default value to keep system running: {default_return}")
                    return default_return
            
            # Safety fallback (should not be reached).
            return default_return
            
        return wrapper
    return decorator

def make_retryable_request(
    request_func: Callable,
    *args,
    max_retries: int = 5,
    **kwargs
) -> Any:
    """Execute a retryable request directly (without external decoration).

    Args:
        request_func: Request function to execute.
        *args: Positional arguments passed to request_func.
        max_retries: Maximum number of retries.
        **kwargs: Keyword arguments passed to request_func.

    Returns:
        Return value from request_func.
    """
    config = RetryConfig(max_retries=max_retries)
    
    @with_retry(config)
    def _execute():
        return request_func(*args, **kwargs)
    
    return _execute()

# Predefined commonly used retry configurations.
LLM_RETRY_CONFIG = RetryConfig(
    max_retries=6,        # Keep extra retries.
    initial_delay=60.0,   # Wait at least 1 minute before first retry.
    backoff_factor=2.0,   # Keep exponential backoff.
    max_delay=600.0       # Maximum single wait is 10 minutes.
)

SEARCH_API_RETRY_CONFIG = RetryConfig(
    max_retries=5,        # Increased to 5 retries.
    initial_delay=2.0,    # Increased initial delay.
    backoff_factor=1.6,   # Tuned backoff factor.
    max_delay=25.0        # Increased maximum delay.
)

DB_RETRY_CONFIG = RetryConfig(
    max_retries=5,        # Increased to 5 retries.
    initial_delay=1.0,    # Keep database retry delay short.
    backoff_factor=1.5,
    max_delay=10.0
)
