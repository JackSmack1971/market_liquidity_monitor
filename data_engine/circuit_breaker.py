"""
Circuit Breaker mechanism for fault tolerance.

Prevents cascading failures by stopping execution when a service is failing consistently.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Callable, Any, TypeVar, Optional
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")

class CircuitBreakerOpen(Exception):
    """Raised when the circuit is open (failing)."""
    pass

class CircuitBreaker:
    """
    State machine for fault tolerance.
    
    States:
    - CLOSED: Normal operation.
    - OPEN: Failing, requests blocked.
    - HALF_OPEN: Testing recovery.
    """

    def __init__(
        self, 
        name: str = "default",
        failure_threshold: int = 5, 
        recovery_timeout: int = 60
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Identifier for logging
            failure_threshold: Number of failures before opening
            recovery_timeout: Seconds to wait before testing recovery
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failures = 0
        self._last_failure_time: Optional[datetime] = None
        self._state = "CLOSED" 

    @property
    def state(self) -> str:
        """Current state of the circuit."""
        if self._state == "OPEN":
            if self._last_failure_time and (datetime.now() - self._last_failure_time) > timedelta(seconds=self.recovery_timeout):
                return "HALF_OPEN"
        return self._state

    async def call(self, func: Callable[..., Any], *args, **kwargs) -> T:
        """
        Execute function with circuit breaker protection.
        
        Raises:
            CircuitBreakerOpen: If circuit is OPEN
            Exception: If underlying function fails
        """
        current_state = self.state
        
        if current_state == "OPEN":
            raise CircuitBreakerOpen(f"Circuit '{self.name}' is OPEN. Requests blocked.")

        try:
            result = await func(*args, **kwargs)
            
            if current_state == "HALF_OPEN":
                self._reset()
                logger.info(f"Circuit '{self.name}' recovered. State -> CLOSED")
                
            return result
            
        except Exception as e:
            self._record_failure()
            logger.warning(f"Circuit '{self.name}' failure {self._failures}/{self.failure_threshold}: {e}")
            raise e

    def _record_failure(self):
        """Register a failure."""
        self._failures += 1
        self._last_failure_time = datetime.now()
        
        if self._failures >= self.failure_threshold:
            if self._state != "OPEN":
                self._state = "OPEN"
                logger.error(f"Circuit '{self.name}' OPENED after {self._failures} failures.")

    def _reset(self):
        """Reset circuit to closed state."""
        self._failures = 0
        self._state = "CLOSED"
        self._last_failure_time = None
