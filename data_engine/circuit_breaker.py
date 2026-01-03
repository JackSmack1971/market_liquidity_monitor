"""
Circuit Breaker mechanism for fault tolerance.

Prevents cascading failures by stopping execution when a service is failing consistently.
"""

import asyncio
from enum import Enum
from datetime import datetime, timedelta
from typing import Optional, Callable, Any, TypeVar
import logfire
from config import settings
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")

class CircuitBreakerOpen(Exception):
    """Raised when the circuit is open (failing)."""
    pass

# Initialize Metric Counters for Health Tracking
CB_TRIPS_COUNTER = logfire.metric_counter(
    "circuit_breaker_trips_total", 
    unit="1", 
    description="Total number of times a circuit breaker has opened"
)
CB_FAILURE_COUNTER = logfire.metric_counter(
    "circuit_breaker_failures_total",
    unit="1",
    description="Total number of individual failures recorded by circuit breakers"
)

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
                if settings.logfire_token:
                    logfire.info("circuit_recovered", circuit=self.name, state="CLOSED")
                
            return result
            
        except Exception as e:
            self._record_failure()
            logger.warning(f"Circuit '{self.name}' failure {self._failures}/{self.failure_threshold}: {e}")
            if settings.logfire_token:
                CB_FAILURE_COUNTER.add(1, {"circuit": self.name, "error": type(e).__name__})
                logfire.error(f"circuit_failure: {str(e)}", circuit=self.name, failures=self._failures)
            raise e

    def _record_failure(self):
        """Register a failure."""
        self._failures += 1
        self._last_failure_time = datetime.now()
        
        if self._failures >= self.failure_threshold:
            if self._state != "OPEN":
                self._state = "OPEN"
                logger.error(f"Circuit '{self.name}' OPENED after {self._failures} failures.")
                if settings.logfire_token:
                    CB_TRIPS_COUNTER.add(1, {"circuit": self.name})
                    logfire.error("circuit_opened", circuit=self.name, failures=self._failures)

    def _reset(self):
        """Reset circuit to closed state."""
        self._failures = 0
        self._state = "CLOSED"
        self._last_failure_time = None
