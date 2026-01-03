import unittest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from data_engine.circuit_breaker import CircuitBreaker, CircuitBreakerOpen

class TestCircuitBreaker(unittest.IsolatedAsyncioTestCase):
    """Comprehensive tests for CircuitBreaker."""

    def setUp(self):
        """Setup test fixtures."""
        self.cb = CircuitBreaker(name="test_cb", failure_threshold=3, recovery_timeout=1)

    async def asyncSetUp(self):
        # We need this for IsolatedAsyncioTestCase if any async setup is needed
        pass

    # ========== HAPPY PATH TESTS ==========

    def test_init_happy_path_typical_input(self):
        """Test CircuitBreaker initialization with typical values."""
        cb = CircuitBreaker(name="custom", failure_threshold=5, recovery_timeout=30)
        self.assertEqual(cb.name, "custom")
        self.assertEqual(cb.failure_threshold, 5)
        self.assertEqual(cb.recovery_timeout, 30)
        self.assertEqual(cb._failures, 0)
        self.assertEqual(cb._state, "CLOSED")

    async def test_call_happy_path_success(self):
        """Test call with typical valid input in CLOSED state."""
        async def mock_func(val):
            return val * 2
        
        result = await self.cb.call(mock_func, 10)
        self.assertEqual(result, 20)
        self.assertEqual(self.cb.state, "CLOSED")
        self.assertEqual(self.cb._failures, 0)

    async def test_recovery_happy_path_half_open_to_closed(self):
        """Test transition from HALF_OPEN to CLOSED on success."""
        # 1. Force OPEN
        for _ in range(3):
            try:
                await self.cb.call(MagicMock(side_effect=Exception("fail")))
            except:
                pass
        
        self.assertEqual(self.cb._state, "OPEN")
        
        # 2. Advance time to reach HALF_OPEN
        future_time = datetime.now() + timedelta(seconds=2)
        with patch('data_engine.circuit_breaker.datetime') as mock_datetime:
            mock_datetime.now.return_value = future_time
            self.assertEqual(self.cb.state, "HALF_OPEN")
            
            # 3. Call successfully
            async def success_func(): return "ok"
            result = await self.cb.call(success_func)
            
            self.assertEqual(result, "ok")
            self.assertEqual(self.cb.state, "CLOSED")
            self.assertEqual(self.cb._failures, 0)

    # ========== EDGE CASE TESTS ==========

    async def test_failure_threshold_edge_exactly(self):
        """Test that circuit opens exactly at threshold."""
        async def failing_func():
            raise Exception("error")

        # Threshold is 3
        # failure 1
        with self.assertRaises(Exception):
            await self.cb.call(failing_func)
        self.assertEqual(self.cb._failures, 1)
        self.assertEqual(self.cb.state, "CLOSED")

        # failure 2
        with self.assertRaises(Exception):
            await self.cb.call(failing_func)
        self.assertEqual(self.cb._failures, 2)
        self.assertEqual(self.cb.state, "CLOSED")

        # failure 3 -> should OPEN
        with self.assertRaises(Exception):
            await self.cb.call(failing_func)
        self.assertEqual(self.cb._failures, 3)
        self.assertEqual(self.cb._state, "OPEN")

    async def test_half_open_transition_edge_boundary(self):
        """Test transition to HALF_OPEN exactly after timeout."""
        self.cb._state = "OPEN"
        now = datetime.now()
        self.cb._last_failure_time = now
        
        # Test just before timeout
        with patch('data_engine.circuit_breaker.datetime') as mock_datetime:
            mock_datetime.now.return_value = now + timedelta(seconds=0.5)
            self.assertEqual(self.cb.state, "OPEN")
            
            # Test exactly at timeout (or just after)
            mock_datetime.now.return_value = now + timedelta(seconds=1.1)
            self.assertEqual(self.cb.state, "HALF_OPEN")

    async def test_half_open_failure_edge_reenters_open(self):
        """Test that failure in HALF_OPEN re-enters OPEN immediately."""
        self.cb._state = "OPEN"
        self.cb._last_failure_time = datetime.now() - timedelta(seconds=2)
        self.assertEqual(self.cb.state, "HALF_OPEN")
        
        async def failing_func():
            raise Exception("still failing")
            
        with self.assertRaises(Exception):
            await self.cb.call(failing_func)
            
        self.assertEqual(self.cb._state, "OPEN")
        # Failures should increment
        self.assertEqual(self.cb._failures, 1) 

    # ========== ERROR SCENARIO TESTS ==========

    async def test_raises_CircuitBreakerOpen_when_open(self):
        """Test raises CircuitBreakerOpen when state is OPEN."""
        self.cb._state = "OPEN"
        async def some_func(): return "wont run"
        
        with self.assertRaises(CircuitBreakerOpen):
            await self.cb.call(some_func)

    async def test_propagates_exception_when_underlying_fails(self):
        """Test CircuitBreaker propagates the underlying exception."""
        class CustomError(Exception): pass
        
        async def failing_func():
            raise CustomError("custom")
            
        with self.assertRaises(CustomError):
            await self.cb.call(failing_func)

if __name__ == '__main__':
    unittest.main()
