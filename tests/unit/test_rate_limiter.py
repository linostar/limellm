import pytest
import time
import asyncio
from unittest.mock import patch, MagicMock

from data_collection.rate_limiter import RateLimiter, AsyncRateLimiter


class TestRateLimiter:
    """Test suite for RateLimiter class."""
    
    def test_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(requests_per_second=2.0, burst_size=10)
        
        assert limiter.requests_per_second == 2.0
        assert limiter.burst_size == 10
        assert limiter.tokens == 10
        assert limiter.min_interval == 0.5
    
    def test_burst_requests(self):
        """Test that burst requests are handled correctly."""
        limiter = RateLimiter(requests_per_second=1.0, burst_size=3)
        
        # First 3 requests should go through immediately
        start_time = time.time()
        for i in range(3):
            assert limiter.acquire() is True
        end_time = time.time()
        
        # Should complete quickly (within burst)
        assert end_time - start_time < 0.1
    
    def test_rate_limiting(self):
        """Test that rate limiting kicks in after burst."""
        limiter = RateLimiter(requests_per_second=10.0, burst_size=2)  # Fast rate for testing
        
        # Use up burst
        for i in range(2):
            limiter.acquire()
        
        # Next request should be rate limited
        start_time = time.time()
        limiter.acquire()
        end_time = time.time()
        
        # Should have waited at least some time
        assert end_time - start_time > 0.05  # Should wait ~0.1 seconds for 10 rps
    
    def test_token_replenishment(self):
        """Test that tokens are replenished over time."""
        limiter = RateLimiter(requests_per_second=100.0, burst_size=2)  # Fast replenishment
        
        # Use up all tokens
        limiter.acquire()
        limiter.acquire()
        assert limiter.tokens < 1
        
        # Wait for replenishment
        time.sleep(0.1)
        
        # Should be able to acquire again
        start_time = time.time()
        limiter.acquire()
        end_time = time.time()
        
        # Should not have waited long due to replenishment
        assert end_time - start_time < 0.05
    
    def test_zero_rate_limiter(self):
        """Test rate limiter with zero rate (unlimited)."""
        limiter = RateLimiter(requests_per_second=0, burst_size=1)
        
        # Should allow immediate requests
        for i in range(10):
            start_time = time.time()
            assert limiter.acquire() is True
            end_time = time.time()
            assert end_time - start_time < 0.01


class TestAsyncRateLimiter:
    """Test suite for AsyncRateLimiter class."""
    
    @pytest.mark.asyncio
    async def test_async_initialization(self):
        """Test async rate limiter initialization."""
        limiter = AsyncRateLimiter(requests_per_second=2.0, burst_size=5)
        
        assert limiter.requests_per_second == 2.0
        assert limiter.burst_size == 5
        assert limiter.tokens == 5
    
    @pytest.mark.asyncio
    async def test_async_burst_requests(self):
        """Test async burst request handling."""
        limiter = AsyncRateLimiter(requests_per_second=1.0, burst_size=3)
        
        start_time = time.time()
        for i in range(3):
            await limiter.acquire()
        end_time = time.time()
        
        # Should complete quickly
        assert end_time - start_time < 0.1
    
    @pytest.mark.asyncio
    async def test_async_rate_limiting(self):
        """Test async rate limiting behavior."""
        limiter = AsyncRateLimiter(requests_per_second=10.0, burst_size=1)
        
        # First request uses up burst
        await limiter.acquire()
        
        # Second request should be rate limited
        start_time = time.time()
        await limiter.acquire()
        end_time = time.time()
        
        # Should have waited
        assert end_time - start_time > 0.05
    
    @pytest.mark.asyncio
    async def test_async_concurrent_requests(self):
        """Test handling concurrent async requests."""
        limiter = AsyncRateLimiter(requests_per_second=5.0, burst_size=2)
        
        async def make_request():
            return await limiter.acquire()
        
        # Create multiple concurrent tasks
        start_time = time.time()
        tasks = [make_request() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # All should succeed
        assert all(results)
        # Should have taken some time due to rate limiting
        assert end_time - start_time > 0.3  # At least 3 requests beyond burst at 5 rps


class TestRateLimiterIntegration:
    """Integration tests for rate limiter usage patterns."""
    
    def test_realistic_web_scraping_pattern(self):
        """Test rate limiter in a realistic web scraping scenario."""
        # Simulate GitHub API rate limiting (5000 requests/hour)
        limiter = RateLimiter(requests_per_second=1.39, burst_size=10)  # ~5000/hour
        
        request_times = []
        
        # Simulate 15 API calls
        for i in range(15):
            start = time.time()
            limiter.acquire()
            end = time.time()
            request_times.append(end - start)
        
        # First 10 should be fast (burst), rest should be rate limited
        assert sum(request_times[:10]) < 1.0  # Burst requests fast
        assert sum(request_times[10:]) > 2.0  # Remaining requests rate limited
    
    @pytest.mark.asyncio
    async def test_realistic_async_pattern(self):
        """Test async rate limiter in realistic scenario."""
        limiter = AsyncRateLimiter(requests_per_second=2.0, burst_size=5)
        
        async def api_call(request_id):
            await limiter.acquire()
            # Simulate API processing time
            await asyncio.sleep(0.01)
            return f"response_{request_id}"
        
        start_time = time.time()
        tasks = [api_call(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Should get all responses
        assert len(results) == 10
        assert all("response_" in r for r in results)
        
        # Should have taken time due to rate limiting
        # 5 burst + 5 rate limited at 2/sec = at least 2.5 seconds
        assert end_time - start_time > 2.0