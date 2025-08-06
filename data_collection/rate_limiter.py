import time
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, requests_per_second: float = 1.0, burst_size: int = 5):
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self.min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0
        
    def acquire(self):
        now = time.time()
        elapsed = now - self.last_update
        
        # Add tokens based on elapsed time
        self.tokens = min(
            self.burst_size, 
            self.tokens + elapsed * self.requests_per_second
        )
        self.last_update = now
        
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        
        # Calculate wait time
        wait_time = (1 - self.tokens) / self.requests_per_second
        time.sleep(wait_time)
        self.tokens = 0
        self.last_update = time.time()
        return True

class AsyncRateLimiter:
    def __init__(self, requests_per_second: float = 1.0, burst_size: int = 5):
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        
    async def acquire(self):
        now = time.time()
        elapsed = now - self.last_update
        
        self.tokens = min(
            self.burst_size, 
            self.tokens + elapsed * self.requests_per_second
        )
        self.last_update = now
        
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        
        wait_time = (1 - self.tokens) / self.requests_per_second
        await asyncio.sleep(wait_time)
        self.tokens = 0
        self.last_update = time.time()
        return True

class DomainRateLimiter:
    def __init__(self):
        self.limiters: Dict[str, RateLimiter] = {}
        self.default_rates = {
            'github.com': 0.5,  # 1 request per 2 seconds
            'docs.python.org': 1.0,  # 1 request per second
            'pypi.org': 2.0,  # 2 requests per second
            'stackoverflow.com': 0.33,  # 1 request per 3 seconds
            'default': 1.0
        }
    
    def get_limiter(self, domain: str) -> RateLimiter:
        if domain not in self.limiters:
            rate = self.default_rates.get(domain, self.default_rates['default'])
            self.limiters[domain] = RateLimiter(requests_per_second=rate, burst_size=3)
            logger.info(f"Created rate limiter for {domain}: {rate} req/s")
        return self.limiters[domain]
    
    def acquire(self, domain: str) -> bool:
        limiter = self.get_limiter(domain)
        return limiter.acquire()

class RetryHandler:
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
    
    async def retry_with_backoff(self, func, *args, **kwargs):
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt == self.max_retries:
                    break
                    
                wait_time = (self.backoff_factor ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                
                if asyncio.iscoroutinefunction(func):
                    await asyncio.sleep(wait_time)
                else:
                    time.sleep(wait_time)
        
        raise last_exception

def respect_robots_txt(url: str, user_agent: str = "*") -> bool:
    try:
        from urllib.robotparser import RobotFileParser
        from urllib.parse import urljoin, urlparse
        
        parsed_url = urlparse(url)
        robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
        
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        
        return rp.can_fetch(user_agent, url)
    except Exception as e:
        logger.warning(f"Could not check robots.txt for {url}: {e}")
        return True  # If we can't check, assume it's allowed