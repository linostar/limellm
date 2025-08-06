import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import os
from typing import List, Dict, Set
import logging
from ..rate_limiter import AsyncRateLimiter, respect_robots_txt

logger = logging.getLogger(__name__)

class PythonDocsScaper:
    def __init__(self, output_dir: str = "data/python_docs"):
        self.base_url = "https://docs.python.org/3/"
        self.output_dir = output_dir
        self.rate_limiter = AsyncRateLimiter(requests_per_second=1.0)
        self.visited_urls: Set[str] = set()
        self.scraped_content: List[Dict] = []
        
        os.makedirs(output_dir, exist_ok=True)
        
    async def scrape_page(self, session: aiohttp.ClientSession, url: str) -> Dict:
        if not respect_robots_txt(url, "LimeLLM-Bot"):
            logger.warning(f"Robots.txt disallows scraping {url}")
            return {}
            
        await self.rate_limiter.acquire()
        
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch {url}: {response.status}")
                    return {}
                    
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove navigation and footer elements
                for element in soup.find_all(['nav', 'footer', 'header', '.sphinxsidebar']):
                    element.decompose()
                
                # Extract main content
                main_content = soup.find('div', class_='body') or soup.find('main') or soup.find('article')
                if not main_content:
                    main_content = soup
                
                # Clean up code blocks and preserve formatting
                code_blocks = main_content.find_all(['pre', 'code'])
                for block in code_blocks:
                    block.string = block.get_text()
                
                title = soup.find('title')
                title_text = title.get_text().strip() if title else "Untitled"
                
                content = {
                    'url': url,
                    'title': title_text,
                    'content': main_content.get_text(),
                    'type': 'documentation',
                    'source': 'python_docs'
                }
                
                logger.info(f"Scraped: {title_text}")
                return content
                
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return {}
    
    def extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('#'):
                continue
                
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)
            
            # Only follow docs.python.org links
            if parsed.netloc == 'docs.python.org' and parsed.path.startswith('/3/'):
                # Skip certain sections we don't need
                skip_patterns = ['/tutorial/', '/installing/', '/distributing/', '/c-api/']
                if not any(pattern in parsed.path for pattern in skip_patterns):
                    links.append(full_url)
        
        return links
    
    async def scrape_all(self, max_pages: int = 1000):
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'LimeLLM-Bot (Educational Purpose)'}
        ) as session:
            
            # Start with main sections
            start_urls = [
                f"{self.base_url}library/index.html",
                f"{self.base_url}reference/index.html",
                f"{self.base_url}tutorial/index.html",
                f"{self.base_url}howto/index.html",
            ]
            
            to_visit = start_urls.copy()
            
            while to_visit and len(self.scraped_content) < max_pages:
                url = to_visit.pop(0)
                
                if url in self.visited_urls:
                    continue
                    
                self.visited_urls.add(url)
                
                # Scrape current page
                content = await self.scrape_page(session, url)
                if content:
                    self.scraped_content.append(content)
                
                # Get more links to follow
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            new_links = self.extract_links(soup, url)
                            
                            for link in new_links:
                                if link not in self.visited_urls and link not in to_visit:
                                    to_visit.append(link)
                except Exception as e:
                    logger.error(f"Error extracting links from {url}: {e}")
                
                if len(self.scraped_content) % 10 == 0:
                    logger.info(f"Scraped {len(self.scraped_content)} pages so far...")
            
            # Save results
            output_file = os.path.join(self.output_dir, "python_docs.jsonl")
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in self.scraped_content:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            logger.info(f"Scraped {len(self.scraped_content)} Python documentation pages")
            return self.scraped_content

async def main():
    scraper = PythonDocsScaper()
    await scraper.scrape_all(max_pages=500)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())