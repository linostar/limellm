import asyncio
import aiohttp
import json
import os
from typing import List, Dict, Set
import logging
from ..rate_limiter import AsyncRateLimiter
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class PyPIScraper:
    def __init__(self, output_dir: str = "data/pypi"):
        self.output_dir = output_dir
        self.base_url = "https://pypi.org/pypi"
        self.rate_limiter = AsyncRateLimiter(requests_per_second=2.0)
        self.scraped_packages: List[Dict] = []
        
        # Most popular Python packages (top packages by downloads)
        self.popular_packages = [
            'requests', 'urllib3', 'boto3', 'botocore', 'setuptools', 'certifi',
            'python-dateutil', 'numpy', 'pandas', 'matplotlib', 'scipy', 'pillow',
            'django', 'flask', 'fastapi', 'sqlalchemy', 'pydantic', 'click',
            'pytest', 'black', 'flake8', 'mypy', 'isort', 'pre-commit',
            'tensorflow', 'torch', 'scikit-learn', 'transformers', 'opencv-python',
            'beautifulsoup4', 'lxml', 'scrapy', 'selenium', 'aiohttp',
            'celery', 'redis', 'psycopg2', 'pymongo', 'elasticsearch',
            'gunicorn', 'uvicorn', 'docker', 'kubernetes', 'pyyaml',
            'jinja2', 'werkzeug', 'itsdangerous', 'markupsafe', 'packaging',
            'wheel', 'pip', 'twine', 'tox', 'coverage', 'sphinx',
            'asyncio', 'multiprocessing', 'threading', 'json', 'argparse'
        ]
        
        os.makedirs(output_dir, exist_ok=True)
    
    def get_headers(self):
        return {
            'User-Agent': 'LimeLLM-Bot/1.0 (Educational Purpose)',
            'Accept': 'application/json'
        }
    
    async def get_package_info(self, session: aiohttp.ClientSession, package_name: str) -> Dict:
        await self.rate_limiter.acquire()
        
        url = f"{self.base_url}/{package_name}/json"
        
        try:
            async with session.get(url, headers=self.get_headers()) as response:
                if response.status == 404:
                    logger.warning(f"Package not found: {package_name}")
                    return {}
                
                if response.status != 200:
                    logger.warning(f"Failed to fetch {package_name}: {response.status}")
                    return {}
                
                data = await response.json()
                info = data.get('info', {})
                
                package_data = {
                    'name': info.get('name', package_name),
                    'version': info.get('version', ''),
                    'summary': info.get('summary', ''),
                    'description': info.get('description', ''),
                    'author': info.get('author', ''),
                    'home_page': info.get('home_page', ''),
                    'docs_url': info.get('docs_url', ''),
                    'project_urls': info.get('project_urls', {}),
                    'classifiers': info.get('classifiers', []),
                    'keywords': info.get('keywords', ''),
                    'license': info.get('license', ''),
                    'requires_dist': info.get('requires_dist', []),
                    'source': 'pypi'
                }
                
                logger.info(f"Scraped package info: {package_name}")
                return package_data
                
        except Exception as e:
            logger.error(f"Error scraping package {package_name}: {e}")
            return {}
    
    def extract_documentation_urls(self, package_info: Dict) -> List[str]:
        urls = []
        
        # Check various URL fields
        if package_info.get('docs_url'):
            urls.append(package_info['docs_url'])
        
        if package_info.get('home_page'):
            home_page = package_info['home_page']
            # Check if it looks like documentation
            if any(keyword in home_page.lower() for keyword in ['docs', 'documentation', 'readthedocs']):
                urls.append(home_page)
        
        # Check project URLs
        project_urls = package_info.get('project_urls', {})
        for key, url in project_urls.items():
            key_lower = key.lower()
            if any(keyword in key_lower for keyword in ['doc', 'documentation', 'manual', 'guide']):
                urls.append(url)
        
        return urls
    
    async def scrape_documentation_page(self, session: aiohttp.ClientSession, url: str) -> str:
        await self.rate_limiter.acquire()
        
        try:
            from bs4 import BeautifulSoup
            
            async with session.get(url, headers=self.get_headers(), timeout=30) as response:
                if response.status != 200:
                    return ""
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove navigation, footer, and other non-content elements
                for element in soup.find_all(['nav', 'footer', 'header', '.sidebar', '.navigation']):
                    element.decompose()
                
                # Find main content
                main_content = (
                    soup.find('main') or 
                    soup.find('div', class_='content') or
                    soup.find('div', class_='document') or
                    soup.find('article') or
                    soup.find('body')
                )
                
                if main_content:
                    # Preserve code blocks
                    for code_block in main_content.find_all(['pre', 'code']):
                        code_block.string = f"\n```python\n{code_block.get_text()}\n```\n"
                    
                    return main_content.get_text().strip()
                
        except Exception as e:
            logger.error(f"Error scraping documentation from {url}: {e}")
        
        return ""
    
    async def get_trending_packages(self, session: aiohttp.ClientSession) -> List[str]:
        # For now, return our curated list of popular packages
        # In a full implementation, you could scrape PyPI's trending page or use pypistats API
        return self.popular_packages
    
    async def scrape_all(self, max_packages: int = 100):
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        ) as session:
            
            # Get list of packages to scrape
            packages_to_scrape = self.popular_packages[:max_packages]
            
            logger.info(f"Scraping {len(packages_to_scrape)} popular Python packages...")
            
            # Scrape each package
            for i, package_name in enumerate(packages_to_scrape):
                try:
                    # Get package metadata
                    package_info = await self.get_package_info(session, package_name)
                    
                    if not package_info:
                        continue
                    
                    # Get documentation URLs and scrape them
                    doc_urls = self.extract_documentation_urls(package_info)
                    documentation_content = []
                    
                    for doc_url in doc_urls[:2]:  # Limit to 2 docs per package
                        if doc_url and doc_url.startswith('http'):
                            doc_content = await self.scrape_documentation_page(session, doc_url)
                            if doc_content and len(doc_content) > 500:  # Only include substantial content
                                documentation_content.append({
                                    'url': doc_url,
                                    'content': doc_content
                                })
                    
                    package_info['documentation'] = documentation_content
                    self.scraped_packages.append(package_info)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Scraped {i + 1}/{len(packages_to_scrape)} packages")
                        
                except Exception as e:
                    logger.error(f"Error processing package {package_name}: {e}")
                    continue
            
            # Save results
            output_file = os.path.join(self.output_dir, "pypi_packages.jsonl")
            with open(output_file, 'w', encoding='utf-8') as f:
                for package in self.scraped_packages:
                    f.write(json.dumps(package, ensure_ascii=False) + '\n')
            
            logger.info(f"Scraped {len(self.scraped_packages)} PyPI packages successfully")
            return self.scraped_packages

async def main():
    scraper = PyPIScraper()
    await scraper.scrape_all(max_packages=50)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())