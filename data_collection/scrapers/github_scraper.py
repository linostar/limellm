import asyncio
import aiohttp
import json
import os
from typing import List, Dict, Optional
import logging
from datetime import datetime
import base64
from ..rate_limiter import AsyncRateLimiter

logger = logging.getLogger(__name__)

class GitHubScraper:
    def __init__(self, output_dir: str = "data/github", github_token: Optional[str] = None):
        self.output_dir = output_dir
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')
        self.rate_limiter = AsyncRateLimiter(requests_per_second=0.5)  # Conservative for GitHub
        self.scraped_repos: List[Dict] = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.github_token:
            logger.warning("No GitHub token provided. Rate limits will be more restrictive.")
    
    def get_headers(self):
        headers = {
            'User-Agent': 'LimeLLM-Bot/1.0 (Educational Purpose)',
            'Accept': 'application/vnd.github.v3+json'
        }
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'
        return headers
    
    async def search_python_repos(self, session: aiohttp.ClientSession, min_stars: int = 300) -> List[Dict]:
        await self.rate_limiter.acquire()
        
        search_url = "https://api.github.com/search/repositories"
        params = {
            'q': f'language:python stars:>={min_stars}',
            'sort': 'stars',
            'order': 'desc',
            'per_page': 100
        }
        
        repos = []
        page = 1
        max_pages = 10  # GitHub API limits to 1000 results
        
        while page <= max_pages:
            params['page'] = page
            
            try:
                async with session.get(search_url, params=params, headers=self.get_headers()) as response:
                    if response.status == 403:  # Rate limit exceeded
                        logger.warning("Rate limit exceeded, waiting...")
                        await asyncio.sleep(60)
                        continue
                    
                    if response.status != 200:
                        logger.error(f"GitHub API error: {response.status}")
                        break
                    
                    data = await response.json()
                    items = data.get('items', [])
                    
                    if not items:
                        break
                    
                    for repo in items:
                        repos.append({
                            'name': repo['full_name'],
                            'stars': repo['stargazers_count'],
                            'url': repo['html_url'],
                            'clone_url': repo['clone_url'],
                            'default_branch': repo['default_branch'],
                            'description': repo.get('description', ''),
                            'language': repo.get('language', 'Python')
                        })
                    
                    logger.info(f"Found {len(items)} repos on page {page}")
                    page += 1
                    
                    await self.rate_limiter.acquire()
                    
            except Exception as e:
                logger.error(f"Error searching repos: {e}")
                break
        
        return repos
    
    async def get_file_content(self, session: aiohttp.ClientSession, repo_name: str, file_path: str, branch: str = 'main') -> Optional[str]:
        await self.rate_limiter.acquire()
        
        url = f"https://api.github.com/repos/{repo_name}/contents/{file_path}"
        params = {'ref': branch}
        
        try:
            async with session.get(url, params=params, headers=self.get_headers()) as response:
                if response.status == 404:
                    return None
                
                if response.status != 200:
                    logger.warning(f"Failed to get {file_path} from {repo_name}: {response.status}")
                    return None
                
                data = await response.json()
                
                if data.get('encoding') == 'base64':
                    content = base64.b64decode(data['content']).decode('utf-8', errors='ignore')
                    return content
                
        except Exception as e:
            logger.error(f"Error fetching {file_path} from {repo_name}: {e}")
        
        return None
    
    async def get_repo_tree(self, session: aiohttp.ClientSession, repo_name: str, branch: str = 'main') -> List[str]:
        await self.rate_limiter.acquire()
        
        url = f"https://api.github.com/repos/{repo_name}/git/trees/{branch}"
        params = {'recursive': 1}
        
        try:
            async with session.get(url, params=params, headers=self.get_headers()) as response:
                if response.status != 200:
                    logger.warning(f"Failed to get tree for {repo_name}: {response.status}")
                    return []
                
                data = await response.json()
                python_files = []
                
                for item in data.get('tree', []):
                    if item['type'] == 'blob' and item['path'].endswith('.py'):
                        # Skip common non-essential files
                        skip_patterns = ['__pycache__', '.git', 'test_', 'tests/', 'setup.py', 'conftest.py']
                        if not any(pattern in item['path'] for pattern in skip_patterns):
                            python_files.append(item['path'])
                
                return python_files[:50]  # Limit files per repo
                
        except Exception as e:
            logger.error(f"Error getting repo tree for {repo_name}: {e}")
            return []
    
    async def scrape_repo(self, session: aiohttp.ClientSession, repo_info: Dict) -> Dict:
        repo_name = repo_info['name']
        branch = repo_info.get('default_branch', 'main')
        
        # Try main branch if default doesn't work
        if branch not in ['main', 'master']:
            branch = 'main'
        
        logger.info(f"Scraping repository: {repo_name}")
        
        # Get Python files
        python_files = await self.get_repo_tree(session, repo_name, branch)
        
        if not python_files:
            # Try master branch
            python_files = await self.get_repo_tree(session, repo_name, 'master')
            branch = 'master'
        
        repo_content = {
            'repo_name': repo_name,
            'stars': repo_info['stars'],
            'description': repo_info['description'],
            'files': []
        }
        
        # Get README
        readme_content = await self.get_file_content(session, repo_name, 'README.md', branch)
        if readme_content:
            repo_content['files'].append({
                'path': 'README.md',
                'content': readme_content,
                'type': 'documentation'
            })
        
        # Get Python files (limit to avoid overwhelming)
        for file_path in python_files[:20]:  # Limit files per repo
            content = await self.get_file_content(session, repo_name, file_path, branch)
            if content and len(content) > 100:  # Skip very small files
                repo_content['files'].append({
                    'path': file_path,
                    'content': content,
                    'type': 'code'
                })
        
        return repo_content
    
    async def scrape_all(self, min_stars: int = 300, max_repos: int = 100):
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        ) as session:
            
            # Search for Python repositories
            logger.info(f"Searching for Python repositories with {min_stars}+ stars...")
            repos = await self.search_python_repos(session, min_stars)
            
            logger.info(f"Found {len(repos)} repositories to scrape")
            
            # Scrape each repository
            for i, repo_info in enumerate(repos[:max_repos]):
                try:
                    repo_content = await self.scrape_repo(session, repo_info)
                    if repo_content['files']:  # Only save if we got some content
                        self.scraped_repos.append(repo_content)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Scraped {i + 1}/{min(len(repos), max_repos)} repositories")
                        
                except Exception as e:
                    logger.error(f"Error scraping {repo_info['name']}: {e}")
                    continue
            
            # Save results
            output_file = os.path.join(self.output_dir, "github_repos.jsonl")
            with open(output_file, 'w', encoding='utf-8') as f:
                for repo in self.scraped_repos:
                    f.write(json.dumps(repo, ensure_ascii=False) + '\n')
            
            logger.info(f"Scraped {len(self.scraped_repos)} repositories successfully")
            return self.scraped_repos

async def main():
    # Requires GitHub token for better rate limits
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("Warning: Set GITHUB_TOKEN environment variable for better rate limits")
    
    scraper = GitHubScraper(github_token=token)
    await scraper.scrape_all(min_stars=300, max_repos=50)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())