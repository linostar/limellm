import asyncio
import logging
import json
import os
from typing import Dict, List
import argparse
from scrapers.python_docs_scraper import PythonDocsScaper
from scrapers.github_scraper import GitHubScraper
from scrapers.stackoverflow_scraper import StackOverflowScraper
from scrapers.pypi_scraper import PyPIScraper

logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        self.scrapers = {}
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize scrapers
        self.scrapers['python_docs'] = PythonDocsScaper(
            output_dir=os.path.join(output_dir, 'python_docs')
        )
        self.scrapers['github'] = GitHubScraper(
            output_dir=os.path.join(output_dir, 'github')
        )
        self.scrapers['stackoverflow'] = StackOverflowScraper(
            output_dir=os.path.join(output_dir, 'stackoverflow')
        )
        self.scrapers['pypi'] = PyPIScraper(
            output_dir=os.path.join(output_dir, 'pypi')
        )
    
    async def collect_all_data(self, config: Dict):
        logger.info("Starting comprehensive data collection...")
        
        tasks = []
        
        # Python Documentation
        if config.get('collect_python_docs', True):
            tasks.append(self.scrapers['python_docs'].scrape_all(
                max_pages=config.get('python_docs_max_pages', 500)
            ))
        
        # GitHub Repositories
        if config.get('collect_github', True):
            tasks.append(self.scrapers['github'].scrape_all(
                min_stars=config.get('github_min_stars', 300),
                max_repos=config.get('github_max_repos', 50)
            ))
        
        # Stack Overflow Q&A
        if config.get('collect_stackoverflow', True):
            tasks.append(self.scrapers['stackoverflow'].scrape_all(
                min_votes=config.get('stackoverflow_min_votes', 10),
                max_questions=config.get('stackoverflow_max_questions', 500)
            ))
        
        # PyPI Packages
        if config.get('collect_pypi', True):
            tasks.append(self.scrapers['pypi'].scrape_all(
                max_packages=config.get('pypi_max_packages', 50)
            ))
        
        # Run all scrapers concurrently
        logger.info(f"Running {len(tasks)} scrapers concurrently...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log results
        for i, result in enumerate(results):
            scraper_name = list(self.scrapers.keys())[i]
            if isinstance(result, Exception):
                logger.error(f"Scraper {scraper_name} failed: {result}")
            else:
                logger.info(f"Scraper {scraper_name} completed successfully with {len(result) if isinstance(result, list) else 'unknown'} items")
        
        # Combine all data into summary
        summary = await self.generate_collection_summary()
        
        summary_file = os.path.join(self.output_dir, "collection_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Data collection complete. Summary saved to {summary_file}")
        return summary
    
    async def generate_collection_summary(self) -> Dict:
        summary = {
            'collection_date': asyncio.get_event_loop().time(),
            'sources': {}
        }
        
        # Count items from each source
        for source_name, scraper in self.scrapers.items():
            source_dir = scraper.output_dir
            files = []
            total_items = 0
            
            if os.path.exists(source_dir):
                for filename in os.listdir(source_dir):
                    if filename.endswith('.jsonl'):
                        filepath = os.path.join(source_dir, filename)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                items = sum(1 for line in f if line.strip())
                                files.append({
                                    'filename': filename,
                                    'items': items,
                                    'size_mb': os.path.getsize(filepath) / (1024 * 1024)
                                })
                                total_items += items
                        except Exception as e:
                            logger.error(f"Error reading {filepath}: {e}")
            
            summary['sources'][source_name] = {
                'total_items': total_items,
                'files': files
            }
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='Collect training data for LimeLLM')
    parser.add_argument('--output-dir', default='data', help='Output directory for collected data')
    parser.add_argument('--config', help='JSON config file with collection parameters')
    parser.add_argument('--python-docs-only', action='store_true', help='Only collect Python docs')
    parser.add_argument('--github-only', action='store_true', help='Only collect GitHub repos')
    parser.add_argument('--stackoverflow-only', action='store_true', help='Only collect Stack Overflow')
    parser.add_argument('--pypi-only', action='store_true', help='Only collect PyPI packages')
    
    args = parser.parse_args()
    
    # Load config
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Set collection flags based on arguments
    if args.python_docs_only:
        config = {'collect_python_docs': True, 'collect_github': False, 'collect_stackoverflow': False, 'collect_pypi': False}
    elif args.github_only:
        config = {'collect_python_docs': False, 'collect_github': True, 'collect_stackoverflow': False, 'collect_pypi': False}
    elif args.stackoverflow_only:
        config = {'collect_python_docs': False, 'collect_github': False, 'collect_stackoverflow': True, 'collect_pypi': False}
    elif args.pypi_only:
        config = {'collect_python_docs': False, 'collect_github': False, 'collect_stackoverflow': False, 'collect_pypi': True}
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create collector and run
    collector = DataCollector(output_dir=args.output_dir)
    
    try:
        summary = asyncio.run(collector.collect_all_data(config))
        print(f"\nCollection Summary:")
        for source, info in summary['sources'].items():
            print(f"{source}: {info['total_items']} items")
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
    except Exception as e:
        logger.error(f"Collection failed: {e}")

if __name__ == "__main__":
    main()