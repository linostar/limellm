#!/usr/bin/env python3

"""
LimeLLM Data Collection Runner

This script orchestrates the data collection process from all sources.
It handles parallel collection, progress monitoring, and error recovery.
"""

import asyncio
import json
import os
import sys
import argparse
import logging
from typing import Dict, List, Optional
from datetime import datetime
import signal
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_collection.data_collector import DataCollector
from data_collection.scrapers.python_docs_scraper import PythonDocsScaper
from data_collection.scrapers.github_scraper import GitHubScraper
from data_collection.scrapers.stackoverflow_scraper import StackOverflowScraper
from data_collection.scrapers.pypi_scraper import PyPIScraper

logger = logging.getLogger(__name__)

class DataCollectionRunner:
    """Orchestrate data collection from all sources."""
    
    def __init__(self, output_dir: str, config: Dict):
        self.output_dir = output_dir
        self.config = config
        self.collectors = {}
        self.stop_requested = False
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize collectors
        self._initialize_collectors()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop_requested = True
    
    def _initialize_collectors(self):
        """Initialize all data collectors."""
        
        if self.config.get('collect_python_docs', True):
            self.collectors['python_docs'] = PythonDocsScaper(
                output_dir=os.path.join(self.output_dir, 'python_docs')
            )
        
        if self.config.get('collect_github', True):
            github_token = os.getenv('GITHUB_TOKEN')
            self.collectors['github'] = GitHubScraper(
                output_dir=os.path.join(self.output_dir, 'github'),
                github_token=github_token
            )
        
        if self.config.get('collect_stackoverflow', True):
            self.collectors['stackoverflow'] = StackOverflowScraper(
                output_dir=os.path.join(self.output_dir, 'stackoverflow')
            )
        
        if self.config.get('collect_pypi', True):
            self.collectors['pypi'] = PyPIScraper(
                output_dir=os.path.join(self.output_dir, 'pypi')
            )
        
        logger.info(f"Initialized {len(self.collectors)} collectors: {list(self.collectors.keys())}")
    
    async def collect_python_docs(self) -> Dict:
        """Collect Python documentation."""
        if 'python_docs' not in self.collectors:
            return {'status': 'skipped', 'reason': 'disabled in config'}
        
        logger.info("Starting Python documentation collection...")
        try:
            collector = self.collectors['python_docs']
            max_pages = self.config.get('python_docs_max_pages', 500)
            
            result = await collector.scrape_all(max_pages=max_pages)
            
            return {
                'status': 'completed',
                'items_collected': len(result),
                'output_dir': collector.output_dir
            }
        
        except Exception as e:
            logger.error(f"Python docs collection failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def collect_github_repos(self) -> Dict:
        """Collect GitHub repositories."""
        if 'github' not in self.collectors:
            return {'status': 'skipped', 'reason': 'disabled in config'}
        
        logger.info("Starting GitHub repository collection...")
        try:
            collector = self.collectors['github']
            min_stars = self.config.get('github_min_stars', 300)
            max_repos = self.config.get('github_max_repos', 50)
            
            result = await collector.scrape_all(
                min_stars=min_stars,
                max_repos=max_repos
            )
            
            return {
                'status': 'completed',
                'items_collected': len(result),
                'output_dir': collector.output_dir
            }
        
        except Exception as e:
            logger.error(f"GitHub collection failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def collect_stackoverflow_qa(self) -> Dict:
        """Collect Stack Overflow Q&A."""
        if 'stackoverflow' not in self.collectors:
            return {'status': 'skipped', 'reason': 'disabled in config'}
        
        logger.info("Starting Stack Overflow Q&A collection...")
        try:
            collector = self.collectors['stackoverflow']
            min_votes = self.config.get('stackoverflow_min_votes', 10)
            max_questions = self.config.get('stackoverflow_max_questions', 500)
            
            result = await collector.scrape_all(
                min_votes=min_votes,
                max_questions=max_questions
            )
            
            return {
                'status': 'completed',
                'items_collected': len(result),
                'output_dir': collector.output_dir
            }
        
        except Exception as e:
            logger.error(f"Stack Overflow collection failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def collect_pypi_packages(self) -> Dict:
        """Collect PyPI package documentation."""
        if 'pypi' not in self.collectors:
            return {'status': 'skipped', 'reason': 'disabled in config'}
        
        logger.info("Starting PyPI package collection...")
        try:
            collector = self.collectors['pypi']
            max_packages = self.config.get('pypi_max_packages', 50)
            
            result = await collector.scrape_all(max_packages=max_packages)
            
            return {
                'status': 'completed',
                'items_collected': len(result),
                'output_dir': collector.output_dir
            }
        
        except Exception as e:
            logger.error(f"PyPI collection failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def run_collection(self) -> Dict:
        """Run the complete data collection process."""
        start_time = datetime.now()
        logger.info("Starting comprehensive data collection...")
        
        # Create collection tasks
        tasks = {}
        
        if self.config.get('collect_python_docs', True):
            tasks['python_docs'] = asyncio.create_task(self.collect_python_docs())
        
        if self.config.get('collect_github', True):
            tasks['github'] = asyncio.create_task(self.collect_github_repos())
        
        if self.config.get('collect_stackoverflow', True):
            tasks['stackoverflow'] = asyncio.create_task(self.collect_stackoverflow_qa())
        
        if self.config.get('collect_pypi', True):
            tasks['pypi'] = asyncio.create_task(self.collect_pypi_packages())
        
        # Monitor tasks
        results = {}
        completed_tasks = set()
        
        while len(completed_tasks) < len(tasks):
            if self.stop_requested:
                logger.info("Stop requested, cancelling remaining tasks...")
                for name, task in tasks.items():
                    if name not in completed_tasks and not task.done():
                        task.cancel()
                break
            
            # Check for completed tasks
            for name, task in tasks.items():
                if name not in completed_tasks and task.done():
                    try:
                        result = await task
                        results[name] = result
                        completed_tasks.add(name)
                        
                        status = result.get('status', 'unknown')
                        if status == 'completed':
                            items = result.get('items_collected', 0)
                            logger.info(f"✅ {name}: {items} items collected")
                        elif status == 'skipped':
                            reason = result.get('reason', 'unknown')
                            logger.info(f"⏭️  {name}: skipped ({reason})")
                        else:
                            error = result.get('error', 'unknown error')
                            logger.error(f"❌ {name}: failed ({error})")
                    
                    except asyncio.CancelledError:
                        results[name] = {'status': 'cancelled'}
                        completed_tasks.add(name)
                        logger.info(f"🛑 {name}: cancelled")
                    
                    except Exception as e:
                        results[name] = {'status': 'failed', 'error': str(e)}
                        completed_tasks.add(name)
                        logger.error(f"❌ {name}: failed with exception: {e}")
            
            # Brief pause to avoid busy waiting
            await asyncio.sleep(1)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Generate summary
        total_items = sum(
            result.get('items_collected', 0) 
            for result in results.values() 
            if result.get('status') == 'completed'
        )
        
        successful_collections = len([
            r for r in results.values() 
            if r.get('status') == 'completed'
        ])
        
        summary = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'total_items_collected': total_items,
            'successful_collections': successful_collections,
            'total_collections': len(tasks),
            'results': results,
            'config': self.config,
            'output_directory': self.output_dir
        }
        
        # Save summary
        summary_path = os.path.join(self.output_dir, 'collection_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Collection completed in {duration:.1f}s")
        logger.info(f"Total items collected: {total_items}")
        logger.info(f"Successful collections: {successful_collections}/{len(tasks)}")
        logger.info(f"Summary saved to: {summary_path}")
        
        return summary
    
    def print_final_report(self, summary: Dict):
        """Print a final collection report."""
        print("\n" + "="*60)
        print("📊 LIMELLM DATA COLLECTION REPORT")
        print("="*60)
        
        print(f"⏱️  Duration: {summary['duration_seconds']:.1f} seconds")
        print(f"📦 Total Items: {summary['total_items_collected']:,}")
        print(f"✅ Success Rate: {summary['successful_collections']}/{summary['total_collections']}")
        
        print("\n📋 Collection Results:")
        for source, result in summary['results'].items():
            status = result.get('status', 'unknown')
            
            if status == 'completed':
                items = result.get('items_collected', 0)
                print(f"  ✅ {source.ljust(15)}: {items:,} items")
            elif status == 'skipped':
                reason = result.get('reason', 'unknown')
                print(f"  ⏭️  {source.ljust(15)}: skipped ({reason})")
            elif status == 'cancelled':
                print(f"  🛑 {source.ljust(15)}: cancelled")
            else:
                error = result.get('error', 'unknown')[:50] + "..." if len(result.get('error', '')) > 50 else result.get('error', 'unknown')
                print(f"  ❌ {source.ljust(15)}: failed ({error})")
        
        print(f"\n📁 Data saved to: {summary['output_directory']}")
        print("="*60)

def load_config(config_path: Optional[str]) -> Dict:
    """Load configuration from file or return defaults."""
    
    default_config = {
        'collect_python_docs': True,
        'collect_github': True,
        'collect_stackoverflow': True,
        'collect_pypi': True,
        'python_docs_max_pages': 500,
        'github_min_stars': 300,
        'github_max_repos': 50,
        'stackoverflow_min_votes': 10,
        'stackoverflow_max_questions': 500,
        'pypi_max_packages': 50
    }
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            default_config.update(file_config)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config file {config_path}: {e}")
            logger.info("Using default configuration")
    else:
        logger.info("Using default configuration")
    
    return default_config

def main():
    parser = argparse.ArgumentParser(
        description='Run LimeLLM data collection from all sources'
    )
    
    parser.add_argument(
        '--output-dir', 
        default='data/raw',
        help='Output directory for collected data'
    )
    
    parser.add_argument(
        '--config',
        help='Configuration file path (JSON)'
    )
    
    parser.add_argument(
        '--python-docs-only',
        action='store_true',
        help='Collect only Python documentation'
    )
    
    parser.add_argument(
        '--github-only',
        action='store_true',
        help='Collect only GitHub repositories'
    )
    
    parser.add_argument(
        '--stackoverflow-only',
        action='store_true',
        help='Collect only Stack Overflow Q&A'
    )
    
    parser.add_argument(
        '--pypi-only',
        action='store_true',
        help='Collect only PyPI packages'
    )
    
    parser.add_argument(
        '--max-items',
        type=int,
        help='Maximum items to collect per source'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be collected without actually collecting'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = load_config(args.config)
    
    # Handle single-source flags
    if args.python_docs_only:
        config.update({
            'collect_python_docs': True,
            'collect_github': False,
            'collect_stackoverflow': False,
            'collect_pypi': False
        })
    elif args.github_only:
        config.update({
            'collect_python_docs': False,
            'collect_github': True,
            'collect_stackoverflow': False,
            'collect_pypi': False
        })
    elif args.stackoverflow_only:
        config.update({
            'collect_python_docs': False,
            'collect_github': False,
            'collect_stackoverflow': True,
            'collect_pypi': False
        })
    elif args.pypi_only:
        config.update({
            'collect_python_docs': False,
            'collect_github': False,
            'collect_stackoverflow': False,
            'collect_pypi': True
        })
    
    # Apply max items limit
    if args.max_items:
        config.update({
            'python_docs_max_pages': min(config['python_docs_max_pages'], args.max_items),
            'github_max_repos': min(config['github_max_repos'], args.max_items),
            'stackoverflow_max_questions': min(config['stackoverflow_max_questions'], args.max_items),
            'pypi_max_packages': min(config['pypi_max_packages'], args.max_items)
        })
    
    # Dry run
    if args.dry_run:
        print("🔍 DRY RUN - Configuration Preview:")
        print(json.dumps(config, indent=2))
        print(f"\n📁 Output directory: {args.output_dir}")
        print("\nSources to collect:")
        for source, enabled in [
            ('Python Docs', config['collect_python_docs']),
            ('GitHub', config['collect_github']),
            ('Stack Overflow', config['collect_stackoverflow']),
            ('PyPI', config['collect_pypi'])
        ]:
            status = "✅ Enabled" if enabled else "❌ Disabled"
            print(f"  {source}: {status}")
        return
    
    # Check for required environment variables
    if config.get('collect_github', True):
        if not os.getenv('GITHUB_TOKEN'):
            logger.warning("GITHUB_TOKEN not set. GitHub collection may hit rate limits.")
    
    # Run collection
    try:
        runner = DataCollectionRunner(args.output_dir, config)
        summary = asyncio.run(runner.run_collection())
        runner.print_final_report(summary)
        
        # Exit with appropriate code
        successful = summary['successful_collections']
        total = summary['total_collections']
        
        if successful == total:
            logger.info("All collections completed successfully!")
            sys.exit(0)
        elif successful > 0:
            logger.warning(f"Partial success: {successful}/{total} collections completed")
            sys.exit(1)
        else:
            logger.error("All collections failed!")
            sys.exit(2)
    
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()