import asyncio
import aiohttp
import json
import os
from typing import List, Dict
import logging
from urllib.parse import quote
import gzip
from datetime import datetime, timedelta
from ..rate_limiter import AsyncRateLimiter

logger = logging.getLogger(__name__)

class StackOverflowScraper:
    def __init__(self, output_dir: str = "data/stackoverflow"):
        self.output_dir = output_dir
        self.base_url = "https://api.stackexchange.com/2.3"
        self.rate_limiter = AsyncRateLimiter(requests_per_second=0.33)  # 1 request per 3 seconds
        self.scraped_qa: List[Dict] = []
        
        os.makedirs(output_dir, exist_ok=True)
    
    def get_headers(self):
        return {
            'User-Agent': 'LimeLLM-Bot/1.0 (Educational Purpose)',
            'Accept-Encoding': 'gzip'
        }
    
    async def search_python_questions(self, session: aiohttp.ClientSession, min_votes: int = 10) -> List[Dict]:
        await self.rate_limiter.acquire()
        
        url = f"{self.base_url}/questions"
        params = {
            'order': 'desc',
            'sort': 'votes',
            'tagged': 'python',
            'site': 'stackoverflow',
            'filter': 'withbody',
            'pagesize': 100,
            'min': min_votes
        }
        
        questions = []
        page = 1
        max_pages = 20  # Get up to 2000 questions
        
        while page <= max_pages:
            params['page'] = page
            
            try:
                async with session.get(url, params=params, headers=self.get_headers()) as response:
                    if response.status == 429:  # Rate limited
                        logger.warning("Rate limited, waiting 60 seconds...")
                        await asyncio.sleep(60)
                        continue
                    
                    if response.status != 200:
                        logger.error(f"StackOverflow API error: {response.status}")
                        break
                    
                    # Handle gzipped response
                    content = await response.read()
                    if response.headers.get('content-encoding') == 'gzip':
                        content = gzip.decompress(content)
                    
                    data = json.loads(content.decode('utf-8'))
                    items = data.get('items', [])
                    
                    if not items:
                        break
                    
                    for question in items:
                        # Only include questions with accepted answers or high score answers
                        if question.get('accepted_answer_id') or question.get('answer_count', 0) > 0:
                            questions.append(question)
                    
                    logger.info(f"Found {len(items)} questions on page {page}")
                    page += 1
                    
                    # Check if we've hit rate limit info
                    quota_remaining = data.get('quota_remaining', 0)
                    if quota_remaining < 100:
                        logger.warning(f"API quota running low: {quota_remaining} requests remaining")
                    
                    await self.rate_limiter.acquire()
                    
            except Exception as e:
                logger.error(f"Error searching questions: {e}")
                break
        
        return questions
    
    async def get_question_answers(self, session: aiohttp.ClientSession, question_id: int) -> List[Dict]:
        await self.rate_limiter.acquire()
        
        url = f"{self.base_url}/questions/{question_id}/answers"
        params = {
            'order': 'desc',
            'sort': 'votes',
            'site': 'stackoverflow',
            'filter': 'withbody',
            'pagesize': 100
        }
        
        try:
            async with session.get(url, params=params, headers=self.get_headers()) as response:
                if response.status != 200:
                    logger.warning(f"Failed to get answers for question {question_id}: {response.status}")
                    return []
                
                content = await response.read()
                if response.headers.get('content-encoding') == 'gzip':
                    content = gzip.decompress(content)
                
                data = json.loads(content.decode('utf-8'))
                return data.get('items', [])
                
        except Exception as e:
            logger.error(f"Error fetching answers for question {question_id}: {e}")
            return []
    
    def clean_html_content(self, html_content: str) -> str:
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Preserve code blocks
        for code_block in soup.find_all(['pre', 'code']):
            code_block.string = f"\n```python\n{code_block.get_text()}\n```\n"
        
        # Clean and return text
        return soup.get_text().strip()
    
    async def process_question_with_answers(self, session: aiohttp.ClientSession, question: Dict) -> Dict:
        question_id = question['question_id']
        
        # Get answers for this question
        answers = await self.get_question_answers(session, question_id)
        
        # Filter answers by votes (minimum 10 upvotes) or accepted answer
        good_answers = []
        for answer in answers:
            if answer.get('is_accepted') or answer.get('score', 0) >= 10:
                good_answers.append({
                    'answer_id': answer['answer_id'],
                    'score': answer.get('score', 0),
                    'is_accepted': answer.get('is_accepted', False),
                    'body': self.clean_html_content(answer.get('body', '')),
                    'creation_date': answer.get('creation_date', 0)
                })
        
        if not good_answers:
            return None
        
        return {
            'question_id': question_id,
            'title': question.get('title', ''),
            'question_body': self.clean_html_content(question.get('body', '')),
            'question_score': question.get('score', 0),
            'tags': question.get('tags', []),
            'answers': good_answers,
            'creation_date': question.get('creation_date', 0),
            'view_count': question.get('view_count', 0),
            'source': 'stackoverflow'
        }
    
    async def scrape_all(self, min_votes: int = 10, max_questions: int = 1000):
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        ) as session:
            
            # Search for Python questions
            logger.info(f"Searching for Python questions with {min_votes}+ votes...")
            questions = await self.search_python_questions(session, min_votes)
            
            logger.info(f"Found {len(questions)} questions to process")
            
            # Process each question and get its answers
            processed_count = 0
            for i, question in enumerate(questions[:max_questions]):
                try:
                    qa_pair = await self.process_question_with_answers(session, question)
                    if qa_pair and qa_pair['answers']:  # Only save if we have good answers
                        self.scraped_qa.append(qa_pair)
                        processed_count += 1
                    
                    if (i + 1) % 50 == 0:
                        logger.info(f"Processed {i + 1}/{min(len(questions), max_questions)} questions, saved {processed_count} Q&A pairs")
                        
                except Exception as e:
                    logger.error(f"Error processing question {question.get('question_id', 'unknown')}: {e}")
                    continue
            
            # Save results
            output_file = os.path.join(self.output_dir, "stackoverflow_qa.jsonl")
            with open(output_file, 'w', encoding='utf-8') as f:
                for qa in self.scraped_qa:
                    f.write(json.dumps(qa, ensure_ascii=False) + '\n')
            
            logger.info(f"Scraped {len(self.scraped_qa)} high-quality Q&A pairs from Stack Overflow")
            return self.scraped_qa

async def main():
    scraper = StackOverflowScraper()
    await scraper.scrape_all(min_votes=10, max_questions=500)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())