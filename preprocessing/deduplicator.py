import hashlib
import json
from typing import Dict, List, Set, Tuple, Optional
import logging
from collections import defaultdict
import re
from difflib import SequenceMatcher
import numpy as np

logger = logging.getLogger(__name__)

class ContentDeduplicator:
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.seen_hashes: Set[str] = set()
        self.seen_content: List[str] = []
        self.deduplicated_items: List[Dict] = []
        self.duplicate_count = 0
        
    def compute_hash(self, text: str, hash_type: str = 'sha256') -> str:
        """Compute hash of normalized text."""
        if not text:
            return ""
        
        # Normalize text for hashing
        normalized = self.normalize_for_dedup(text)
        
        if hash_type == 'sha256':
            return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
        elif hash_type == 'md5':
            return hashlib.md5(normalized.encode('utf-8')).hexdigest()
        else:
            raise ValueError(f"Unsupported hash type: {hash_type}")
    
    def normalize_for_dedup(self, text: str) -> str:
        """Normalize text for deduplication (remove non-content differences)."""
        if not text:
            return ""
        
        # Basic normalization
        text = text.strip().lower()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common variations that don't affect content
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        
        return text
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        if not text1 or not text2:
            return 0.0
        
        # Use SequenceMatcher for character-level similarity
        matcher = SequenceMatcher(None, text1, text2)
        return matcher.ratio()
    
    def find_near_duplicates(self, text: str, threshold: float = None) -> List[Tuple[int, float]]:
        """Find near-duplicate content in already processed items."""
        if threshold is None:
            threshold = self.similarity_threshold
        
        near_duplicates = []
        normalized_text = self.normalize_for_dedup(text)
        
        for i, existing_text in enumerate(self.seen_content):
            similarity = self.compute_similarity(normalized_text, existing_text)
            if similarity >= threshold:
                near_duplicates.append((i, similarity))
        
        return near_duplicates
    
    def is_duplicate(self, item: Dict, content_fields: List[str] = None) -> Tuple[bool, Dict]:
        """Check if an item is a duplicate."""
        if content_fields is None:
            content_fields = ['content', 'body', 'text', 'description']
        
        # Extract main content
        main_content = ""
        for field in content_fields:
            if field in item and item[field]:
                main_content += str(item[field]) + " "
        
        if not main_content.strip():
            return True, {'reason': 'empty_content'}
        
        # Check exact duplicates first (faster)
        content_hash = self.compute_hash(main_content)
        if content_hash in self.seen_hashes:
            return True, {'reason': 'exact_duplicate', 'hash': content_hash}
        
        # Check near duplicates
        near_duplicates = self.find_near_duplicates(main_content)
        if near_duplicates:
            best_match = max(near_duplicates, key=lambda x: x[1])
            return True, {
                'reason': 'near_duplicate', 
                'similarity': best_match[1],
                'match_index': best_match[0]
            }
        
        # Not a duplicate, add to seen content
        self.seen_hashes.add(content_hash)
        self.seen_content.append(self.normalize_for_dedup(main_content))
        
        return False, {'reason': 'unique'}
    
    def deduplicate_batch(self, items: List[Dict], content_fields: List[str] = None) -> List[Dict]:
        """Deduplicate a batch of items."""
        unique_items = []
        duplicate_info = []
        
        for i, item in enumerate(items):
            is_dup, dup_info = self.is_duplicate(item, content_fields)
            
            if is_dup:
                self.duplicate_count += 1
                duplicate_info.append({
                    'index': i,
                    'item_id': item.get('id', f'item_{i}'),
                    **dup_info
                })
                logger.debug(f"Duplicate found: {dup_info['reason']} for item {i}")
            else:
                unique_items.append(item)
        
        logger.info(f"Deduplication complete: {len(unique_items)} unique items, {len(duplicate_info)} duplicates")
        
        return unique_items

class CodeDeduplicator:
    def __init__(self, similarity_threshold: float = 0.9):
        self.similarity_threshold = similarity_threshold
        self.seen_code_hashes: Set[str] = set()
        self.seen_code_normalized: List[str] = []
        
    def normalize_code(self, code: str) -> str:
        """Normalize code for deduplication."""
        if not code:
            return ""
        
        # Remove comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        
        # Remove empty lines
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        code = '\n'.join(lines)
        
        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code)
        
        # Remove docstrings (basic removal)
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
        
        return code.strip()
    
    def compute_code_hash(self, code: str) -> str:
        """Compute hash of normalized code."""
        normalized = self.normalize_code(code)
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def is_code_duplicate(self, code: str) -> Tuple[bool, Dict]:
        """Check if code is a duplicate."""
        if not code or not code.strip():
            return True, {'reason': 'empty_code'}
        
        # Check exact duplicates
        code_hash = self.compute_code_hash(code)
        if code_hash in self.seen_code_hashes:
            return True, {'reason': 'exact_duplicate', 'hash': code_hash}
        
        # Check near duplicates for code
        normalized_code = self.normalize_code(code)
        for i, existing_code in enumerate(self.seen_code_normalized):
            similarity = SequenceMatcher(None, normalized_code, existing_code).ratio()
            if similarity >= self.similarity_threshold:
                return True, {
                    'reason': 'near_duplicate',
                    'similarity': similarity,
                    'match_index': i
                }
        
        # Not a duplicate
        self.seen_code_hashes.add(code_hash)
        self.seen_code_normalized.append(normalized_code)
        
        return False, {'reason': 'unique'}

class AdvancedDeduplicator:
    def __init__(self, 
                 content_similarity_threshold: float = 0.85,
                 code_similarity_threshold: float = 0.9):
        self.content_dedup = ContentDeduplicator(content_similarity_threshold)
        self.code_dedup = CodeDeduplicator(code_similarity_threshold)
        
    def extract_code_from_item(self, item: Dict) -> List[str]:
        """Extract code snippets from an item."""
        code_snippets = []
        
        # Look for code in various fields
        text_fields = ['content', 'body', 'description', 'text']
        for field in text_fields:
            if field in item and item[field]:
                text = str(item[field])
                
                # Extract markdown code blocks
                code_pattern = re.compile(r'```(?:python|py)?\n(.*?)\n```', re.DOTALL | re.IGNORECASE)
                for match in code_pattern.finditer(text):
                    code_snippets.append(match.group(1))
                
                # Extract inline code
                inline_pattern = re.compile(r'`([^`\n]+)`')
                for match in inline_pattern.finditer(text):
                    code = match.group(1)
                    if any(keyword in code for keyword in ['def ', 'class ', 'import ', '=', '(']):
                        code_snippets.append(code)
        
        # If the item itself is code (e.g., from GitHub)
        if item.get('type') == 'code' or item.get('file_type') == 'python':
            code_snippets.append(item.get('content', ''))
        
        return code_snippets
    
    def deduplicate_comprehensive(self, items: List[Dict]) -> Dict:
        """Comprehensive deduplication considering both content and code."""
        unique_items = []
        content_duplicates = []
        code_duplicates = []
        stats = {
            'total_items': len(items),
            'unique_items': 0,
            'content_duplicates': 0,
            'code_duplicates': 0,
            'both_duplicates': 0
        }
        
        for i, item in enumerate(items):
            # Check content duplication
            is_content_dup, content_info = self.content_dedup.is_duplicate(item)
            
            # Check code duplication
            code_snippets = self.extract_code_from_item(item)
            is_code_dup = False
            code_info = {'reason': 'no_code'}
            
            if code_snippets:
                main_code = '\n'.join(code_snippets)
                is_code_dup, code_info = self.code_dedup.is_code_duplicate(main_code)
            
            # Decide whether to keep the item
            if is_content_dup and is_code_dup:
                stats['both_duplicates'] += 1
                logger.debug(f"Item {i}: Both content and code duplicate")
            elif is_content_dup:
                stats['content_duplicates'] += 1
                logger.debug(f"Item {i}: Content duplicate")
            elif is_code_dup:
                stats['code_duplicates'] += 1
                logger.debug(f"Item {i}: Code duplicate")
            else:
                unique_items.append(item)
                stats['unique_items'] += 1
            
            if i % 1000 == 0 and i > 0:
                logger.info(f"Processed {i}/{len(items)} items. Unique so far: {len(unique_items)}")
        
        return {
            'unique_items': unique_items,
            'stats': stats
        }

def deduplicate_jsonl_file(input_file: str, output_file: str, 
                          content_fields: List[str] = None,
                          similarity_threshold: float = 0.85) -> Dict:
    """Deduplicate a JSONL file."""
    if content_fields is None:
        content_fields = ['content', 'body', 'text', 'description']
    
    deduplicator = AdvancedDeduplicator(content_similarity_threshold=similarity_threshold)
    
    # Load items
    items = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                if line.strip():
                    item = json.loads(line)
                    items.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
    
    logger.info(f"Loaded {len(items)} items from {input_file}")
    
    # Deduplicate
    result = deduplicator.deduplicate_comprehensive(items)
    
    # Save unique items
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in result['unique_items']:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(result['unique_items'])} unique items to {output_file}")
    logger.info(f"Deduplication stats: {result['stats']}")
    
    return result

if __name__ == "__main__":
    # Test deduplication
    test_items = [
        {'id': 1, 'content': 'def hello(): print("Hello, World!")'},
        {'id': 2, 'content': 'def hello(): print("Hello, World!")'},  # Exact duplicate
        {'id': 3, 'content': 'def hello():\n    print("Hello, World!")'},  # Near duplicate (formatting)
        {'id': 4, 'content': 'def goodbye(): print("Goodbye!")'},  # Unique
        {'id': 5, 'content': ''},  # Empty
    ]
    
    deduplicator = AdvancedDeduplicator()
    result = deduplicator.deduplicate_comprehensive(test_items)
    
    print(f"Original items: {len(test_items)}")
    print(f"Unique items: {len(result['unique_items'])}")
    print(f"Stats: {result['stats']}")
    
    print("\nUnique items:")
    for item in result['unique_items']:
        print(f"ID {item['id']}: {item['content'][:50]}...")