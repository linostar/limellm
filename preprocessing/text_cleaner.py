import re
import html
from typing import List, Dict, Optional
import logging
from bs4 import BeautifulSoup
import unicodedata

logger = logging.getLogger(__name__)

class TextCleaner:
    def __init__(self):
        # Patterns for cleaning
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.excessive_whitespace = re.compile(r'\s+')
        self.code_block_pattern = re.compile(r'```[\w]*\n(.*?)\n```', re.DOTALL)
        self.inline_code_pattern = re.compile(r'`([^`]+)`')
        
        # Common non-informative patterns
        self.noise_patterns = [
            re.compile(r'^\s*#\s*-+\s*$', re.MULTILINE),  # Comment separators
            re.compile(r'^\s*#+\s*$', re.MULTILINE),       # Empty headers
            re.compile(r'^\s*\*+\s*$', re.MULTILINE),      # Star separators
            re.compile(r'^\s*=+\s*$', re.MULTILINE),       # Equal separators
            re.compile(r'^\s*-+\s*$', re.MULTILINE),       # Dash separators
        ]
        
        # File extensions and patterns to preserve as code
        self.code_indicators = ['.py', '.python', 'def ', 'class ', 'import ', 'from ']
    
    def clean_html(self, text: str) -> str:
        """Remove HTML tags and decode entities."""
        if not text:
            return ""
        
        # Decode HTML entities first
        text = html.unescape(text)
        
        # Use BeautifulSoup for robust HTML cleaning
        soup = BeautifulSoup(text, 'html.parser')
        
        # Preserve code blocks with special formatting
        for pre_tag in soup.find_all('pre'):
            pre_tag.string = f"\n```\n{pre_tag.get_text()}\n```\n"
        
        for code_tag in soup.find_all('code'):
            if code_tag.parent.name != 'pre':  # Don't double-wrap
                code_tag.string = f"`{code_tag.get_text()}`"
        
        # Extract text content
        return soup.get_text()
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        if not text:
            return ""
        
        # Normalize to NFC form
        text = unicodedata.normalize('NFC', text)
        
        # Replace common unicode quotes and dashes
        replacements = {
            '"': '"', '"': '"',  # Smart quotes
            ''': "'", ''': "'",  # Smart apostrophes  
            '–': '-', '—': '-',  # En dash, em dash
            '…': '...',          # Ellipsis
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def clean_whitespace(self, text: str) -> str:
        """Clean excessive whitespace while preserving code formatting."""
        if not text:
            return ""
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Preserve indentation for code-like content
            if any(indicator in line for indicator in self.code_indicators):
                # Minimal cleaning for code lines
                cleaned_lines.append(line.rstrip())
            else:
                # More aggressive cleaning for prose
                cleaned = self.excessive_whitespace.sub(' ', line).strip()
                if cleaned:  # Only add non-empty lines
                    cleaned_lines.append(cleaned)
        
        return '\n'.join(cleaned_lines)
    
    def remove_noise_patterns(self, text: str) -> str:
        """Remove common noise patterns."""
        if not text:
            return ""
        
        for pattern in self.noise_patterns:
            text = pattern.sub('', text)
        
        return text
    
    def preserve_code_structure(self, text: str) -> str:
        """Preserve and enhance code block structure."""
        if not text:
            return ""
        
        # Enhance existing code blocks
        def enhance_code_block(match):
            code_content = match.group(1)
            # Basic Python detection
            if any(keyword in code_content for keyword in ['def ', 'class ', 'import ', 'from ']):
                return f'```python\n{code_content}\n```'
            return match.group(0)
        
        text = self.code_block_pattern.sub(enhance_code_block, text)
        
        return text
    
    def filter_by_length(self, text: str, min_length: int = 50, max_length: int = 10000) -> bool:
        """Check if text meets length requirements."""
        if not text:
            return False
        
        text_length = len(text.strip())
        return min_length <= text_length <= max_length
    
    def detect_language(self, text: str) -> str:
        """Simple language detection (focused on English)."""
        if not text:
            return "unknown"
        
        # Simple heuristics for English
        english_indicators = [
            'the ', 'and ', 'or ', 'in ', 'on ', 'at ', 'to ', 'for ',
            'a ', 'an ', 'is ', 'are ', 'was ', 'were ', 'be ', 'been '
        ]
        
        text_lower = text.lower()
        english_count = sum(1 for indicator in english_indicators if indicator in text_lower)
        
        if english_count >= 3:  # Arbitrary threshold
            return "english"
        
        # Check if it's primarily code
        code_indicators = ['def ', 'class ', 'import ', 'function', '()', '{', '}', ';']
        code_count = sum(1 for indicator in code_indicators if indicator in text)
        
        if code_count >= 2:
            return "code"
        
        return "unknown"
    
    def clean_text(self, text: str, preserve_code: bool = True) -> Dict[str, any]:
        """Main cleaning function that returns cleaned text with metadata."""
        if not text:
            return {
                'cleaned_text': '',
                'is_valid': False,
                'language': 'unknown',
                'original_length': 0,
                'cleaned_length': 0
            }
        
        original_length = len(text)
        
        # Step 1: Basic HTML cleaning
        cleaned = self.clean_html(text)
        
        # Step 2: Unicode normalization
        cleaned = self.normalize_unicode(cleaned)
        
        # Step 3: Preserve/enhance code structure
        if preserve_code:
            cleaned = self.preserve_code_structure(cleaned)
        
        # Step 4: Clean whitespace
        cleaned = self.clean_whitespace(cleaned)
        
        # Step 5: Remove noise patterns
        cleaned = self.remove_noise_patterns(cleaned)
        
        # Step 6: Final cleanup
        cleaned = cleaned.strip()
        
        # Metadata
        cleaned_length = len(cleaned)
        is_valid = self.filter_by_length(cleaned)
        language = self.detect_language(cleaned)
        
        return {
            'cleaned_text': cleaned,
            'is_valid': is_valid,
            'language': language,
            'original_length': original_length,
            'cleaned_length': cleaned_length,
            'compression_ratio': cleaned_length / original_length if original_length > 0 else 0
        }

def clean_dataset_item(item: Dict, text_fields: List[str] = None) -> Dict:
    """Clean a dataset item with multiple text fields."""
    if text_fields is None:
        text_fields = ['content', 'body', 'description', 'text']
    
    cleaner = TextCleaner()
    cleaned_item = item.copy()
    
    total_original = 0
    total_cleaned = 0
    
    for field in text_fields:
        if field in item and item[field]:
            result = cleaner.clean_text(str(item[field]))
            cleaned_item[field] = result['cleaned_text']
            cleaned_item[f'{field}_metadata'] = {
                'is_valid': result['is_valid'],
                'language': result['language'],
                'compression_ratio': result['compression_ratio']
            }
            
            total_original += result['original_length']
            total_cleaned += result['cleaned_length']
    
    # Overall item metadata
    cleaned_item['cleaning_metadata'] = {
        'total_original_length': total_original,
        'total_cleaned_length': total_cleaned,
        'overall_compression_ratio': total_cleaned / total_original if total_original > 0 else 0
    }
    
    return cleaned_item

if __name__ == "__main__":
    # Test the cleaner
    cleaner = TextCleaner()
    
    test_text = """
    <h1>Python Tutorial</h1>
    <p>This is a simple example of Python code:</p>
    <pre><code>
    def hello_world():
        print("Hello, World!")
        return True
    </code></pre>
    <p>The function above prints a greeting message.</p>
    """
    
    result = cleaner.clean_text(test_text)
    print("Cleaned text:")
    print(result['cleaned_text'])
    print(f"\nMetadata: {result}")