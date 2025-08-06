import ast
import re
import tokenize
import io
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PythonCodeValidator:
    def __init__(self):
        self.python_keywords = {
            'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
            'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
            'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
            'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return',
            'try', 'while', 'with', 'yield'
        }
        
        self.builtin_functions = {
            'abs', 'all', 'any', 'bin', 'bool', 'bytearray', 'bytes', 'callable',
            'chr', 'classmethod', 'compile', 'complex', 'delattr', 'dict', 'dir',
            'divmod', 'enumerate', 'eval', 'exec', 'filter', 'float', 'format',
            'frozenset', 'getattr', 'globals', 'hasattr', 'hash', 'help', 'hex',
            'id', 'input', 'int', 'isinstance', 'issubclass', 'iter', 'len',
            'list', 'locals', 'map', 'max', 'memoryview', 'min', 'next', 'object',
            'oct', 'open', 'ord', 'pow', 'print', 'property', 'range', 'repr',
            'reversed', 'round', 'set', 'setattr', 'slice', 'sorted', 'staticmethod',
            'str', 'sum', 'super', 'tuple', 'type', 'vars', 'zip'
        }
        
        # Common Python library imports
        self.common_imports = {
            'os', 'sys', 'json', 'datetime', 'time', 'random', 'math', 'collections',
            'itertools', 'functools', 're', 'pathlib', 'typing', 'dataclasses',
            'enum', 'logging', 'unittest', 'pytest', 'requests', 'numpy', 'pandas',
            'matplotlib', 'scipy', 'sklearn', 'tensorflow', 'torch', 'django',
            'flask', 'fastapi', 'sqlalchemy', 'pydantic', 'click', 'asyncio'
        }
        
        # Regex patterns for code detection
        self.code_patterns = [
            re.compile(r'^\s*def\s+\w+\s*\(', re.MULTILINE),
            re.compile(r'^\s*class\s+\w+', re.MULTILINE),
            re.compile(r'^\s*(from\s+\w+\s+)?import\s+', re.MULTILINE),
            re.compile(r'^\s*if\s+__name__\s*==\s*[\'"]__main__[\'"]', re.MULTILINE),
            re.compile(r'^\s*@\w+', re.MULTILINE),  # Decorators
            re.compile(r'^\s*with\s+\w+', re.MULTILINE),
            re.compile(r'^\s*for\s+\w+\s+in\s+', re.MULTILINE),
            re.compile(r'^\s*while\s+.+:', re.MULTILINE),
            re.compile(r'^\s*try\s*:', re.MULTILINE),
            re.compile(r'^\s*except\s+\w*', re.MULTILINE),
        ]
    
    def extract_code_blocks(self, text: str) -> List[Dict]:
        """Extract potential Python code blocks from text."""
        code_blocks = []
        
        # Extract markdown code blocks
        markdown_pattern = re.compile(r'```(?:python|py)?\n(.*?)\n```', re.DOTALL | re.IGNORECASE)
        for match in markdown_pattern.finditer(text):
            code_blocks.append({
                'content': match.group(1),
                'type': 'markdown_block',
                'start': match.start(),
                'end': match.end()
            })
        
        # Extract inline code
        inline_pattern = re.compile(r'`([^`\n]+)`')
        for match in inline_pattern.finditer(text):
            content = match.group(1)
            if self.looks_like_python_code(content):
                code_blocks.append({
                    'content': content,
                    'type': 'inline_code',
                    'start': match.start(),
                    'end': match.end()
                })
        
        # Extract indented code blocks (common in documentation)
        lines = text.split('\n')
        in_code_block = False
        current_block = []
        block_start = 0
        
        for i, line in enumerate(lines):
            if line.startswith('    ') or line.startswith('\t'):  # Indented line
                if not in_code_block:
                    in_code_block = True
                    block_start = i
                current_block.append(line.strip())
            else:
                if in_code_block and current_block:
                    block_content = '\n'.join(current_block)
                    if self.looks_like_python_code(block_content):
                        code_blocks.append({
                            'content': block_content,
                            'type': 'indented_block',
                            'start': block_start,
                            'end': i
                        })
                in_code_block = False
                current_block = []
        
        return code_blocks
    
    def looks_like_python_code(self, text: str) -> bool:
        """Heuristic to determine if text looks like Python code."""
        if not text or len(text) < 5:
            return False
        
        text_lower = text.lower().strip()
        
        # Check for Python keywords
        keyword_count = sum(1 for keyword in self.python_keywords 
                          if f' {keyword.lower()} ' in f' {text_lower} ')
        
        # Check for Python patterns
        pattern_count = sum(1 for pattern in self.code_patterns if pattern.search(text))
        
        # Check for common Python characteristics
        characteristics = [
            ':' in text,  # Colons for blocks
            '(' in text and ')' in text,  # Function calls/definitions
            text.count('"') >= 2 or text.count("'") >= 2,  # String literals
            '=' in text,  # Assignments
            'print(' in text_lower or 'print ' in text_lower,  # Print statements
        ]
        
        characteristic_count = sum(characteristics)
        
        # Scoring system
        score = keyword_count * 2 + pattern_count * 3 + characteristic_count
        
        return score >= 3
    
    def validate_python_syntax(self, code: str) -> Dict:
        """Validate Python syntax using AST parsing."""
        if not code or not code.strip():
            return {
                'is_valid': False,
                'error': 'Empty code',
                'error_type': 'empty',
                'line_number': None
            }
        
        try:
            # Try to parse as a complete program
            ast.parse(code)
            return {
                'is_valid': True,
                'error': None,
                'error_type': None,
                'line_number': None
            }
        except SyntaxError as e:
            # Try to parse as an expression
            try:
                ast.parse(code, mode='eval')
                return {
                    'is_valid': True,
                    'error': None,
                    'error_type': None,
                    'line_number': None,
                    'note': 'Valid as expression'
                }
            except SyntaxError:
                pass
            
            # Try to parse individual lines (for code snippets)
            lines = code.split('\n')
            valid_lines = 0
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    ast.parse(line)
                    valid_lines += 1
                except SyntaxError:
                    try:
                        ast.parse(line, mode='eval')
                        valid_lines += 1
                    except SyntaxError:
                        pass
            
            # If most lines are valid, consider it acceptable
            total_code_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
            if total_code_lines > 0 and valid_lines / total_code_lines >= 0.7:
                return {
                    'is_valid': True,
                    'error': f'Partial syntax error: {str(e)}',
                    'error_type': 'partial',
                    'line_number': e.lineno,
                    'note': f'{valid_lines}/{total_code_lines} lines valid'
                }
            
            return {
                'is_valid': False,
                'error': str(e),
                'error_type': 'syntax',
                'line_number': e.lineno
            }
        
        except Exception as e:
            return {
                'is_valid': False,
                'error': str(e),
                'error_type': 'other',
                'line_number': None
            }
    
    def analyze_code_quality(self, code: str) -> Dict:
        """Analyze code quality and extract features."""
        if not code:
            return {'quality_score': 0, 'features': {}}
        
        features = {
            'has_functions': bool(re.search(r'^\s*def\s+', code, re.MULTILINE)),
            'has_classes': bool(re.search(r'^\s*class\s+', code, re.MULTILINE)),
            'has_imports': bool(re.search(r'^\s*(from\s+\w+\s+)?import\s+', code, re.MULTILINE)),
            'has_docstrings': bool(re.search(r'""".*?"""', code, re.DOTALL)),
            'has_comments': bool(re.search(r'#.*$', code, re.MULTILINE)),
            'has_type_hints': bool(re.search(r':\s*\w+', code)),
            'has_main_guard': bool(re.search(r'if\s+__name__\s*==\s*[\'"]__main__[\'"]', code)),
            'line_count': len([l for l in code.split('\n') if l.strip()]),
            'avg_line_length': sum(len(l) for l in code.split('\n')) / max(len(code.split('\n')), 1)
        }
        
        # Calculate quality score
        score = 0
        if features['has_functions']: score += 20
        if features['has_classes']: score += 15
        if features['has_imports']: score += 10
        if features['has_docstrings']: score += 15
        if features['has_comments']: score += 10
        if features['has_type_hints']: score += 10
        if features['has_main_guard']: score += 5
        if 5 <= features['line_count'] <= 100: score += 10  # Reasonable length
        if 20 <= features['avg_line_length'] <= 80: score += 5  # Reasonable line length
        
        return {
            'quality_score': score,
            'features': features
        }
    
    def validate_and_clean_code(self, text: str) -> Dict:
        """Main function to validate and clean code from text."""
        result = {
            'original_text': text,
            'code_blocks': [],
            'valid_code_blocks': [],
            'total_code_lines': 0,
            'valid_code_lines': 0
        }
        
        # Extract code blocks
        code_blocks = self.extract_code_blocks(text)
        result['code_blocks'] = code_blocks
        
        # Validate each code block
        for block in code_blocks:
            validation = self.validate_python_syntax(block['content'])
            quality = self.analyze_code_quality(block['content'])
            
            block_result = {
                **block,
                'validation': validation,
                'quality': quality
            }
            
            result['code_blocks'].append(block_result)
            result['total_code_lines'] += block_result['quality']['features']['line_count']
            
            if validation['is_valid']:
                result['valid_code_blocks'].append(block_result)
                result['valid_code_lines'] += block_result['quality']['features']['line_count']
        
        # Calculate overall statistics
        result['code_validity_ratio'] = (
            result['valid_code_lines'] / result['total_code_lines'] 
            if result['total_code_lines'] > 0 else 0
        )
        
        return result

def validate_dataset_code(item: Dict, text_fields: List[str] = None) -> Dict:
    """Validate code in a dataset item."""
    if text_fields is None:
        text_fields = ['content', 'body', 'description']
    
    validator = PythonCodeValidator()
    validated_item = item.copy()
    
    total_code_blocks = 0
    total_valid_blocks = 0
    
    for field in text_fields:
        if field in item and item[field]:
            validation_result = validator.validate_and_clean_code(str(item[field]))
            
            validated_item[f'{field}_code_analysis'] = {
                'code_blocks_found': len(validation_result['code_blocks']),
                'valid_code_blocks': len(validation_result['valid_code_blocks']),
                'code_validity_ratio': validation_result['code_validity_ratio'],
                'total_code_lines': validation_result['total_code_lines']
            }
            
            total_code_blocks += len(validation_result['code_blocks'])
            total_valid_blocks += len(validation_result['valid_code_blocks'])
    
    # Overall code analysis
    validated_item['overall_code_analysis'] = {
        'total_code_blocks': total_code_blocks,
        'total_valid_blocks': total_valid_blocks,
        'overall_validity_ratio': total_valid_blocks / total_code_blocks if total_code_blocks > 0 else 0
    }
    
    return validated_item

if __name__ == "__main__":
    # Test the validator
    validator = PythonCodeValidator()
    
    test_text = """
    Here's a simple Python example:
    
    ```python
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    
    # Calculate first 10 numbers
    for i in range(10):
        print(f"F({i}) = {fibonacci(i)}")
    ```
    
    You can also use inline code like `print("hello")` in your text.
    """
    
    result = validator.validate_and_clean_code(test_text)
    print(f"Found {len(result['code_blocks'])} code blocks")
    print(f"Valid blocks: {len(result['valid_code_blocks'])}")
    print(f"Validity ratio: {result['code_validity_ratio']:.2f}")
    
    for block in result['valid_code_blocks']:
        print(f"\nBlock type: {block['type']}")
        print(f"Quality score: {block['quality']['quality_score']}")
        print("Content:")
        print(block['content'][:200] + "..." if len(block['content']) > 200 else block['content'])