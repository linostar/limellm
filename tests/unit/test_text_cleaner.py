import pytest
from preprocessing.text_cleaner import TextCleaner


class TestTextCleaner:
    """Test suite for TextCleaner class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cleaner = TextCleaner()
    
    def test_initialization(self):
        """Test that TextCleaner initializes correctly."""
        assert self.cleaner is not None
        assert hasattr(self.cleaner, 'url_pattern')
        assert hasattr(self.cleaner, 'email_pattern')
        assert hasattr(self.cleaner, 'excessive_whitespace')
        assert len(self.cleaner.code_indicators) > 0
    
    def test_url_removal(self):
        """Test URL removal functionality."""
        text = "Check out https://example.com and http://test.org for more info."
        cleaned = self.cleaner.remove_urls(text)
        
        assert "https://example.com" not in cleaned
        assert "http://test.org" not in cleaned
        assert "Check out" in cleaned
        assert "for more info" in cleaned
    
    def test_email_removal(self):
        """Test email removal functionality."""
        text = "Contact us at info@example.com or support@test.org"
        cleaned = self.cleaner.remove_emails(text)
        
        assert "info@example.com" not in cleaned
        assert "support@test.org" not in cleaned
        assert "Contact us at" in cleaned
    
    def test_html_cleaning(self):
        """Test HTML cleaning functionality."""
        html_text = '<div class="content">Hello <b>world</b>!</div>'
        cleaned = self.cleaner.clean_html(html_text)
        
        assert "<div" not in cleaned
        assert "<b>" not in cleaned
        assert "Hello world!" in cleaned
    
    def test_excessive_whitespace_cleanup(self):
        """Test excessive whitespace cleanup."""
        text = "Hello    world\\n\\n\\n\\nThis   has     too   much   whitespace"
        cleaned = self.cleaner.normalize_whitespace(text)
        
        # Should have single spaces between words
        assert "Hello world" in cleaned
        assert "This has too much whitespace" in cleaned
        # Should not have excessive newlines or spaces
        assert "    " not in cleaned
        assert "\\n\\n\\n" not in cleaned
    
    def test_code_block_preservation(self):
        """Test that code blocks are preserved correctly."""
        text = '''
        Here's some Python code:
        ```python
        def hello():
            print("Hello, world!")
            return True
        ```
        And that's it.
        '''
        
        cleaned = self.cleaner.clean_text(text)
        
        # Code should be preserved
        assert "def hello():" in cleaned
        assert "print(\\"Hello, world!\\")" in cleaned
        assert "return True" in cleaned
    
    def test_inline_code_preservation(self):
        """Test that inline code is preserved."""
        text = "Use `print()` function to display output and `len()` for length."
        cleaned = self.cleaner.clean_text(text)
        
        assert "print()" in cleaned
        assert "len()" in cleaned
    
    def test_noise_pattern_removal(self):
        """Test removal of noise patterns."""
        text = '''
        # ----------------
        
        Real content here
        
        ****************
        
        More content
        
        ================
        '''
        
        cleaned = self.cleaner.remove_noise_patterns(text)
        
        assert "----------------" not in cleaned
        assert "****************" not in cleaned
        assert "================" not in cleaned
        assert "Real content here" in cleaned
        assert "More content" in cleaned
    
    def test_python_code_detection(self):
        """Test Python code detection."""
        python_code = '''
        def factorial(n):
            if n <= 1:
                return 1
            return n * factorial(n - 1)
        
        class Calculator:
            def __init__(self):
                self.history = []
        
        import math
        from collections import defaultdict
        '''
        
        is_code = self.cleaner.is_likely_code(python_code)
        assert is_code is True
        
        regular_text = "This is just regular text about Python programming."
        is_code = self.cleaner.is_likely_code(regular_text)
        assert is_code is False
    
    def test_comprehensive_cleaning(self):
        """Test comprehensive text cleaning pipeline."""
        messy_text = '''
        <h1>Python Tutorial</h1>
        
        Contact: admin@example.com
        Visit: https://python.org
        
        # ----------------
        
        ```python
        def greet(name):
            print(f"Hello, {name}!")
        ```
        
        Use `print()` for output.    Extra    spaces    everywhere.
        
        
        
        Multiple blank lines above.
        '''
        
        cleaned = self.cleaner.clean_text(messy_text)
        
        # Should remove HTML
        assert "<h1>" not in cleaned
        # Should remove email and URL
        assert "admin@example.com" not in cleaned
        assert "https://python.org" not in cleaned
        # Should preserve code
        assert "def greet(name):" in cleaned
        assert "print(f\\"Hello, {name}!\\")" in cleaned
        # Should normalize whitespace
        assert "    " not in cleaned
        # Should contain main content
        assert "Python Tutorial" in cleaned
        assert "Use print() for output" in cleaned
    
    def test_empty_text_handling(self):
        """Test handling of empty or whitespace-only text."""
        assert self.cleaner.clean_text("") == ""
        assert self.cleaner.clean_text("   \\n\\n  ").strip() == ""
        assert self.cleaner.clean_text(None) is None
    
    def test_unicode_handling(self):
        """Test handling of Unicode characters."""
        unicode_text = "Python supports émojis 🐍 and ünïcödë characters"
        cleaned = self.cleaner.clean_text(unicode_text)
        
        # Should preserve Unicode content
        assert "émojis" in cleaned
        assert "🐍" in cleaned
        assert "ünïcödë" in cleaned
    
    def test_code_comment_preservation(self):
        """Test that code comments are preserved."""
        code_with_comments = '''
        # This is a Python function
        def calculate_sum(a, b):
            """Calculate sum of two numbers."""
            return a + b  # Return the result
        '''
        
        cleaned = self.cleaner.clean_text(code_with_comments)
        
        # Comments should be preserved in code context
        assert "# This is a Python function" in cleaned
        assert "Return the result" in cleaned
        assert '"""Calculate sum of two numbers."""' in cleaned
    
    def test_special_characters_in_code(self):
        """Test handling of special characters in code."""
        code_text = '''
        regex_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
        symbols = ["@", "#", "$", "%", "&", "*"]
        equation = "E = mc²"
        '''
        
        cleaned = self.cleaner.clean_text(code_text)
        
        # Special characters in code should be preserved
        assert "regex_pattern" in cleaned
        assert "@" in cleaned
        assert "mc²" in cleaned