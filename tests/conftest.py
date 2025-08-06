"""
Pytest configuration and shared fixtures for LimeLLM tests.
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

from model.config import ModelConfig, TrainingConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_model_config():
    """Create a sample model configuration for testing."""
    return ModelConfig(
        vocab_size=1000,
        n_positions=512,
        n_embd=256,
        n_layer=4,
        n_head=4,
        gradient_checkpointing=False
    )


@pytest.fixture
def sample_training_config():
    """Create a sample training configuration for testing."""
    return TrainingConfig(
        learning_rate=1e-4,
        batch_size=8,
        max_steps=100,
        warmup_steps=10,
        save_steps=50,
        eval_steps=25
    )


@pytest.fixture
def sample_config_file(temp_dir, sample_model_config):
    """Create a sample configuration file."""
    config_file = temp_dir / "test_config.json"
    sample_model_config.save_json(str(config_file))
    return config_file


@pytest.fixture
def sample_python_code():
    """Sample Python code for testing."""
    return '''
def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

# Example usage
calc = Calculator()
result = calc.add(5, 3)
print(f"Result: {result}")

# Calculate Fibonacci numbers
for i in range(10):
    fib = fibonacci(i)
    print(f"F({i}) = {fib}")
'''


@pytest.fixture
def sample_text_data():
    """Sample text data for testing."""
    return '''
<html>
<head><title>Python Tutorial</title></head>
<body>
<h1>Learn Python Programming</h1>

<p>Python is a high-level programming language. 
Contact us at info@example.com or visit https://python.org</p>

<div class="code-section">
<h2>Example Code</h2>

```python
def greet(name):
    print(f"Hello, {name}!")
    return True
```

You can use `print()` to display output.

</div>

# ----------------
# Additional Notes
# ----------------

More content here with    excessive    whitespace.



Multiple empty lines above.

</body>
</html>
'''


@pytest.fixture
def mock_http_response():
    """Mock HTTP response for testing web scrapers."""
    response = MagicMock()
    response.status_code = 200
    response.text = "<html><body><h1>Test Page</h1></body></html>"
    response.json.return_value = {"data": "test"}
    return response


@pytest.fixture
def sample_data_files(temp_dir):
    """Create sample data files for testing."""
    data_dir = temp_dir / "data"
    data_dir.mkdir()
    
    # Create sample JSON lines files
    train_file = data_dir / "train.jsonl"
    eval_file = data_dir / "eval.jsonl"
    
    train_data = [
        {"text": "def hello(): print('Hello, World!')", "type": "code"},
        {"text": "Python is a programming language", "type": "text"},
        {"text": "import numpy as np", "type": "code"},
    ]
    
    eval_data = [
        {"text": "def add(a, b): return a + b", "type": "code"},
        {"text": "Machine learning with Python", "type": "text"},
    ]
    
    with open(train_file, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\\n')
    
    with open(eval_file, 'w') as f:
        for item in eval_data:
            f.write(json.dumps(item) + '\\n')
    
    return {
        "train": train_file,
        "eval": eval_file,
        "dir": data_dir
    }


@pytest.fixture
def mock_github_api():
    """Mock GitHub API responses."""
    return {
        "search_repositories": {
            "total_count": 100,
            "items": [
                {
                    "name": "awesome-python",
                    "full_name": "user/awesome-python",
                    "description": "A curated list of Python frameworks",
                    "stargazers_count": 1000,
                    "language": "Python",
                    "html_url": "https://github.com/user/awesome-python"
                }
            ]
        },
        "repository_content": {
            "name": "main.py",
            "content": "ZGVmIG1haW4oKTpccHJpbnQoJ0hlbGxvLCBXb3JsZCEnKQ=="  # base64 encoded
        }
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    # Set test environment variables
    monkeypatch.setenv("TESTING", "1")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    
    # Mock sensitive environment variables
    monkeypatch.setenv("GITHUB_TOKEN", "test_token_123")
    monkeypatch.setenv("WANDB_API_KEY", "test_wandb_key")


# Custom markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "network: mark test as requiring network access"
    )


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file location."""
    for item in items:
        # Add unit marker for tests in unit directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker for tests in integration directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker for tests that might be slow
        if "training" in str(item.fspath) or "model" in str(item.fspath):
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session")
def test_data_dir():
    """Get the test data directory."""
    return Path(__file__).parent / "fixtures"