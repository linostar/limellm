"""
Integration tests for the complete data processing pipeline.
"""

import pytest
import json
import tempfile
from pathlib import Path

from preprocessing.text_cleaner import TextCleaner
from data_collection.rate_limiter import RateLimiter


@pytest.mark.integration
class TestDataPipeline:
    """Integration tests for data processing pipeline."""
    
    def test_text_processing_pipeline(self, sample_text_data):
        """Test complete text processing pipeline."""
        cleaner = TextCleaner()
        
        # Process the sample text
        cleaned = cleaner.clean_text(sample_text_data)
        
        # Verify expected transformations
        assert cleaned is not None
        assert len(cleaned) > 0
        
        # Should remove HTML tags
        assert "<html>" not in cleaned
        assert "<div>" not in cleaned
        
        # Should preserve code blocks
        assert "def greet(name):" in cleaned
        
        # Should normalize whitespace
        lines = cleaned.split('\\n')
        # Should not have excessive empty lines
        empty_line_count = sum(1 for line in lines if line.strip() == '')
        total_lines = len(lines)
        assert empty_line_count / total_lines < 0.5  # Less than 50% empty lines
    
    def test_config_integration(self, temp_dir):
        """Test configuration loading and saving integration."""
        from model.config import ModelConfig, TrainingConfig
        
        # Create configs
        model_config = ModelConfig(vocab_size=1000, n_embd=128)
        training_config = TrainingConfig(batch_size=16, learning_rate=2e-4)
        
        # Save configs
        model_config_path = temp_dir / "model.json"
        training_config_path = temp_dir / "training.json"
        
        model_config.save_json(str(model_config_path))
        training_config.save_json(str(training_config_path))
        
        # Load configs back
        loaded_model = ModelConfig.from_json(str(model_config_path))
        loaded_training = TrainingConfig.from_json(str(training_config_path))
        
        # Verify integrity
        assert loaded_model.vocab_size == 1000
        assert loaded_model.n_embd == 128
        assert loaded_training.batch_size == 16
        assert loaded_training.learning_rate == 2e-4
    
    def test_rate_limiter_with_data_collection_simulation(self):
        """Test rate limiter in a simulated data collection scenario."""
        import time
        
        limiter = RateLimiter(requests_per_second=10.0, burst_size=3)
        
        # Simulate collecting data with rate limiting
        collection_times = []
        
        for i in range(8):  # More than burst size
            start_time = time.time()
            limiter.acquire()
            
            # Simulate API call processing time
            time.sleep(0.01)
            
            end_time = time.time()
            collection_times.append(end_time - start_time)
        
        # First few should be fast (burst), later ones should be rate limited
        assert collection_times[0] < 0.02  # First request fast
        assert collection_times[1] < 0.02  # Second request fast
        assert collection_times[2] < 0.02  # Third request fast
        
        # Later requests should show rate limiting effects
        avg_later_time = sum(collection_times[3:]) / len(collection_times[3:])
        assert avg_later_time > 0.05  # Should take longer due to rate limiting
    
    @pytest.mark.slow
    def test_large_text_processing(self):
        """Test processing of large text documents."""
        cleaner = TextCleaner()
        
        # Create a large text document
        large_text = """
        # Python Programming Tutorial
        
        """ + "\\n".join([f"This is line {i} of the document." for i in range(1000)]) + """
        
        ```python
        def process_large_data(data):
            results = []
            for item in data:
                if len(item) > 10:
                    results.append(item.upper())
            return results
        
        # Process data
        large_dataset = [f"item_{i}" for i in range(1000)]
        processed = process_large_data(large_dataset)
        ```
        
        And more content...
        """
        
        # Should handle large text without errors
        cleaned = cleaner.clean_text(large_text)
        
        assert cleaned is not None
        assert len(cleaned) > 0
        assert "def process_large_data(data):" in cleaned
        assert "This is line 500 of the document." in cleaned
    
    def test_end_to_end_config_and_processing(self, temp_dir, sample_python_code):
        """Test end-to-end configuration and text processing."""
        from model.config import ModelConfig
        
        # Create configuration
        config = ModelConfig(
            vocab_size=5000,
            n_embd=256,
            n_layer=6,
            n_head=8
        )
        
        # Save configuration
        config_path = temp_dir / "model_config.json"
        config.save_json(str(config_path))
        
        # Process sample code
        cleaner = TextCleaner()
        processed_code = cleaner.clean_text(sample_python_code)
        
        # Create processed data file
        data_file = temp_dir / "processed_data.jsonl"
        with open(data_file, 'w') as f:
            f.write(json.dumps({
                "text": processed_code,
                "type": "code",
                "config_used": str(config_path)
            }) + '\\n')
        
        # Verify everything is accessible
        loaded_config = ModelConfig.from_json(str(config_path))
        assert loaded_config.vocab_size == 5000
        
        with open(data_file, 'r') as f:
            data = json.loads(f.readline())
            assert "def fibonacci(n):" in data["text"]
            assert data["type"] == "code"
    
    def test_error_handling_in_pipeline(self):
        """Test error handling throughout the pipeline."""
        cleaner = TextCleaner()
        
        # Test with various problematic inputs
        test_cases = [
            None,
            "",
            "   \\n\\n   ",
            "Valid text",
            "<invalid><<>><html>",
            "🎉 Unicode emoji test 🐍",
            "Very long text " + "x" * 10000
        ]
        
        for test_input in test_cases:
            try:
                result = cleaner.clean_text(test_input)
                # Should either return a string or None, never crash
                assert result is None or isinstance(result, str)
            except Exception as e:
                pytest.fail(f"Unexpected exception for input '{test_input}': {e}")
    
    @pytest.mark.network
    def test_mock_network_integration(self, mock_http_response):
        """Test integration with mocked network components."""
        # This would test actual network scrapers in a real scenario
        # For now, just test that our mocks work correctly
        
        assert mock_http_response.status_code == 200
        assert "<h1>Test Page</h1>" in mock_http_response.text
        assert mock_http_response.json()["data"] == "test"