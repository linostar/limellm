import pytest
import json
import tempfile
import os
from pathlib import Path

from model.config import ModelConfig, TrainingConfig, get_model_config


class TestModelConfig:
    """Test suite for ModelConfig class."""
    
    def test_default_config(self):
        """Test that default config is created with expected values."""
        config = ModelConfig()
        
        assert config.vocab_size == 50304
        assert config.n_positions == 4096
        assert config.n_embd == 1536
        assert config.n_layer == 24
        assert config.n_head == 12
        assert config.activation_function == "gelu_new"
        assert config.gradient_checkpointing is True
    
    def test_custom_config(self):
        """Test creating config with custom parameters."""
        config = ModelConfig(
            vocab_size=32000,
            n_embd=768,
            n_layer=12,
            n_head=6
        )
        
        assert config.vocab_size == 32000
        assert config.n_embd == 768
        assert config.n_layer == 12
        assert config.n_head == 6
    
    def test_code_token_types_initialization(self):
        """Test that code token types are properly initialized."""
        config = ModelConfig()
        
        expected_types = {
            "function_def": 0,
            "class_def": 1,
            "import": 2,
            "comment": 3,
            "string": 4,
            "number": 5,
            "keyword": 6,
            "identifier": 7,
            "operator": 8,
            "punctuation": 9,
        }
        
        for key, value in expected_types.items():
            assert config.code_token_types[key] == value
    
    def test_parameter_calculation(self):
        """Test that parameter count is calculated correctly."""
        config = ModelConfig()
        param_count = config.total_parameters
        
        # Basic sanity check - should be in billions for default config
        assert param_count > 1_000_000_000
        assert param_count < 10_000_000_000
    
    def test_json_serialization(self):
        """Test saving and loading config to/from JSON."""
        config = ModelConfig(vocab_size=32000, n_embd=768)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.save_json(f.name)
            
            # Load back and compare
            loaded_config = ModelConfig.from_json(f.name)
            
            assert loaded_config.vocab_size == 32000
            assert loaded_config.n_embd == 768
            assert loaded_config.n_layer == config.n_layer  # Default value
        
        os.unlink(f.name)
    
    def test_yaml_serialization(self):
        """Test saving and loading config to/from YAML."""
        config = ModelConfig(vocab_size=32000, n_embd=768)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.save_yaml(f.name)
            
            # Load back and compare
            loaded_config = ModelConfig.from_yaml(f.name)
            
            assert loaded_config.vocab_size == 32000
            assert loaded_config.n_embd == 768
            assert loaded_config.n_layer == config.n_layer
        
        os.unlink(f.name)
    
    def test_validation(self):
        """Test config validation."""
        # Test valid config
        config = ModelConfig()
        assert config.validate() is True
        
        # Test invalid configs
        with pytest.raises(ValueError):
            ModelConfig(vocab_size=0)  # Should fail
        
        with pytest.raises(ValueError):
            ModelConfig(n_embd=0)  # Should fail
        
        with pytest.raises(ValueError):
            ModelConfig(n_head=0)  # Should fail


class TestTrainingConfig:
    """Test suite for TrainingConfig class."""
    
    def test_default_training_config(self):
        """Test default training configuration."""
        config = TrainingConfig()
        
        assert config.learning_rate > 0
        assert config.batch_size > 0
        assert config.max_steps > 0
        assert config.warmup_steps >= 0
        assert isinstance(config.fp16, bool)
    
    def test_custom_training_config(self):
        """Test custom training configuration."""
        config = TrainingConfig(
            learning_rate=2e-4,
            batch_size=64,
            max_steps=50000
        )
        
        assert config.learning_rate == 2e-4
        assert config.batch_size == 64
        assert config.max_steps == 50000
    
    def test_training_config_serialization(self):
        """Test training config serialization."""
        config = TrainingConfig(learning_rate=1e-5, batch_size=16)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.save_json(f.name)
            
            loaded_config = TrainingConfig.from_json(f.name)
            
            assert loaded_config.learning_rate == 1e-5
            assert loaded_config.batch_size == 16
        
        os.unlink(f.name)


class TestModelConfigPresets:
    """Test suite for predefined model configurations."""
    
    def test_get_model_config_2b(self):
        """Test getting 2B model configuration."""
        config = get_model_config("limellm-2b")
        
        assert config.n_embd == 1536
        assert config.n_layer == 24
        assert config.n_head == 12
    
    def test_get_model_config_1b(self):
        """Test getting 1B model configuration."""
        config = get_model_config("limellm-1b")
        
        assert config.n_embd < 1536  # Should be smaller than 2B
        assert config.n_layer < 24   # Should have fewer layers
    
    def test_get_model_config_500m(self):
        """Test getting 500M model configuration."""
        config = get_model_config("limellm-500m")
        
        assert config.n_embd < 1024  # Should be smaller
        assert config.n_layer < 20   # Should have fewer layers
    
    def test_invalid_model_config(self):
        """Test error handling for invalid model config."""
        with pytest.raises(ValueError):
            get_model_config("invalid-model")
    
    def test_parameter_scaling(self):
        """Test that parameter counts scale appropriately."""
        config_500m = get_model_config("limellm-500m")
        config_1b = get_model_config("limellm-1b")
        config_2b = get_model_config("limellm-2b")
        
        # Parameter counts should increase
        assert config_500m.total_parameters < config_1b.total_parameters
        assert config_1b.total_parameters < config_2b.total_parameters
        
        # Rough sanity checks for parameter counts
        assert 400_000_000 < config_500m.total_parameters < 600_000_000
        assert 800_000_000 < config_1b.total_parameters < 1_200_000_000
        assert 1_800_000_000 < config_2b.total_parameters < 2_200_000_000