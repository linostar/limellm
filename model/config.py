from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import yaml

@dataclass
class ModelConfig:
    # Model architecture
    vocab_size: int = 50304  # GPT-2 vocab size, rounded to nearest multiple of 64
    n_positions: int = 4096  # Maximum sequence length
    n_embd: int = 1536       # Embedding dimension (~2B parameters)
    n_layer: int = 24        # Number of transformer layers
    n_head: int = 12         # Number of attention heads
    
    # Model behavior
    use_cache: bool = True
    pad_token_id: int = 50256
    eos_token_id: int = 50256
    bos_token_id: int = 50256
    
    # Regularization
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    
    # Activation functions
    activation_function: str = "gelu_new"
    
    # Training specific
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    
    # Code-specific features
    code_token_types: Dict[str, int] = None
    special_tokens: Dict[str, str] = None
    
    def __post_init__(self):
        if self.code_token_types is None:
            self.code_token_types = {
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
                "newline": 10,
                "indent": 11,
                "dedent": 12,
            }
        
        if self.special_tokens is None:
            self.special_tokens = {
                "pad": "<|pad|>",
                "eos": "<|endoftext|>",
                "bos": "<|startoftext|>",
                "code_start": "<|code|>",
                "code_end": "<|/code|>",
                "docstring_start": "<|docstring|>",
                "docstring_end": "<|/docstring|>",
                "comment_start": "<|comment|>",
                "comment_end": "<|/comment|>",
                "function_start": "<|function|>",
                "function_end": "<|/function|>",
                "class_start": "<|class|>",
                "class_end": "<|/class|>",
                "python_start": "<|python|>",
                "python_end": "<|/python|>",
            }
    
    @property
    def total_parameters(self) -> int:
        """Estimate total number of parameters."""
        # Token embeddings
        token_emb = self.vocab_size * self.n_embd
        
        # Position embeddings
        pos_emb = self.n_positions * self.n_embd
        
        # Transformer layers
        # Each layer has:
        # - Multi-head attention: 4 * n_embd^2 (Q, K, V, output projections)
        # - Layer norms: 2 * n_embd (pre-attention, pre-mlp)
        # - MLP: 2 * n_embd * (4 * n_embd) (up and down projections, 4x expansion)
        per_layer = (4 * self.n_embd * self.n_embd) + (2 * self.n_embd) + (2 * self.n_embd * 4 * self.n_embd)
        transformer_layers = self.n_layer * per_layer
        
        # Final layer norm
        final_ln = self.n_embd
        
        # LM head (typically tied to token embeddings, but counted separately)
        lm_head = self.vocab_size * self.n_embd
        
        total = token_emb + pos_emb + transformer_layers + final_ln + lm_head
        return total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'n_positions': self.n_positions,
            'n_embd': self.n_embd,
            'n_layer': self.n_layer,
            'n_head': self.n_head,
            'use_cache': self.use_cache,
            'pad_token_id': self.pad_token_id,
            'eos_token_id': self.eos_token_id,
            'bos_token_id': self.bos_token_id,
            'embd_pdrop': self.embd_pdrop,
            'resid_pdrop': self.resid_pdrop,
            'attn_pdrop': self.attn_pdrop,
            'layer_norm_epsilon': self.layer_norm_epsilon,
            'activation_function': self.activation_function,
            'gradient_checkpointing': self.gradient_checkpointing,
            'use_flash_attention': self.use_flash_attention,
            'code_token_types': self.code_token_types,
            'special_tokens': self.special_tokens,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save_json(self, filepath: str):
        """Save config to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def save_yaml(self, filepath: str):
        """Save config to YAML file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'ModelConfig':
        """Load config from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'ModelConfig':
        """Load config from YAML file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

@dataclass
class TrainingConfig:
    # Training hyperparameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    
    # Learning rate schedule
    warmup_steps: int = 2000
    lr_decay_steps: int = 100000
    min_lr_ratio: float = 0.1
    
    # Training dynamics
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Training length
    max_steps: int = 100000
    save_steps: int = 5000
    eval_steps: int = 1000
    logging_steps: int = 100
    
    # Data
    max_length: int = 4096
    train_data_path: str = "data/train"
    eval_data_path: str = "data/eval"
    
    # Distributed training
    use_ddp: bool = True
    use_deepspeed: bool = True
    deepspeed_config: str = "configs/deepspeed_config.json"
    
    # Mixed precision
    fp16: bool = True
    bf16: bool = False  # Use if available on newer GPUs
    
    # Checkpointing
    output_dir: str = "outputs"
    resume_from_checkpoint: Optional[str] = None
    
    # Logging
    wandb_project: str = "limellm"
    wandb_run_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            'warmup_steps': self.warmup_steps,
            'lr_decay_steps': self.lr_decay_steps,
            'min_lr_ratio': self.min_lr_ratio,
            'batch_size': self.batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'max_grad_norm': self.max_grad_norm,
            'max_steps': self.max_steps,
            'save_steps': self.save_steps,
            'eval_steps': self.eval_steps,
            'logging_steps': self.logging_steps,
            'max_length': self.max_length,
            'train_data_path': self.train_data_path,
            'eval_data_path': self.eval_data_path,
            'use_ddp': self.use_ddp,
            'use_deepspeed': self.use_deepspeed,
            'deepspeed_config': self.deepspeed_config,
            'fp16': self.fp16,
            'bf16': self.bf16,
            'output_dir': self.output_dir,
            'resume_from_checkpoint': self.resume_from_checkpoint,
            'wandb_project': self.wandb_project,
            'wandb_run_name': self.wandb_run_name,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save_json(self, filepath: str):
        """Save config to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'TrainingConfig':
        """Load config from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

# Predefined configurations for different model sizes
MODEL_CONFIGS = {
    "limellm-2b": ModelConfig(
        vocab_size=50304,
        n_positions=4096,
        n_embd=1536,
        n_layer=24,
        n_head=12,
    ),
    "limellm-1b": ModelConfig(
        vocab_size=50304,
        n_positions=4096,
        n_embd=1280,
        n_layer=20,
        n_head=10,
    ),
    "limellm-500m": ModelConfig(
        vocab_size=50304,
        n_positions=2048,
        n_embd=1024,
        n_layer=16,
        n_head=8,
    ),
}

def get_model_config(model_name: str) -> ModelConfig:
    """Get a predefined model configuration."""
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    else:
        available = list(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model config: {model_name}. Available: {available}")

if __name__ == "__main__":
    # Test configurations
    config = get_model_config("limellm-2b")
    print(f"LimeLLM-2B parameters: {config.total_parameters:,}")
    print(f"Embedding dimension: {config.n_embd}")
    print(f"Layers: {config.n_layer}")
    print(f"Attention heads: {config.n_head}")
    print(f"Context length: {config.n_positions}")
    
    # Test saving/loading
    config.save_json("test_config.json")
    loaded_config = ModelConfig.from_json("test_config.json")
    print(f"Config loaded successfully: {loaded_config.n_embd}")
    
    import os
    os.remove("test_config.json")