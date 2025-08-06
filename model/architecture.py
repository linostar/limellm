import torch
import torch.nn as nn
from torch.nn import LayerNorm
from typing import Optional, Tuple, Dict, Any
import math

from .config import ModelConfig
from .layers import TransformerBlock, CodeAwareEmbedding

class LimeLLMModel(nn.Module):
    """LimeLLM: A 2B parameter language model specialized for Python code."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings (with code-aware features)
        self.embeddings = CodeAwareEmbedding(config)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final layer norm
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Language modeling head (tied to input embeddings for parameter efficiency)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights between input embeddings and output projection
        self.tie_weights()
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Enable gradient checkpointing if specified
        if config.gradient_checkpointing:
            self.gradient_checkpointing_enable()
    
    def tie_weights(self):
        """Tie input and output embeddings."""
        self.lm_head.weight = self.embeddings.token_embeddings.weight
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            # Slightly different initialization for different layers
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        elif isinstance(module, LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def get_num_params(self, non_embedding: bool = False) -> int:
        """Get number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embeddings.token_embeddings.weight.numel()
        return n_params
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        code_type_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None
    ) -> Dict[str, Any]:
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else True
        
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        hidden_states = self.embeddings(input_ids, code_type_ids=code_type_ids)
        
        # Prepare attention mask
        if attention_mask is not None:
            # Convert attention mask to the format expected by attention layers
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        
        # Initialize past key values if not provided
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        
        # Forward through transformer layers
        present_key_values = []
        for i, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            
            if self.config.gradient_checkpointing and self.training:
                # Use gradient checkpointing during training
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    use_cache,
                    past_key_value,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    past_key_value=past_key_value,
                )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                present_key_values.append(layer_outputs[1])
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Calculate language modeling loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        if not return_dict:
            output = (logits,)
            if use_cache:
                output = output + (present_key_values,)
            return ((loss,) + output) if loss is not None else output
        
        return {
            'loss': loss,
            'logits': logits,
            'past_key_values': present_key_values if use_cache else None,
            'hidden_states': hidden_states,
        }
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        code_type_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare inputs for generation."""
        
        # If we have past key values, only use the last token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            if code_type_ids is not None:
                code_type_ids = code_type_ids[:, -1:]
        
        # Extend attention mask
        if attention_mask is not None and past_key_values is not None:
            attention_mask = torch.cat([
                attention_mask,
                attention_mask.new_ones((attention_mask.shape[0], 1))
            ], dim=-1)
        
        return {
            'input_ids': input_ids,
            'past_key_values': past_key_values,
            'attention_mask': attention_mask,
            'code_type_ids': code_type_ids,
            'use_cache': True,
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate text using the model."""
        
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        
        # Initialize
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            next_token_logits = outputs['logits'][:, -1, :]
            past_key_values = outputs['past_key_values']
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Update input_ids
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    attention_mask.new_ones((attention_mask.shape[0], 1))
                ], dim=-1)
            
            # Check for finished sequences
            finished = finished | (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break
        
        return input_ids

class LimeLLMForCausalLM(nn.Module):
    """LimeLLM wrapper for causal language modeling with HuggingFace compatibility."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = LimeLLMModel(config)
    
    def forward(self, *args, **kwargs):
        """Forward pass (delegates to the main model)."""
        return self.model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """Generation (delegates to the main model)."""
        return self.model.generate(*args, **kwargs)
    
    def get_num_params(self):
        """Get number of parameters."""
        return self.model.get_num_params()
    
    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state dict
        torch.save(self.state_dict(), os.path.join(save_directory, 'pytorch_model.bin'))
        
        # Save config
        config_dict = self.config.to_dict()
        with open(os.path.join(save_directory, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, model_path: str):
        """Load model from directory."""
        import os
        import json
        
        # Load config
        config_path = os.path.join(model_path, 'config.json')
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = ModelConfig.from_dict(config_dict)
        
        # Create model
        model = cls(config)
        
        # Load state dict
        state_dict_path = os.path.join(model_path, 'pytorch_model.bin')
        state_dict = torch.load(state_dict_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        print(f"Model loaded from {model_path}")
        return model

if __name__ == "__main__":
    # Test the model
    config = ModelConfig(
        vocab_size=50304,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12
    )
    
    model = LimeLLMForCausalLM(config)
    
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Expected ~2B parameters: {config.total_parameters:,}")
    
    # Test forward pass
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids)
        print(f"Output logits shape: {outputs['logits'].shape}")
        
        # Test generation
        generated = model.generate(
            input_ids[:1, :10],  # Use first batch, first 10 tokens
            max_length=20,
            do_sample=False
        )
        print(f"Generated sequence shape: {generated.shape}")
    
    print("Model architecture test completed successfully!")