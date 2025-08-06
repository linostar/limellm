import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
import math
from typing import Optional, Tuple, Union
from .config import ModelConfig

class RotaryPositionalEmbedding(nn.Module):
    """RoPE (Rotary Position Embedding) for improved position encoding."""
    
    def __init__(self, dim: int, max_position: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position = max_position
        self.base = base
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate cos and sin for rotary embeddings."""
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        return cos, sin

def apply_rotary_pos_emb(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to tensor."""
    # tensor shape: [batch_size, num_heads, seq_len, head_dim]
    # cos, sin shape: [seq_len, head_dim // 2]
    
    # Split tensor into first half and second half
    tensor_1, tensor_2 = tensor[..., : tensor.shape[-1] // 2], tensor[..., tensor.shape[-1] // 2 :]
    
    # Apply rotation
    return torch.cat([
        tensor_1 * cos - tensor_2 * sin,
        tensor_1 * sin + tensor_2 * cos
    ], dim=-1)

class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional flash attention and RoPE."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        
        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(self.n_embd, 3 * self.n_embd, bias=True)
        self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        
        # Rotary position embedding
        self.rope = RotaryPositionalEmbedding(self.head_dim, config.n_positions)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        # Flash attention flag
        self.use_flash_attention = config.use_flash_attention and hasattr(F, 'scaled_dot_product_attention')
        
        # Causal mask
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(config.n_positions, config.n_positions, dtype=torch.bool)).view(
                1, 1, config.n_positions, config.n_positions
            ),
            persistent=False
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rope(seq_len, x.device)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        
        # Handle past key-values for generation
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        
        present = (k, v) if use_cache else None
        
        # Attention computation
        if self.use_flash_attention:
            # Use PyTorch's flash attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.config.attn_pdrop if self.training else 0.0,
                is_causal=True
            )
        else:
            # Manual attention computation
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            # Apply causal mask
            if seq_len > 1:
                causal_mask = self.causal_mask[:, :, :seq_len, :k.size(-2)]
                attn_weights = attn_weights.masked_fill(~causal_mask, -1e4)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.n_embd
        )
        
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)
        
        return output, present

class MLP(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 4x expansion factor is standard for transformers
        intermediate_size = 4 * config.n_embd
        
        self.fc1 = nn.Linear(config.n_embd, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, config.n_embd, bias=True)
        
        # Activation function
        if config.activation_function == "gelu_new":
            self.activation = self.gelu_new
        elif config.activation_function == "gelu":
            self.activation = F.gelu
        elif config.activation_function == "relu":
            self.activation = F.relu
        elif config.activation_function == "swish":
            self.activation = F.silu
        else:
            raise ValueError(f"Unsupported activation function: {config.activation_function}")
        
        self.dropout = nn.Dropout(config.resid_pdrop)
    
    def gelu_new(self, x: torch.Tensor) -> torch.Tensor:
        """GELU activation function with tanh approximation (more stable)."""
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(x)
        hidden = self.activation(hidden)
        output = self.fc2(hidden)
        output = self.dropout(output)
        return output

class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Pre-norm architecture (more stable training)
        self.ln_1 = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attention = MultiHeadAttention(config)
        
        self.ln_2 = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        # Self-attention with residual connection
        residual = x
        x = self.ln_1(x)
        attn_output, present = self.attention(
            x, 
            attention_mask=attention_mask, 
            use_cache=use_cache,
            past_key_value=past_key_value
        )
        x = residual + attn_output
        
        # MLP with residual connection
        residual = x
        x = self.ln_2(x)
        mlp_output = self.mlp(x)
        x = residual + mlp_output
        
        return x, present

class CodeAwareEmbedding(nn.Module):
    """Enhanced token embedding with code-aware features."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Main token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Code type embeddings (optional enhancement for code understanding)
        if config.code_token_types:
            self.code_type_embeddings = nn.Embedding(
                len(config.code_token_types), config.n_embd
            )
        else:
            self.code_type_embeddings = None
        
        # Dropout
        self.dropout = nn.Dropout(config.embd_pdrop)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        if self.code_type_embeddings is not None:
            nn.init.normal_(self.code_type_embeddings.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        code_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        # Token embeddings
        token_embeds = self.token_embeddings(input_ids)
        
        # Add code type embeddings if available
        if code_type_ids is not None and self.code_type_embeddings is not None:
            code_embeds = self.code_type_embeddings(code_type_ids)
            token_embeds = token_embeds + code_embeds
        
        embeddings = self.dropout(token_embeds)
        return embeddings

if __name__ == "__main__":
    # Test the layers
    from config import ModelConfig
    
    config = ModelConfig(
        vocab_size=50304,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12
    )
    
    # Test embedding
    embedding = CodeAwareEmbedding(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 64))  # batch_size=2, seq_len=64
    embeds = embedding(input_ids)
    print(f"Embedding output shape: {embeds.shape}")
    
    # Test transformer block
    block = TransformerBlock(config)
    output, _ = block(embeds)
    print(f"Transformer block output shape: {output.shape}")
    
    # Test attention
    attention = MultiHeadAttention(config)
    attn_out, _ = attention(embeds)
    print(f"Attention output shape: {attn_out.shape}")
    
    print(f"Total parameters in embedding: {sum(p.numel() for p in embedding.parameters()):,}")
    print(f"Total parameters in transformer block: {sum(p.numel() for p in block.parameters()):,}")