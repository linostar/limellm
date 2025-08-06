import os
import json
import struct
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from pathlib import Path
import tempfile
import subprocess

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.architecture import LimeLLMForCausalLM

logger = logging.getLogger(__name__)

class OllamaModelConverter:
    """Convert LimeLLM models to Ollama-compatible GGUF format."""
    
    def __init__(self):
        self.supported_quantizations = {
            'f32': {'dtype': torch.float32, 'suffix': 'f32'},
            'f16': {'dtype': torch.float16, 'suffix': 'f16'},
            'q8_0': {'suffix': 'q8_0'},
            'q4_0': {'suffix': 'q4_0'},
            'q4_1': {'suffix': 'q4_1'},
            'q5_0': {'suffix': 'q5_0'},
            'q5_1': {'suffix': 'q5_1'},
        }
    
    def convert_to_ollama(
        self,
        model_path: str,
        output_dir: str,
        model_name: str = "limellm",
        quantization: str = "f16",
        template: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Convert LimeLLM model to Ollama format.
        
        Args:
            model_path: Path to the trained model
            output_dir: Output directory for Ollama files
            model_name: Name for the Ollama model
            quantization: Quantization type (f32, f16, q8_0, q4_0, etc.)
            template: Custom chat template
            system_prompt: Default system prompt
            
        Returns:
            Path to the created Ollama model directory
        """
        
        if quantization not in self.supported_quantizations:
            raise ValueError(f"Unsupported quantization: {quantization}. Supported: {list(self.supported_quantizations.keys())}")
        
        logger.info(f"Converting {model_path} to Ollama format...")
        logger.info(f"Target quantization: {quantization}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the model
        model, config, tokenizer = self._load_model(model_path)
        
        # Convert to GGUF format
        gguf_path = self._convert_to_gguf(
            model=model,
            config=config,
            tokenizer=tokenizer,
            output_path=os.path.join(output_dir, f"{model_name}.gguf"),
            quantization=quantization
        )
        
        # Create Modelfile
        modelfile_path = self._create_modelfile(
            gguf_path=gguf_path,
            output_dir=output_dir,
            model_name=model_name,
            template=template,
            system_prompt=system_prompt,
            config=config
        )
        
        # Create documentation
        self._create_documentation(output_dir, model_name, config, quantization)
        
        logger.info(f"Ollama model created successfully at: {output_dir}")
        logger.info(f"To use with Ollama: ollama create {model_name} -f {modelfile_path}")
        
        return output_dir
    
    def _load_model(self, model_path: str) -> Tuple[torch.nn.Module, ModelConfig, Any]:
        """Load the trained model and associated files."""
        
        if os.path.isdir(model_path):
            # HuggingFace format
            try:
                model = LimeLLMForCausalLM.from_pretrained(model_path)
                
                # Load config
                config_path = os.path.join(model_path, 'config.json')
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = ModelConfig.from_dict(config_dict)
                
                # Load tokenizer
                from transformers import GPT2TokenizerFast
                tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
                
            except Exception as e:
                raise ValueError(f"Failed to load HuggingFace format model: {e}")
        
        else:
            # PyTorch checkpoint
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                
                config = ModelConfig.from_dict(checkpoint['model_config'])
                model = LimeLLMForCausalLM(config)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # Use default tokenizer
                from transformers import GPT2TokenizerFast
                tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
                tokenizer.pad_token = tokenizer.eos_token
                
            except Exception as e:
                raise ValueError(f"Failed to load checkpoint: {e}")
        
        model.eval()
        logger.info(f"Model loaded successfully. Parameters: {model.get_num_params():,}")
        
        return model, config, tokenizer
    
    def _convert_to_gguf(
        self,
        model: torch.nn.Module,
        config: ModelConfig,
        tokenizer: Any,
        output_path: str,
        quantization: str
    ) -> str:
        """Convert model to GGUF format."""
        
        logger.info("Converting model to GGUF format...")
        
        try:
            # Try to use llama.cpp's convert script if available
            return self._convert_with_llamacpp(model, config, tokenizer, output_path, quantization)
        
        except Exception as e:
            logger.warning(f"llama.cpp conversion failed: {e}")
            logger.info("Falling back to manual GGUF conversion...")
            return self._convert_manual_gguf(model, config, tokenizer, output_path, quantization)
    
    def _convert_with_llamacpp(
        self,
        model: torch.nn.Module,
        config: ModelConfig,
        tokenizer: Any,
        output_path: str,
        quantization: str
    ) -> str:
        """Convert using llama.cpp tools if available."""
        
        # First save as HuggingFace format
        with tempfile.TemporaryDirectory() as temp_dir:
            hf_path = os.path.join(temp_dir, "hf_model")
            model.save_pretrained(hf_path)
            tokenizer.save_pretrained(hf_path)
            
            # Try to find llama.cpp convert script
            convert_script_paths = [
                "convert.py",  # In current directory
                "llama.cpp/convert.py",  # Common location
                os.path.expanduser("~/llama.cpp/convert.py"),  # User home
            ]
            
            convert_script = None
            for path in convert_script_paths:
                if os.path.exists(path):
                    convert_script = path
                    break
            
            if not convert_script:
                raise FileNotFoundError("llama.cpp convert script not found")
            
            # Convert to GGUF
            cmd = [
                "python", convert_script,
                hf_path,
                "--outfile", output_path,
                "--outtype", quantization
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Conversion failed: {result.stderr}")
        
        return output_path
    
    def _convert_manual_gguf(
        self,
        model: torch.nn.Module,
        config: ModelConfig,
        tokenizer: Any,
        output_path: str,
        quantization: str
    ) -> str:
        """Manual GGUF conversion (simplified)."""
        
        logger.warning("Manual GGUF conversion is simplified and may not be fully compatible")
        
        # This is a simplified implementation
        # For production use, you should use llama.cpp conversion tools
        
        # Collect model weights
        weights = {}
        
        # Token embeddings
        if hasattr(model.model.embeddings, 'token_embeddings'):
            weights['token_embd.weight'] = model.model.embeddings.token_embeddings.weight
        
        # Transformer blocks
        for i, layer in enumerate(model.model.layers):
            prefix = f'blk.{i}.'
            
            # Attention weights
            if hasattr(layer.attention, 'qkv_proj'):
                weights[f'{prefix}attn_qkv.weight'] = layer.attention.qkv_proj.weight
            if hasattr(layer.attention, 'out_proj'):
                weights[f'{prefix}attn_output.weight'] = layer.attention.out_proj.weight
            
            # MLP weights
            if hasattr(layer.mlp, 'fc1'):
                weights[f'{prefix}ffn_gate.weight'] = layer.mlp.fc1.weight
            if hasattr(layer.mlp, 'fc2'):
                weights[f'{prefix}ffn_down.weight'] = layer.mlp.fc2.weight
            
            # Layer norms
            if hasattr(layer, 'ln_1'):
                weights[f'{prefix}attn_norm.weight'] = layer.ln_1.weight
            if hasattr(layer, 'ln_2'):
                weights[f'{prefix}ffn_norm.weight'] = layer.ln_2.weight
        
        # Final layer norm
        if hasattr(model.model, 'ln_f'):
            weights['output_norm.weight'] = model.model.ln_f.weight
        
        # Output projection (language modeling head)
        if hasattr(model.model, 'lm_head'):
            weights['output.weight'] = model.model.lm_head.weight
        
        # Apply quantization
        quantized_weights = self._quantize_weights(weights, quantization)
        
        # Write GGUF file (simplified format)
        self._write_gguf_file(quantized_weights, config, tokenizer, output_path)
        
        return output_path
    
    def _quantize_weights(self, weights: Dict[str, torch.Tensor], quantization: str) -> Dict[str, torch.Tensor]:
        """Apply quantization to weights."""
        
        quantized = {}
        quant_info = self.supported_quantizations[quantization]
        
        for name, tensor in weights.items():
            if quantization in ['f32', 'f16']:
                # Simple dtype conversion
                dtype = quant_info['dtype']
                quantized[name] = tensor.to(dtype)
            
            else:
                # For quantized formats, we'd need more sophisticated quantization
                # This is a placeholder - use llama.cpp for proper quantization
                logger.warning(f"Quantization {quantization} not fully implemented in manual converter")
                quantized[name] = tensor.to(torch.float16)  # Fallback to f16
        
        return quantized
    
    def _write_gguf_file(
        self,
        weights: Dict[str, torch.Tensor],
        config: ModelConfig,
        tokenizer: Any,
        output_path: str
    ):
        """Write GGUF file (simplified implementation)."""
        
        # This is a very simplified GGUF writer
        # For production, use the official GGUF implementation
        
        with open(output_path, 'wb') as f:
            # GGUF header
            f.write(b'GGUF')
            f.write(struct.pack('<I', 3))  # Version
            f.write(struct.pack('<Q', len(weights)))  # Tensor count
            f.write(struct.pack('<Q', 0))  # Metadata count (simplified)
            
            # Write tensors (simplified)
            for name, tensor in weights.items():
                name_bytes = name.encode('utf-8')
                f.write(struct.pack('<I', len(name_bytes)))
                f.write(name_bytes)
                
                # Tensor info
                f.write(struct.pack('<I', len(tensor.shape)))  # n_dims
                for dim in tensor.shape:
                    f.write(struct.pack('<Q', dim))
                
                # Data type (simplified mapping)
                f.write(struct.pack('<I', 1))  # GGML_TYPE_F32 (simplified)
                
                # Offset (we'll write data immediately)
                f.write(struct.pack('<Q', 0))
                
                # Write tensor data
                tensor_bytes = tensor.detach().cpu().numpy().astype(np.float32).tobytes()
                f.write(tensor_bytes)
        
        logger.info(f"GGUF file written: {output_path}")
    
    def _create_modelfile(
        self,
        gguf_path: str,
        output_dir: str,
        model_name: str,
        template: Optional[str],
        system_prompt: Optional[str],
        config: ModelConfig
    ) -> str:
        """Create Ollama Modelfile."""
        
        modelfile_path = os.path.join(output_dir, "Modelfile")
        
        # Default template for code generation
        if template is None:
            template = '''{{ if .System }}System: {{ .System }}

{{ end }}{{ if .Prompt }}User: {{ .Prompt }}

{{ end }}Assistant: '''
        
        # Default system prompt
        if system_prompt is None:
            system_prompt = "You are LimeLLM, a helpful AI assistant specialized in Python programming. You can help with code generation, debugging, refactoring, and explaining code concepts."
        
        modelfile_content = f'''FROM {os.path.basename(gguf_path)}

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 50
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx {config.n_positions}
PARAMETER stop "<|endoftext|>"
PARAMETER stop "User:"
PARAMETER stop "System:"

# System prompt
SYSTEM """{system_prompt}"""

# Chat template
TEMPLATE """{template}"""

# Metadata
PARAMETER num_gpu 999  # Use all available GPUs
'''
        
        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        
        logger.info(f"Modelfile created: {modelfile_path}")
        return modelfile_path
    
    def _create_documentation(
        self,
        output_dir: str,
        model_name: str,
        config: ModelConfig,
        quantization: str
    ):
        """Create documentation for the Ollama model."""
        
        readme_content = f'''# {model_name.title()} - Ollama Model

## Overview
This is LimeLLM, a {config.total_parameters:,} parameter language model specialized for Python code generation and programming assistance.

## Model Details
- **Architecture**: Transformer decoder
- **Parameters**: {config.total_parameters:,}
- **Context Length**: {config.n_positions:,} tokens
- **Quantization**: {quantization}
- **Vocabulary Size**: {config.vocab_size:,}

## Usage

### Installation
First, make sure you have Ollama installed, then create the model:

```bash
ollama create {model_name} -f Modelfile
```

### Running the Model
```bash
# Interactive chat
ollama run {model_name}

# Single prompt
ollama run {model_name} "Write a Python function to calculate fibonacci numbers"
```

### API Usage
```bash
curl -X POST http://localhost:11434/api/generate \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "{model_name}",
    "prompt": "def quicksort(arr):",
    "stream": false
  }}'
```

## Capabilities
- **Code Generation**: Generate Python functions, classes, and complete programs
- **Code Completion**: Complete partial code snippets
- **Debugging**: Help identify and fix code issues
- **Refactoring**: Improve code structure and readability
- **Documentation**: Generate docstrings and comments
- **Explanations**: Explain code concepts and algorithms

## Best Practices
1. **Be specific**: Provide clear, detailed prompts for better results
2. **Use examples**: Include input/output examples when possible
3. **Set context**: Explain the purpose and requirements
4. **Temperature**: Use lower temperature (0.1-0.3) for code generation, higher (0.7-0.9) for creative tasks

## Example Prompts

### Code Generation
```
Write a Python function that reads a CSV file and returns a pandas DataFrame with data validation.
```

### Debugging
```
Here's my Python code that's giving an error. Can you help fix it?

def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result

# Error: TypeError: '>' not supported between instances of 'str' and 'int'
```

### Refactoring
```
Please refactor this code to make it more readable and efficient:

def calc(x, y, op):
    if op == 'add':
        return x + y
    elif op == 'sub':
        return x - y
    elif op == 'mul':
        return x * y
    elif op == 'div':
        return x / y
```

## Limitations
- Specialized for Python (other languages may have lower quality)
- Based on training data cutoff
- May occasionally generate plausible but incorrect code
- Always review and test generated code

## License
This model is released under the same license as the original LimeLLM project.
'''
        
        readme_path = os.path.join(output_dir, 'README.md')
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # Create usage examples
        examples_content = '''# LimeLLM Usage Examples

## Basic Code Generation

### Function Generation
**Prompt**: `Write a Python function to calculate the factorial of a number using recursion.`

**Expected Output**:
```python
def factorial(n):
    """Calculate factorial using recursion."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

### Class Generation
**Prompt**: `Create a Python class for a simple bank account with deposit and withdraw methods.`

### Data Processing
**Prompt**: `Write a function to parse a log file and extract error messages with timestamps.`

## Code Completion

### Partial Function
**Prompt**: 
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    # Complete this function
```

### Class Method
**Prompt**:
```python
class DataProcessor:
    def __init__(self, data):
        self.data = data
    
    def clean_data(self):
        # Complete this method to remove null values and duplicates
```

## Debugging and Refactoring

### Bug Fix
**Prompt**: `This code has a bug. Can you fix it and explain the issue?`

### Code Review
**Prompt**: `Review this code and suggest improvements for readability and performance.`

## Advanced Usage

### Algorithm Implementation
**Prompt**: `Implement the A* pathfinding algorithm in Python with proper documentation.`

### Design Pattern
**Prompt**: `Show me how to implement the Observer pattern in Python with a practical example.`

### Testing
**Prompt**: `Write unit tests for this function using pytest.`
'''
        
        examples_path = os.path.join(output_dir, 'EXAMPLES.md')
        with open(examples_path, 'w', encoding='utf-8') as f:
            f.write(examples_content)
        
        logger.info(f"Documentation created in {output_dir}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert LimeLLM to Ollama format')
    parser.add_argument('--model-path', required=True, help='Path to the trained model')
    parser.add_argument('--output-dir', required=True, help='Output directory for Ollama files')
    parser.add_argument('--model-name', default='limellm', help='Name for the Ollama model')
    parser.add_argument('--quantization', default='f16', 
                        choices=['f32', 'f16', 'q8_0', 'q4_0', 'q4_1', 'q5_0', 'q5_1'],
                        help='Quantization format')
    parser.add_argument('--template', help='Custom chat template')
    parser.add_argument('--system-prompt', help='Custom system prompt')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Convert model
    converter = OllamaModelConverter()
    
    try:
        output_dir = converter.convert_to_ollama(
            model_path=args.model_path,
            output_dir=args.output_dir,
            model_name=args.model_name,
            quantization=args.quantization,
            template=args.template,
            system_prompt=args.system_prompt
        )
        
        print(f"\n✅ Conversion completed successfully!")
        print(f"📁 Output directory: {output_dir}")
        print(f"\n🚀 To use with Ollama:")
        print(f"   cd {output_dir}")
        print(f"   ollama create {args.model_name} -f Modelfile")
        print(f"   ollama run {args.model_name}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise

if __name__ == "__main__":
    main()