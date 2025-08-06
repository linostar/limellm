import struct
import numpy as np
from typing import Dict, List, Tuple, Any, BinaryIO, Optional
import logging
import torch
from enum import IntEnum

logger = logging.getLogger(__name__)

class GGMLType(IntEnum):
    """GGML tensor data types."""
    F32     = 0
    F16     = 1
    Q4_0    = 2
    Q4_1    = 3
    Q5_0    = 6
    Q5_1    = 7
    Q8_0    = 8
    Q8_1    = 9
    Q2_K    = 10
    Q3_K    = 11
    Q4_K    = 12
    Q5_K    = 13
    Q6_K    = 14
    Q8_K    = 15

class GGUFValueType(IntEnum):
    """GGUF metadata value types."""
    UINT8   = 0
    INT8    = 1
    UINT16  = 2
    INT16   = 3
    UINT32  = 4
    INT32   = 5
    FLOAT32 = 6
    BOOL    = 7
    STRING  = 8
    ARRAY   = 9
    UINT64  = 10
    INT64   = 11
    FLOAT64 = 12

class GGUFWriter:
    """GGUF file format writer for model conversion."""
    
    def __init__(self):
        self.metadata: Dict[str, Any] = {}
        self.tensors: List[Dict[str, Any]] = []
        self.data_alignment = 32
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata key-value pair."""
        self.metadata[key] = value
    
    def add_tensor(self, name: str, tensor: torch.Tensor, ggml_type: GGMLType = GGMLType.F32):
        """Add a tensor to the GGUF file."""
        # Convert tensor to appropriate format
        if ggml_type == GGMLType.F32:
            data = tensor.detach().cpu().numpy().astype(np.float32)
        elif ggml_type == GGMLType.F16:
            data = tensor.detach().cpu().numpy().astype(np.float16)
        else:
            # For quantized types, we'd need quantization logic
            # This is a simplified implementation
            logger.warning(f"Quantization type {ggml_type} not fully implemented, using F16")
            data = tensor.detach().cpu().numpy().astype(np.float16)
            ggml_type = GGMLType.F16
        
        tensor_info = {
            'name': name,
            'shape': list(tensor.shape),
            'dtype': ggml_type,
            'data': data
        }
        
        self.tensors.append(tensor_info)
        logger.debug(f"Added tensor: {name}, shape: {tensor.shape}, type: {ggml_type}")
    
    def write(self, filepath: str):
        """Write GGUF file to disk."""
        logger.info(f"Writing GGUF file: {filepath}")
        
        with open(filepath, 'wb') as f:
            # Write header
            self._write_header(f)
            
            # Write metadata
            self._write_metadata(f)
            
            # Write tensor info
            self._write_tensor_info(f)
            
            # Align to data boundary
            self._align_file(f, self.data_alignment)
            
            # Write tensor data
            self._write_tensor_data(f)
        
        logger.info(f"GGUF file written successfully: {filepath}")
    
    def _write_header(self, f: BinaryIO):
        """Write GGUF header."""
        # Magic number
        f.write(b'GGUF')
        
        # Version (3 is the current version)
        f.write(struct.pack('<I', 3))
        
        # Tensor count
        f.write(struct.pack('<Q', len(self.tensors)))
        
        # Metadata count
        f.write(struct.pack('<Q', len(self.metadata)))
    
    def _write_metadata(self, f: BinaryIO):
        """Write metadata section."""
        for key, value in self.metadata.items():
            # Write key
            self._write_string(f, key)
            
            # Write value
            self._write_value(f, value)
    
    def _write_tensor_info(self, f: BinaryIO):
        """Write tensor information section."""
        for tensor_info in self.tensors:
            # Tensor name
            self._write_string(f, tensor_info['name'])
            
            # Number of dimensions
            f.write(struct.pack('<I', len(tensor_info['shape'])))
            
            # Dimensions
            for dim in tensor_info['shape']:
                f.write(struct.pack('<Q', dim))
            
            # Data type
            f.write(struct.pack('<I', tensor_info['dtype']))
            
            # Offset (will be filled when writing data)
            f.write(struct.pack('<Q', 0))  # Placeholder
    
    def _write_tensor_data(self, f: BinaryIO):
        """Write tensor data section."""
        for tensor_info in self.tensors:
            data = tensor_info['data']
            
            # Write tensor data
            f.write(data.tobytes())
            
            # Align to next boundary if needed
            self._align_file(f, self.data_alignment)
    
    def _write_string(self, f: BinaryIO, s: str):
        """Write string with length prefix."""
        s_bytes = s.encode('utf-8')
        f.write(struct.pack('<Q', len(s_bytes)))
        f.write(s_bytes)
    
    def _write_value(self, f: BinaryIO, value: Any):
        """Write a metadata value with appropriate type."""
        if isinstance(value, bool):
            f.write(struct.pack('<I', GGUFValueType.BOOL))
            f.write(struct.pack('<B', value))
        
        elif isinstance(value, int):
            if -(2**63) <= value < 2**63:
                f.write(struct.pack('<I', GGUFValueType.INT64))
                f.write(struct.pack('<q', value))
            else:
                f.write(struct.pack('<I', GGUFValueType.UINT64))
                f.write(struct.pack('<Q', value))
        
        elif isinstance(value, float):
            f.write(struct.pack('<I', GGUFValueType.FLOAT64))
            f.write(struct.pack('<d', value))
        
        elif isinstance(value, str):
            f.write(struct.pack('<I', GGUFValueType.STRING))
            self._write_string(f, value)
        
        elif isinstance(value, list):
            f.write(struct.pack('<I', GGUFValueType.ARRAY))
            
            # Array element type (assume all elements are same type)
            if value:
                element_type = type(value[0])
                if element_type == int:
                    f.write(struct.pack('<I', GGUFValueType.INT64))
                elif element_type == float:
                    f.write(struct.pack('<I', GGUFValueType.FLOAT64))
                elif element_type == str:
                    f.write(struct.pack('<I', GGUFValueType.STRING))
                else:
                    f.write(struct.pack('<I', GGUFValueType.STRING))  # Default to string
            else:
                f.write(struct.pack('<I', GGUFValueType.STRING))  # Empty array as string array
            
            # Array length
            f.write(struct.pack('<Q', len(value)))
            
            # Array elements
            for item in value:
                if isinstance(item, int):
                    f.write(struct.pack('<q', item))
                elif isinstance(item, float):
                    f.write(struct.pack('<d', item))
                else:
                    self._write_string(f, str(item))
        
        else:
            # Default to string representation
            f.write(struct.pack('<I', GGUFValueType.STRING))
            self._write_string(f, str(value))
    
    def _align_file(self, f: BinaryIO, alignment: int):
        """Align file position to specified boundary."""
        pos = f.tell()
        aligned_pos = (pos + alignment - 1) // alignment * alignment
        
        if aligned_pos > pos:
            # Write padding bytes
            f.write(b'\x00' * (aligned_pos - pos))

class GGUFReader:
    """GGUF file format reader."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.metadata: Dict[str, Any] = {}
        self.tensors: Dict[str, Dict[str, Any]] = {}
        
        # Read the file
        self._read_file()
    
    def _read_file(self):
        """Read and parse GGUF file."""
        with open(self.filepath, 'rb') as f:
            # Read header
            magic = f.read(4)
            if magic != b'GGUF':
                raise ValueError("Invalid GGUF file: wrong magic number")
            
            version = struct.unpack('<I', f.read(4))[0]
            if version != 3:
                logger.warning(f"GGUF version {version} may not be fully supported")
            
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            metadata_count = struct.unpack('<Q', f.read(8))[0]
            
            logger.info(f"GGUF file: version={version}, tensors={tensor_count}, metadata={metadata_count}")
            
            # Read metadata
            self._read_metadata(f, metadata_count)
            
            # Read tensor info
            self._read_tensor_info(f, tensor_count)
    
    def _read_metadata(self, f: BinaryIO, count: int):
        """Read metadata section."""
        for _ in range(count):
            key = self._read_string(f)
            value = self._read_value(f)
            self.metadata[key] = value
    
    def _read_tensor_info(self, f: BinaryIO, count: int):
        """Read tensor information."""
        for _ in range(count):
            name = self._read_string(f)
            
            n_dims = struct.unpack('<I', f.read(4))[0]
            shape = []
            for _ in range(n_dims):
                dim = struct.unpack('<Q', f.read(8))[0]
                shape.append(dim)
            
            dtype = GGMLType(struct.unpack('<I', f.read(4))[0])
            offset = struct.unpack('<Q', f.read(8))[0]
            
            self.tensors[name] = {
                'shape': shape,
                'dtype': dtype,
                'offset': offset
            }
    
    def _read_string(self, f: BinaryIO) -> str:
        """Read string with length prefix."""
        length = struct.unpack('<Q', f.read(8))[0]
        return f.read(length).decode('utf-8')
    
    def _read_value(self, f: BinaryIO) -> Any:
        """Read a metadata value."""
        value_type = GGUFValueType(struct.unpack('<I', f.read(4))[0])
        
        if value_type == GGUFValueType.BOOL:
            return struct.unpack('<B', f.read(1))[0] != 0
        
        elif value_type == GGUFValueType.INT64:
            return struct.unpack('<q', f.read(8))[0]
        
        elif value_type == GGUFValueType.UINT64:
            return struct.unpack('<Q', f.read(8))[0]
        
        elif value_type == GGUFValueType.FLOAT64:
            return struct.unpack('<d', f.read(8))[0]
        
        elif value_type == GGUFValueType.STRING:
            return self._read_string(f)
        
        elif value_type == GGUFValueType.ARRAY:
            element_type = GGUFValueType(struct.unpack('<I', f.read(4))[0])
            length = struct.unpack('<Q', f.read(8))[0]
            
            elements = []
            for _ in range(length):
                if element_type == GGUFValueType.INT64:
                    elements.append(struct.unpack('<q', f.read(8))[0])
                elif element_type == GGUFValueType.FLOAT64:
                    elements.append(struct.unpack('<d', f.read(8))[0])
                elif element_type == GGUFValueType.STRING:
                    elements.append(self._read_string(f))
                else:
                    # Skip unknown types
                    logger.warning(f"Unknown array element type: {element_type}")
                    break
            
            return elements
        
        else:
            logger.warning(f"Unknown value type: {value_type}")
            return None

def convert_pytorch_to_gguf(
    model: torch.nn.Module,
    output_path: str,
    model_config: Dict[str, Any],
    tokenizer_config: Optional[Dict[str, Any]] = None,
    quantization: str = "f16"
) -> str:
    """
    Convert PyTorch model to GGUF format.
    
    Args:
        model: PyTorch model
        output_path: Output GGUF file path
        model_config: Model configuration dictionary
        tokenizer_config: Tokenizer configuration
        quantization: Quantization type (f16, f32, q4_0, etc.)
    
    Returns:
        Path to the created GGUF file
    """
    
    writer = GGUFWriter()
    
    # Add model metadata
    writer.add_metadata("general.architecture", "llama")  # Use llama architecture as base
    writer.add_metadata("general.name", "LimeLLM")
    writer.add_metadata("general.description", "LimeLLM - Python specialized language model")
    
    # Model architecture metadata
    writer.add_metadata("llama.context_length", model_config.get("n_positions", 4096))
    writer.add_metadata("llama.embedding_length", model_config.get("n_embd", 1536))
    writer.add_metadata("llama.block_count", model_config.get("n_layer", 24))
    writer.add_metadata("llama.attention.head_count", model_config.get("n_head", 12))
    writer.add_metadata("llama.attention.layer_norm_epsilon", model_config.get("layer_norm_epsilon", 1e-5))
    writer.add_metadata("llama.vocab_size", model_config.get("vocab_size", 50304))
    
    # Tokenizer metadata
    if tokenizer_config:
        writer.add_metadata("tokenizer.ggml.model", "gpt2")
        writer.add_metadata("tokenizer.ggml.tokens", tokenizer_config.get("vocab", []))
        writer.add_metadata("tokenizer.ggml.scores", [0.0] * len(tokenizer_config.get("vocab", [])))
        writer.add_metadata("tokenizer.ggml.token_type", [1] * len(tokenizer_config.get("vocab", [])))
        
        # Special tokens
        writer.add_metadata("tokenizer.ggml.bos_token_id", tokenizer_config.get("bos_token_id", 50256))
        writer.add_metadata("tokenizer.ggml.eos_token_id", tokenizer_config.get("eos_token_id", 50256))
        writer.add_metadata("tokenizer.ggml.pad_token_id", tokenizer_config.get("pad_token_id", 50256))
    
    # Determine quantization type
    if quantization == "f32":
        ggml_type = GGMLType.F32
    elif quantization == "f16":
        ggml_type = GGMLType.F16
    elif quantization == "q4_0":
        ggml_type = GGMLType.Q4_0
    elif quantization == "q4_1":
        ggml_type = GGMLType.Q4_1
    elif quantization == "q5_0":
        ggml_type = GGMLType.Q5_0
    elif quantization == "q5_1":
        ggml_type = GGMLType.Q5_1
    elif quantization == "q8_0":
        ggml_type = GGMLType.Q8_0
    else:
        logger.warning(f"Unknown quantization {quantization}, using F16")
        ggml_type = GGMLType.F16
    
    # Add tensors
    model_dict = model.state_dict()
    
    for name, tensor in model_dict.items():
        # Map parameter names to GGUF format
        gguf_name = _map_parameter_name(name)
        if gguf_name:
            writer.add_tensor(gguf_name, tensor, ggml_type)
            logger.debug(f"Added tensor: {name} -> {gguf_name}")
    
    # Write GGUF file
    writer.write(output_path)
    
    return output_path

def _map_parameter_name(pytorch_name: str) -> Optional[str]:
    """Map PyTorch parameter name to GGUF format."""
    
    # Token embeddings
    if "embeddings.token_embeddings.weight" in pytorch_name:
        return "token_embd.weight"
    
    # Transformer blocks
    if "layers." in pytorch_name:
        parts = pytorch_name.split(".")
        layer_idx = parts[1]  # Extract layer index
        
        remainder = ".".join(parts[2:])
        
        # Attention weights
        if "attention.qkv_proj.weight" in remainder:
            return f"blk.{layer_idx}.attn_qkv.weight"
        elif "attention.out_proj.weight" in remainder:
            return f"blk.{layer_idx}.attn_output.weight"
        elif "attention.qkv_proj.bias" in remainder:
            return f"blk.{layer_idx}.attn_qkv.bias"
        elif "attention.out_proj.bias" in remainder:
            return f"blk.{layer_idx}.attn_output.bias"
        
        # MLP weights
        elif "mlp.fc1.weight" in remainder:
            return f"blk.{layer_idx}.ffn_gate.weight"
        elif "mlp.fc2.weight" in remainder:
            return f"blk.{layer_idx}.ffn_down.weight"
        elif "mlp.fc1.bias" in remainder:
            return f"blk.{layer_idx}.ffn_gate.bias"
        elif "mlp.fc2.bias" in remainder:
            return f"blk.{layer_idx}.ffn_down.bias"
        
        # Layer norms
        elif "ln_1.weight" in remainder:
            return f"blk.{layer_idx}.attn_norm.weight"
        elif "ln_2.weight" in remainder:
            return f"blk.{layer_idx}.ffn_norm.weight"
        elif "ln_1.bias" in remainder:
            return f"blk.{layer_idx}.attn_norm.bias"
        elif "ln_2.bias" in remainder:
            return f"blk.{layer_idx}.ffn_norm.bias"
    
    # Final layer norm
    if "ln_f.weight" in pytorch_name:
        return "output_norm.weight"
    elif "ln_f.bias" in pytorch_name:
        return "output_norm.bias"
    
    # Output projection
    if "lm_head.weight" in pytorch_name:
        return "output.weight"
    elif "lm_head.bias" in pytorch_name:
        return "output.bias"
    
    # If we can't map it, skip
    logger.warning(f"Could not map parameter: {pytorch_name}")
    return None

if __name__ == "__main__":
    # Test GGUF utilities
    import tempfile
    
    # Create a simple test tensor
    test_tensor = torch.randn(10, 20)
    
    with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as tmp:
        # Test writing
        writer = GGUFWriter()
        writer.add_metadata("test.name", "test_model")
        writer.add_metadata("test.version", 1.0)
        writer.add_tensor("test_tensor", test_tensor)
        writer.write(tmp.name)
        
        print(f"Test GGUF file written: {tmp.name}")
        
        # Test reading
        reader = GGUFReader(tmp.name)
        print(f"Metadata: {reader.metadata}")
        print(f"Tensors: {list(reader.tensors.keys())}")
        
        # Cleanup
        import os
        os.unlink(tmp.name)