# LimeLLM

The first LLM ever created by an LLM.

A 2-billion parameter language model specialized for Python programming tasks. LimeLLM is designed to excel at code generation, debugging, refactoring, and providing programming assistance.

## 🚀 Features

- **2B Parameter Transformer**: Optimized architecture for code understanding
- **Python Specialization**: Trained on curated Python code, documentation, and Q&A
- **Code-Aware**: Enhanced tokenization and processing for programming languages
- **Ollama Compatible**: Easy deployment with Ollama for local inference
- **Comprehensive Toolchain**: Complete pipeline from data collection to deployment

## 📋 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM (32GB+ recommended for training)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/limellm.git
cd limellm

# Install dependencies
pip install -r requirements.txt
```

### Using Pre-trained Model with Ollama

```bash
# Convert trained model to Ollama format
python export/ollama_converter.py \\
    --model-path /path/to/trained/model \\
    --output-dir ./ollama-model \\
    --quantization f16

# Create Ollama model
cd ollama-model
ollama create limellm -f Modelfile

# Run the model
ollama run limellm "Write a Python function to calculate fibonacci numbers"
```

## 🎯 Use Cases

### Code Generation
Generate Python functions, classes, and complete programs:

```python
# Prompt: "Write a class for a binary search tree with insert and search methods"
class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, val):
        if not self.root:
            self.root = TreeNode(val)
        else:
            self._insert_recursive(self.root, val)
    # ... (continued)
```

### Code Debugging
Identify and fix issues in existing code:

```python
# Prompt: "Fix this buggy function"
# Input: def divide(a, b): return a / b
# Output:
def divide(a, b):
    if b == 0:
        raise ValueError("Division by zero is not allowed")
    return a / b
```

### Code Refactoring
Improve code structure and readability:

```python
# Transform procedural code into clean, object-oriented design
# Add type hints, docstrings, and error handling
# Optimize algorithms and data structures
```

## 🏗️ Training Your Own Model

### 1. Data Collection

Collect training data from various Python sources:

```bash
# Collect all data sources
python data_collection/data_collector.py \\
    --output-dir data/raw \\
    --config configs/data_collection.json

# Or collect specific sources
python data_collection/data_collector.py --python-docs-only
python data_collection/data_collector.py --github-only
python data_collection/data_collector.py --stackoverflow-only
```

### 2. Data Preprocessing

Clean and prepare the collected data:

```bash
python preprocessing/process_data.py \\
    --input-dir data/raw \\
    --output-dir data/processed \\
    --min-length 50 \\
    --max-length 4096
```

### 3. Training

Train the model with your data:

```bash
# Single GPU training
python training/train.py \\
    --train-data data/processed \\
    --model-config configs/model_config.json \\
    --training-config configs/training_config.json \\
    --output-dir outputs/limellm-2b

# Multi-GPU training with DeepSpeed
deepspeed training/train.py \\
    --train-data data/processed \\
    --model-config configs/model_config.json \\
    --training-config configs/training_config.json \\
    --output-dir outputs/limellm-2b \\
    --deepspeed configs/deepspeed_config.json
```

### 4. Evaluation

Evaluate your trained model:

```bash
# Run Human-Eval benchmark
python evaluation/benchmarks/humaneval_runner.py \\
    --model-path outputs/limellm-2b \\
    --output-file results/humaneval_results.json

# Generate evaluation report
python evaluation/generate_report.py \\
    --results-file results/humaneval_results.json \\
    --output-file results/evaluation_report.md
```

### 5. Export to Ollama

Convert your trained model for deployment:

```bash
python export/ollama_converter.py \\
    --model-path outputs/limellm-2b \\
    --output-dir ollama-models/limellm-2b \\
    --quantization f16 \\
    --model-name limellm-2b
```

## 🔧 Configuration

### Model Configuration

Configure model architecture in `configs/model_config.json`:

```json
{
  "vocab_size": 50304,
  "n_positions": 4096,
  "n_embd": 1536,
  "n_layer": 24,
  "n_head": 12,
  "activation_function": "gelu_new",
  "layer_norm_epsilon": 1e-5,
  "gradient_checkpointing": true
}
```

### Training Configuration

Set training parameters in `configs/training_config.json`:

```json
{
  "learning_rate": 1e-4,
  "batch_size": 32,
  "gradient_accumulation_steps": 4,
  "max_steps": 100000,
  "warmup_steps": 2000,
  "save_steps": 5000,
  "eval_steps": 1000,
  "fp16": true,
  "use_deepspeed": true
}
```

## 📊 Hardware Requirements

#### Training
- **Minimum**: 1x RTX 3090 (24GB VRAM)
- **Recommended**: 4x A100 (40GB VRAM each)
- **Memory**: 64GB+ system RAM
- **Storage**: 1TB+ SSD for datasets

#### Inference
- **CPU Only**: 16GB+ RAM
- **GPU**: RTX 3060+ (8GB VRAM)
- **Quantized**: RTX 3050+ (4GB VRAM with Q4 quantization)

## 🛠️ Development

### Project Structure

```
limellm/
├── data_collection/          # Web scrapers for training data
├── preprocessing/           # Data cleaning and processing
├── model/                  # Model architecture and configuration
├── training/               # Training scripts and utilities
├── evaluation/             # Evaluation benchmarks and metrics
├── export/                 # Model export utilities (Ollama, GGUF)
├── configs/                # Configuration files
├── scripts/                # Utility scripts
└── docs/                   # Documentation
```

### Adding New Features

1. **Data Sources**: Add new scrapers in `data_collection/scrapers/`
2. **Model Components**: Extend architecture in `model/`
3. **Training Methods**: Add training techniques in `training/`
4. **Evaluation Metrics**: Add benchmarks in `evaluation/`

### Testing

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Run code quality checks
black . --check
flake8 .
mypy .
```

## 📚 Documentation

- [Training Guide](docs/training.md) - Detailed training instructions
- [Evaluation Guide](docs/evaluation.md) - How to evaluate models
- [Deployment Guide](docs/deployment.md) - Deployment options and configurations
- [API Reference](docs/api.md) - Code API documentation
- [Contributing](CONTRIBUTING.md) - How to contribute to the project

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/your-username/limellm.git
cd limellm

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

## 📄 License

This project is licensed under the GNU GPL-3.0 license - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Transformers](https://github.com/huggingface/transformers) for the transformer implementation foundation
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) for distributed training capabilities
- [Ollama](https://github.com/jmorganca/ollama) for local model deployment
- [Human-Eval](https://github.com/openai/human-eval) for code generation benchmarking
