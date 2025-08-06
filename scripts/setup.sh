#!/bin/bash

# LimeLLM Setup Script
# This script sets up the development environment for LimeLLM

set -e  # Exit on any error

echo "🍋 LimeLLM Setup Script"
echo "======================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running on supported OS
print_step "Checking operating system..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    print_status "Detected Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    print_status "Detected macOS"
elif [[ "$OSTYPE" == "msys" ]]; then
    OS="windows"
    print_status "Detected Windows (Git Bash/MSYS2)"
else
    print_warning "Unknown OS: $OSTYPE. Proceeding with default Linux setup..."
    OS="linux"
fi

# Check Python version
print_step "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d " " -f 2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 8 ]]; then
        print_status "Python $PYTHON_VERSION is supported"
        PYTHON_CMD="python3"
    else
        print_error "Python 3.8+ is required. Found: $PYTHON_VERSION"
        exit 1
    fi
elif command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | cut -d " " -f 2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 8 ]]; then
        print_status "Python $PYTHON_VERSION is supported"
        PYTHON_CMD="python"
    else
        print_error "Python 3.8+ is required. Found: $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python is not installed or not in PATH"
    exit 1
fi

# Check if we're in the right directory
print_step "Checking project directory..."
if [[ ! -f "requirements.txt" ]] || [[ ! -f "README.md" ]] || [[ ! -d "model" ]]; then
    print_error "Please run this script from the LimeLLM root directory"
    exit 1
fi
print_status "In LimeLLM project directory"

# Create virtual environment
print_step "Creating virtual environment..."
if [[ ! -d "venv" ]]; then
    $PYTHON_CMD -m venv venv
    print_status "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_step "Activating virtual environment..."
if [[ "$OS" == "windows" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi
print_status "Virtual environment activated"

# Upgrade pip
print_step "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch (with CUDA support if available)
print_step "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    print_status "NVIDIA GPU detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    print_warning "No NVIDIA GPU detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install main requirements
print_step "Installing project dependencies..."
pip install -r requirements.txt

# Install development dependencies if they exist
if [[ -f "requirements-dev.txt" ]]; then
    print_step "Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

# Create necessary directories
print_step "Creating project directories..."
mkdir -p data/{raw,processed,train,eval}
mkdir -p outputs/{models,logs,checkpoints}
mkdir -p configs
mkdir -p ollama-models
mkdir -p results
print_status "Project directories created"

# Create default config files
print_step "Creating default configuration files..."

# Model config
cat > configs/model_config.json << EOF
{
    "vocab_size": 50304,
    "n_positions": 4096,
    "n_embd": 1536,
    "n_layer": 24,
    "n_head": 12,
    "use_cache": true,
    "pad_token_id": 50256,
    "eos_token_id": 50256,
    "bos_token_id": 50256,
    "embd_pdrop": 0.1,
    "resid_pdrop": 0.1,
    "attn_pdrop": 0.1,
    "layer_norm_epsilon": 1e-5,
    "activation_function": "gelu_new",
    "gradient_checkpointing": true,
    "use_flash_attention": true
}
EOF

# Training config
cat > configs/training_config.json << EOF
{
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "beta1": 0.9,
    "beta2": 0.95,
    "epsilon": 1e-8,
    "warmup_steps": 2000,
    "lr_decay_steps": 100000,
    "min_lr_ratio": 0.1,
    "batch_size": 32,
    "gradient_accumulation_steps": 4,
    "max_grad_norm": 1.0,
    "max_steps": 100000,
    "save_steps": 5000,
    "eval_steps": 1000,
    "logging_steps": 100,
    "max_length": 4096,
    "train_data_path": "data/train",
    "eval_data_path": "data/eval",
    "use_ddp": true,
    "use_deepspeed": true,
    "deepspeed_config": "configs/deepspeed_config.json",
    "fp16": true,
    "bf16": false,
    "output_dir": "outputs",
    "wandb_project": "limellm"
}
EOF

# Data collection config
cat > configs/data_config.json << EOF
{
    "collect_python_docs": true,
    "collect_github": true,
    "collect_stackoverflow": true,
    "collect_pypi": true,
    "python_docs_max_pages": 500,
    "github_min_stars": 300,
    "github_max_repos": 50,
    "stackoverflow_min_votes": 10,
    "stackoverflow_max_questions": 500,
    "pypi_max_packages": 50,
    "data_mixing_ratios": {
        "github": 0.4,
        "stackoverflow": 0.3,
        "python_docs": 0.2,
        "pypi": 0.1
    }
}
EOF

# DeepSpeed config
cat > configs/deepspeed_config.json << EOF
{
    "train_batch_size": 128,
    "train_micro_batch_size_per_gpu": 32,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "scheduler": {
        "type": "WarmupCosineLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-4,
            "warmup_num_steps": 2000,
            "total_num_steps": 100000
        }
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 200000000,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 200000000,
        "contiguous_gradients": true
    },
    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": false,
        "contiguous_memory_optimization": false,
        "synchronize_checkpoint_boundary": false
    }
}
EOF

print_status "Default configuration files created"

# Set up git hooks if git is available and this is a git repo
if command -v git &> /dev/null && [[ -d ".git" ]]; then
    print_step "Setting up git hooks..."
    
    # Create pre-commit hook
    cat > .git/hooks/pre-commit << EOF
#!/bin/bash
# LimeLLM pre-commit hook

echo "Running pre-commit checks..."

# Check if black is available
if command -v black &> /dev/null; then
    echo "Running black formatter..."
    black --check . || {
        echo "Code formatting issues found. Run 'black .' to fix them."
        exit 1
    }
fi

# Check if flake8 is available
if command -v flake8 &> /dev/null; then
    echo "Running flake8 linter..."
    flake8 . || {
        echo "Linting issues found. Please fix them before committing."
        exit 1
    }
fi

echo "Pre-commit checks passed!"
EOF
    
    chmod +x .git/hooks/pre-commit
    print_status "Git hooks set up"
fi

# Check GPU availability
print_step "Checking GPU availability..."
if $PYTHON_CMD -c "import torch; print('CUDA available:', torch.cuda.is_available())" | grep -q "True"; then
    GPU_COUNT=$($PYTHON_CMD -c "import torch; print(torch.cuda.device_count())")
    GPU_NAME=$($PYTHON_CMD -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
    print_status "CUDA is available with $GPU_COUNT GPU(s): $GPU_NAME"
else
    print_warning "CUDA is not available. Training will use CPU (very slow)"
fi

# Create quick start script
print_step "Creating quick start script..."
cat > scripts/quickstart.sh << 'EOF'
#!/bin/bash

# LimeLLM Quick Start Script

echo "🍋 LimeLLM Quick Start"
echo "===================="

# Activate virtual environment
if [[ -d "venv" ]]; then
    source venv/bin/activate || source venv/Scripts/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Run setup.sh first."
    exit 1
fi

echo ""
echo "Available commands:"
echo "1. Collect data: python data_collection/data_collector.py --output-dir data/raw"
echo "2. Process data: python preprocessing/process_data.py --input-dir data/raw --output-dir data/processed"
echo "3. Train model: python training/train.py --train-data data/processed --output-dir outputs/limellm-2b"
echo "4. Evaluate model: python evaluation/benchmarks/humaneval_runner.py --model-path outputs/limellm-2b"
echo "5. Export to Ollama: python export/ollama_converter.py --model-path outputs/limellm-2b --output-dir ollama-models/limellm"
echo ""
echo "For more information, see README.md"
EOF

chmod +x scripts/quickstart.sh

# Create environment file
print_step "Creating environment file..."
cat > .env.example << EOF
# LimeLLM Environment Variables

# Training
WANDB_PROJECT=limellm
WANDB_API_KEY=your_wandb_api_key_here

# GitHub API (for data collection)
GITHUB_TOKEN=your_github_token_here

# DeepSpeed
CUDA_VISIBLE_DEVICES=0,1,2,3

# Data paths
DATA_DIR=./data
OUTPUT_DIR=./outputs
MODEL_DIR=./models

# Logging
LOG_LEVEL=INFO
EOF

print_status "Environment example file created (.env.example)"

# Final checks and summary
print_step "Running final checks..."

# Test imports
$PYTHON_CMD -c "
import torch
import transformers
import datasets
import numpy as np
print('✅ All required packages imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'Transformers version: {transformers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. (Optional) Set up environment variables: cp .env.example .env && nano .env"
echo "3. Start collecting data: python data_collection/data_collector.py --output-dir data/raw"
echo "4. Or run the quickstart script: ./scripts/quickstart.sh"
echo ""
echo "📚 Read the README.md for detailed instructions"
echo "🐛 If you encounter issues, check the troubleshooting section in docs/"
echo ""
echo "Happy coding with LimeLLM! 🍋✨"
EOF