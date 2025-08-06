#!/usr/bin/env python3

"""
LimeLLM Training Runner

This script provides a comprehensive interface for training LimeLLM models
with various configurations and monitoring options.
"""

import os
import sys
import json
import argparse
import logging
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.config import ModelConfig, TrainingConfig, get_model_config

logger = logging.getLogger(__name__)

class TrainingRunner:
    """Orchestrate LimeLLM training with monitoring and checkpointing."""
    
    def __init__(self, args):
        self.args = args
        self.start_time = datetime.now()
        
        # Setup paths
        self.project_root = project_root
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configurations
        self.model_config = self._load_model_config()
        self.training_config = self._load_training_config()
        
        # Setup logging
        self._setup_logging()
        
        # Validate environment
        self._validate_environment()
    
    def _load_model_config(self) -> ModelConfig:
        """Load model configuration."""
        if self.args.model_config:
            config_path = Path(self.args.model_config)
            if config_path.exists():
                return ModelConfig.from_json(str(config_path))
            else:
                logger.error(f"Model config file not found: {config_path}")
                sys.exit(1)
        
        elif self.args.model_size:
            try:
                return get_model_config(f"limellm-{self.args.model_size}")
            except ValueError as e:
                logger.error(f"Invalid model size: {e}")
                sys.exit(1)
        
        else:
            # Default 2B model
            return get_model_config("limellm-2b")
    
    def _load_training_config(self) -> TrainingConfig:
        """Load training configuration."""
        if self.args.training_config:
            config_path = Path(self.args.training_config)
            if config_path.exists():
                config = TrainingConfig.from_json(str(config_path))
            else:
                logger.error(f"Training config file not found: {config_path}")
                sys.exit(1)
        else:
            config = TrainingConfig()
        
        # Override with command line arguments
        if self.args.learning_rate:
            config.learning_rate = self.args.learning_rate
        
        if self.args.batch_size:
            config.batch_size = self.args.batch_size
        
        if self.args.max_steps:
            config.max_steps = self.args.max_steps
        
        if self.args.output_dir:
            config.output_dir = self.args.output_dir
        
        if self.args.wandb_project:
            config.wandb_project = self.args.wandb_project
        
        if self.args.no_wandb:
            config.wandb_project = None
        
        return config
    
    def _setup_logging(self):
        """Setup comprehensive logging."""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Create timestamped log file
        log_file = log_dir / f"training_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log"
        
        # Configure logging
        log_level = logging.DEBUG if self.args.verbose else logging.INFO
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        logger.info(f"Logging setup complete. Log file: {log_file}")
    
    def _validate_environment(self):
        """Validate training environment and dependencies."""
        logger.info("Validating training environment...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major != 3 or python_version.minor < 8:
            logger.error(f"Python 3.8+ required, found {python_version.major}.{python_version.minor}")
            sys.exit(1)
        
        # Check required packages
        required_packages = [
            'torch', 'transformers', 'datasets', 'accelerate',
            'numpy', 'tqdm', 'wandb'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing required packages: {', '.join(missing_packages)}")
            logger.error("Run: pip install -r requirements.txt")
            sys.exit(1)
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"GPU available: {gpu_count} x {gpu_name}")
            else:
                logger.warning("No GPU available. Training will be very slow on CPU.")
        except Exception as e:
            logger.warning(f"Could not check GPU availability: {e}")
        
        # Check data directory
        if self.args.train_data:
            data_path = Path(self.args.train_data)
            if not data_path.exists():
                logger.error(f"Training data path does not exist: {data_path}")
                sys.exit(1)
            
            # Check for data files
            data_files = list(data_path.glob("**/*.jsonl"))
            if not data_files:
                logger.error(f"No JSONL data files found in: {data_path}")
                sys.exit(1)
            
            total_size = sum(f.stat().st_size for f in data_files) / (1024**3)  # GB
            logger.info(f"Found {len(data_files)} data files, total size: {total_size:.2f} GB")
        
        # Check DeepSpeed if requested
        if self.args.deepspeed or self.training_config.use_deepspeed:
            try:
                import deepspeed
                logger.info(f"DeepSpeed available: {deepspeed.__version__}")
            except ImportError:
                logger.error("DeepSpeed requested but not installed. Run: pip install deepspeed")
                sys.exit(1)
        
        logger.info("Environment validation passed")
    
    def _save_configs(self):
        """Save model and training configurations."""
        config_dir = self.output_dir / "configs"
        config_dir.mkdir(exist_ok=True)
        
        # Save model config
        model_config_path = config_dir / "model_config.json"
        self.model_config.save_json(str(model_config_path))
        
        # Save training config
        training_config_path = config_dir / "training_config.json"
        self.training_config.save_json(str(training_config_path))
        
        # Save run metadata
        metadata = {
            'start_time': self.start_time.isoformat(),
            'command_line': ' '.join(sys.argv),
            'working_directory': str(Path.cwd()),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'estimated_parameters': self.model_config.total_parameters,
            'training_args': vars(self.args)
        }
        
        metadata_path = config_dir / "run_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Configurations saved to: {config_dir}")
    
    def _estimate_training_time(self) -> Tuple[str, Dict]:
        """Estimate training time and resource requirements."""
        
        # Simple estimates based on model size and steps
        params = self.model_config.total_parameters
        steps = self.training_config.max_steps
        batch_size = self.training_config.batch_size
        
        # Rough estimates (very approximate)
        if params <= 0.5e9:  # 500M
            seconds_per_step = 1.0
            memory_gb = 8
        elif params <= 1e9:  # 1B
            seconds_per_step = 2.0
            memory_gb = 16
        elif params <= 2e9:  # 2B
            seconds_per_step = 4.0
            memory_gb = 24
        else:  # > 2B
            seconds_per_step = 8.0
            memory_gb = 32
        
        # Adjust for batch size
        seconds_per_step *= (batch_size / 32)
        
        total_seconds = steps * seconds_per_step
        hours = total_seconds / 3600
        
        estimates = {
            'total_steps': steps,
            'estimated_seconds_per_step': seconds_per_step,
            'estimated_total_hours': hours,
            'estimated_memory_gb': memory_gb,
            'parameters': params
        }
        
        if hours < 1:
            time_str = f"{hours * 60:.0f} minutes"
        elif hours < 24:
            time_str = f"{hours:.1f} hours"
        else:
            days = hours / 24
            time_str = f"{days:.1f} days"
        
        return time_str, estimates
    
    def run_training(self):
        """Execute the training process."""
        
        # Save configurations
        self._save_configs()
        
        # Print training info
        time_estimate, estimates = self._estimate_training_time()
        
        print("\n" + "="*60)
        print("🍋 LIMELLM TRAINING")
        print("="*60)
        print(f"Model Size: {self.model_config.total_parameters:,} parameters")
        print(f"Training Steps: {self.training_config.max_steps:,}")
        print(f"Batch Size: {self.training_config.batch_size}")
        print(f"Learning Rate: {self.training_config.learning_rate}")
        print(f"Estimated Time: {time_estimate}")
        print(f"Output Directory: {self.output_dir}")
        print("="*60)
        
        if self.args.dry_run:
            print("🔍 DRY RUN - Training configuration validated")
            print("\nEstimated resource requirements:")
            for key, value in estimates.items():
                print(f"  {key}: {value}")
            return
        
        # Confirm training start
        if not self.args.yes:
            response = input("\nStart training? [y/N]: ")
            if response.lower() not in ['y', 'yes']:
                print("Training cancelled")
                return
        
        logger.info("Starting LimeLLM training...")
        
        # Prepare training command
        if self.args.deepspeed or self.training_config.use_deepspeed:
            cmd = self._build_deepspeed_command()
        else:
            cmd = self._build_training_command()
        
        logger.info(f"Training command: {' '.join(cmd)}")
        
        # Execute training
        try:
            env = os.environ.copy()
            
            # Set training environment variables
            if self.training_config.wandb_project:
                env['WANDB_PROJECT'] = self.training_config.wandb_project
            
            # Run training
            process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output
            for line in process.stdout:
                print(line.rstrip())
                
            process.wait()
            
            if process.returncode == 0:
                logger.info("Training completed successfully!")
                self._post_training_tasks()
            else:
                logger.error(f"Training failed with return code: {process.returncode}")
                sys.exit(process.returncode)
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            if 'process' in locals():
                process.terminate()
            sys.exit(130)
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            sys.exit(1)
    
    def _build_training_command(self) -> List[str]:
        """Build training command for single-GPU or DDP."""
        
        cmd = [sys.executable, "training/train.py"]
        
        # Required arguments
        cmd.extend(["--train-data", self.args.train_data])
        cmd.extend(["--output-dir", str(self.output_dir)])
        
        # Optional arguments
        if self.args.eval_data:
            cmd.extend(["--eval-data", self.args.eval_data])
        
        if self.args.model_config:
            cmd.extend(["--model-config", self.args.model_config])
        
        if self.args.training_config:
            cmd.extend(["--training-config", self.args.training_config])
        
        if self.args.no_wandb:
            cmd.append("--no-wandb")
        
        if self.args.seed:
            cmd.extend(["--seed", str(self.args.seed)])
        
        return cmd
    
    def _build_deepspeed_command(self) -> List[str]:
        """Build DeepSpeed training command."""
        
        cmd = ["deepspeed"]
        
        # DeepSpeed configuration
        if self.args.deepspeed_config:
            cmd.extend(["--deepspeed", self.args.deepspeed_config])
        elif (self.project_root / "configs/deepspeed_config.json").exists():
            cmd.extend(["--deepspeed", "configs/deepspeed_config.json"])
        
        # Add training script
        cmd.append("training/train.py")
        
        # Add training arguments
        training_args = self._build_training_command()[1:]  # Skip python executable
        cmd.extend(training_args[1:])  # Skip script name
        
        return cmd
    
    def _post_training_tasks(self):
        """Perform post-training tasks."""
        logger.info("Performing post-training tasks...")
        
        # Find the final model checkpoint
        checkpoints = list(self.output_dir.glob("checkpoint-*"))
        if not checkpoints:
            logger.warning("No checkpoints found")
            return
        
        # Get the latest checkpoint
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split('-')[1]))
        logger.info(f"Latest checkpoint: {latest_checkpoint}")
        
        # Copy best model if available
        best_model_path = self.output_dir / "best_model.pt"
        if best_model_path.exists():
            final_model_path = self.output_dir / "final_best_model.pt"
            shutil.copy2(best_model_path, final_model_path)
            logger.info(f"Best model copied to: {final_model_path}")
        
        # Generate training summary
        self._generate_training_summary()
        
        # Optional: Start evaluation
        if self.args.auto_eval and self.args.eval_data:
            self._run_evaluation()
    
    def _generate_training_summary(self):
        """Generate a training summary report."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        summary = {
            'training_completed': True,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_hours': duration.total_seconds() / 3600,
            'model_parameters': self.model_config.total_parameters,
            'training_steps': self.training_config.max_steps,
            'output_directory': str(self.output_dir),
            'checkpoints': [str(p) for p in self.output_dir.glob("checkpoint-*")],
        }
        
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Training summary saved: {summary_path}")
        
        # Print summary
        print(f"\n🎉 Training completed in {duration}")
        print(f"📁 Model saved to: {self.output_dir}")
        print(f"📊 Training summary: {summary_path}")
    
    def _run_evaluation(self):
        """Run post-training evaluation."""
        logger.info("Starting post-training evaluation...")
        
        eval_cmd = [
            sys.executable,
            "evaluation/benchmarks/humaneval_runner.py",
            "--model-path", str(self.output_dir / "huggingface_model"),
            "--output-file", str(self.output_dir / "evaluation_results.json")
        ]
        
        try:
            subprocess.run(eval_cmd, cwd=str(self.project_root), check=True)
            logger.info("Evaluation completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Evaluation failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='Run LimeLLM training')
    
    # Data arguments
    parser.add_argument('--train-data', required=True, help='Training data directory')
    parser.add_argument('--eval-data', help='Evaluation data directory')
    
    # Model arguments
    parser.add_argument('--model-config', help='Model configuration file')
    parser.add_argument('--model-size', choices=['500m', '1b', '2b'], help='Predefined model size')
    
    # Training arguments
    parser.add_argument('--training-config', help='Training configuration file')
    parser.add_argument('--output-dir', default='outputs/limellm-training', help='Output directory')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--max-steps', type=int, help='Maximum training steps')
    
    # Distributed training
    parser.add_argument('--deepspeed', action='store_true', help='Use DeepSpeed for training')
    parser.add_argument('--deepspeed-config', help='DeepSpeed configuration file')
    
    # Monitoring
    parser.add_argument('--wandb-project', help='Weights & Biases project name')
    parser.add_argument('--no-wandb', action='store_true', help='Disable Weights & Biases logging')
    
    # Execution control
    parser.add_argument('--dry-run', action='store_true', help='Show configuration without training')
    parser.add_argument('--yes', action='store_true', help='Skip confirmation prompt')
    parser.add_argument('--auto-eval', action='store_true', help='Run evaluation after training')
    
    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Create and run training
    runner = TrainingRunner(args)
    runner.run_training()

if __name__ == "__main__":
    main()