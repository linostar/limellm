import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json
import logging
import wandb
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import argparse
from transformers import GPT2TokenizerFast, get_cosine_schedule_with_warmup
import numpy as np
import random
from contextlib import nullcontext

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig, TrainingConfig
from model.architecture import LimeLLMForCausalLM
from training.data_loader import create_dataloaders
from training.distributed_trainer import DistributedTrainer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class LimeLLMTrainer:
    def __init__(
        self, 
        model_config: ModelConfig, 
        training_config: TrainingConfig,
        tokenizer: GPT2TokenizerFast,
        device: torch.device
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.tokenizer = tokenizer
        self.device = device
        
        # Initialize model
        self.model = LimeLLMForCausalLM(model_config)
        self.model.to(device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Mixed precision
        self.use_amp = training_config.fp16 or training_config.bf16
        self.scaler = torch.cuda.amp.GradScaler() if training_config.fp16 else None
        self.autocast_dtype = torch.bfloat16 if training_config.bf16 else torch.float16
        
        # Gradient accumulation
        self.gradient_accumulation_steps = training_config.gradient_accumulation_steps
        
        # Logging
        self.use_wandb = bool(training_config.wandb_project)
        if self.use_wandb:
            wandb.init(
                project=training_config.wandb_project,
                name=training_config.wandb_run_name,
                config={
                    **model_config.to_dict(),
                    **training_config.to_dict()
                }
            )
            wandb.watch(self.model, log='all', log_freq=training_config.logging_steps)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay."""
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Don't apply weight decay to biases and layer norms
                if any(nd in name for nd in ['bias', 'layer_norm', 'ln_']):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        param_groups = [
            {
                'params': decay_params,
                'weight_decay': self.training_config.weight_decay
            },
            {
                'params': no_decay_params,
                'weight_decay': 0.0
            }
        ]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.training_config.learning_rate,
            betas=(self.training_config.beta1, self.training_config.beta2),
            eps=self.training_config.epsilon
        )
        
        return optimizer
    
    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler."""
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.training_config.warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=0.5
        )
    
    def _save_checkpoint(self, save_path: str, is_best: bool = False):
        """Save training checkpoint."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_eval_loss': self.best_eval_loss,
            'model_config': self.model_config.to_dict(),
            'training_config': self.training_config.to_dict()
        }
        
        torch.save(checkpoint, save_path)
        
        if is_best:
            best_path = os.path.join(os.path.dirname(save_path), 'best_model.pt')
            torch.save(checkpoint, best_path)
        
        logger.info(f"Checkpoint saved: {save_path}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if checkpoint['scaler_state_dict'] and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_eval_loss = checkpoint['best_eval_loss']
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.autocast_dtype):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs['loss']
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return {'loss': loss.item() * self.gradient_accumulation_steps}
    
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single evaluation step."""
        self.model.eval()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.autocast_dtype):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs['loss']
        
        return {'loss': loss.item()}
    
    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        logger.info("Running evaluation...")
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                metrics = self.eval_step(batch)
                total_loss += metrics['loss']
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        eval_metrics = {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity
        }
        
        logger.info(f"Evaluation results: Loss={avg_loss:.4f}, Perplexity={perplexity:.2f}")
        
        return eval_metrics
    
    def train(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None):
        """Main training loop."""
        
        # Calculate total training steps
        num_training_steps = self.training_config.max_steps
        self._create_scheduler(num_training_steps)
        
        # Load checkpoint if resuming
        if self.training_config.resume_from_checkpoint:
            self._load_checkpoint(self.training_config.resume_from_checkpoint)
        
        logger.info(f"Starting training for {num_training_steps} steps")
        logger.info(f"Model parameters: {self.model.get_num_params():,}")
        
        # Training loop
        self.model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(total=num_training_steps, initial=self.global_step, desc="Training")
        
        while self.global_step < num_training_steps:
            for batch in train_dataloader:
                if self.global_step >= num_training_steps:
                    break
                
                # Training step
                metrics = self.train_step(batch)
                running_loss += metrics['loss']
                
                # Gradient accumulation and optimization
                if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                    # Clip gradients
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.training_config.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.training_config.max_grad_norm
                        )
                        self.optimizer.step()
                    
                    # Update learning rate
                    if self.scheduler:
                        self.scheduler.step()
                    
                    # Zero gradients
                    self.optimizer.zero_grad()
                
                self.global_step += 1
                progress_bar.update(1)
                
                # Logging
                if self.global_step % self.training_config.logging_steps == 0:
                    avg_loss = running_loss / self.training_config.logging_steps
                    current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.training_config.learning_rate
                    
                    log_metrics = {
                        'train_loss': avg_loss,
                        'learning_rate': current_lr,
                        'global_step': self.global_step,
                        'epoch': self.epoch
                    }
                    
                    logger.info(f"Step {self.global_step}: Loss={avg_loss:.4f}, LR={current_lr:.2e}")
                    
                    if self.use_wandb:
                        wandb.log(log_metrics, step=self.global_step)
                    
                    running_loss = 0.0
                
                # Evaluation
                if eval_dataloader and self.global_step % self.training_config.eval_steps == 0:
                    eval_metrics = self.evaluate(eval_dataloader)
                    
                    if self.use_wandb:
                        wandb.log(eval_metrics, step=self.global_step)
                    
                    # Save best model
                    is_best = eval_metrics['eval_loss'] < self.best_eval_loss
                    if is_best:
                        self.best_eval_loss = eval_metrics['eval_loss']
                    
                    # Resume training
                    self.model.train()
                
                # Save checkpoint
                if self.global_step % self.training_config.save_steps == 0:
                    save_path = os.path.join(
                        self.training_config.output_dir, 
                        f"checkpoint-{self.global_step}.pt"
                    )
                    is_best = eval_dataloader is None or eval_metrics.get('eval_loss', float('inf')) < self.best_eval_loss
                    self._save_checkpoint(save_path, is_best=is_best)
        
        progress_bar.close()
        
        # Final save
        final_save_path = os.path.join(self.training_config.output_dir, "final_model.pt")
        self._save_checkpoint(final_save_path)
        
        # Save HuggingFace format
        hf_save_path = os.path.join(self.training_config.output_dir, "huggingface_model")
        self.model.save_pretrained(hf_save_path)
        self.tokenizer.save_pretrained(hf_save_path)
        
        logger.info("Training completed!")

def main():
    parser = argparse.ArgumentParser(description='Train LimeLLM')
    parser.add_argument('--model-config', type=str, help='Path to model config JSON')
    parser.add_argument('--training-config', type=str, help='Path to training config JSON')
    parser.add_argument('--train-data', type=str, required=True, help='Path to training data')
    parser.add_argument('--eval-data', type=str, help='Path to evaluation data')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load configs
    if args.model_config:
        model_config = ModelConfig.from_json(args.model_config)
    else:
        model_config = ModelConfig()  # Use default 2B config
    
    if args.training_config:
        training_config = TrainingConfig.from_json(args.training_config)
    else:
        training_config = TrainingConfig()
    
    # Override output dir
    training_config.output_dir = args.output_dir
    
    # Disable wandb if requested
    if args.no_wandb:
        training_config.wandb_project = None
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Update vocab size if needed
    if len(tokenizer) != model_config.vocab_size:
        logger.info(f"Updating vocab size from {model_config.vocab_size} to {len(tokenizer)}")
        model_config.vocab_size = len(tokenizer)
    
    # Create dataloaders
    train_dataloader, eval_dataloader = create_dataloaders(
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
        tokenizer=tokenizer,
        batch_size=training_config.batch_size,
        max_length=training_config.max_length
    )
    
    # Initialize trainer
    trainer = LimeLLMTrainer(
        model_config=model_config,
        training_config=training_config,
        tokenizer=tokenizer,
        device=device
    )
    
    # Start training
    trainer.train(train_dataloader, eval_dataloader)

if __name__ == "__main__":
    main()