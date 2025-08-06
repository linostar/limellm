import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
import logging
from typing import Dict, Optional, Any
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from model.config import ModelConfig, TrainingConfig
from model.architecture import LimeLLMForCausalLM

logger = logging.getLogger(__name__)

class DistributedTrainer:
    """Distributed training wrapper for LimeLLM using DeepSpeed and DDP."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        tokenizer,
        local_rank: int = -1
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.tokenizer = tokenizer
        self.local_rank = local_rank
        
        # Initialize distributed training
        self._init_distributed()
        
        # Initialize model
        self.model = LimeLLMForCausalLM(model_config)
        
        # Initialize DeepSpeed or DDP
        if training_config.use_deepspeed:
            self._init_deepspeed()
        elif training_config.use_ddp:
            self._init_ddp()
        
        # Training state
        self.global_step = 0
        self.best_eval_loss = float('inf')
    
    def _init_distributed(self):
        """Initialize distributed training environment."""
        if self.local_rank == -1:
            # Not using distributed training
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return
        
        # Initialize process group
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f'cuda:{self.local_rank}')
        
        dist.init_process_group(backend='nccl')
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        logger.info(f"Initialized distributed training: rank={self.rank}, world_size={self.world_size}, local_rank={self.local_rank}")
    
    def _init_deepspeed(self):
        """Initialize DeepSpeed."""
        # DeepSpeed config
        ds_config = self._get_deepspeed_config()
        
        # Initialize DeepSpeed
        self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.model,
            config=ds_config,
            model_parameters=self.model.parameters()
        )
        
        self.use_deepspeed = True
        logger.info("DeepSpeed initialized successfully")
    
    def _init_ddp(self):
        """Initialize DistributedDataParallel."""
        self.model = self.model.to(self.device)
        
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
        
        # Create optimizer manually for DDP
        self.optimizer = self._create_optimizer()
        self.scheduler = None
        self.use_deepspeed = False
        
        logger.info("DDP initialized successfully")
    
    def _get_deepspeed_config(self) -> Dict[str, Any]:
        """Generate DeepSpeed configuration."""
        config = {
            "train_batch_size": self.training_config.batch_size * self.training_config.gradient_accumulation_steps * self.world_size,
            "train_micro_batch_size_per_gpu": self.training_config.batch_size,
            "gradient_accumulation_steps": self.training_config.gradient_accumulation_steps,
            "steps_per_print": self.training_config.logging_steps,
            "wall_clock_breakdown": False,
            
            # Mixed precision
            "fp16": {
                "enabled": self.training_config.fp16,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1
            } if self.training_config.fp16 else {"enabled": False},
            
            "bf16": {
                "enabled": self.training_config.bf16
            } if self.training_config.bf16 else {"enabled": False},
            
            # Optimizer
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.training_config.learning_rate,
                    "betas": [self.training_config.beta1, self.training_config.beta2],
                    "eps": self.training_config.epsilon,
                    "weight_decay": self.training_config.weight_decay
                }
            },
            
            # Learning rate scheduler
            "scheduler": {
                "type": "WarmupCosineLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": self.training_config.learning_rate,
                    "warmup_num_steps": self.training_config.warmup_steps,
                    "total_num_steps": self.training_config.max_steps
                }
            },
            
            # Gradient clipping
            "gradient_clipping": self.training_config.max_grad_norm,
            
            # Zero optimization
            "zero_optimization": {
                "stage": 2,  # ZeRO stage 2 for 2B model
                "allgather_partitions": True,
                "allgather_bucket_size": 200000000,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 200000000,
                "contiguous_gradients": True,
                "cpu_offload": False  # Keep on GPU for 2B model
            },
            
            # Activation checkpointing
            "activation_checkpointing": {
                "partition_activations": True,
                "cpu_checkpointing": False,
                "contiguous_memory_optimization": False,
                "number_checkpoints": None,
                "synchronize_checkpoint_boundary": False,
                "profile": False
            } if self.model_config.gradient_checkpointing else None,
            
            # Communication backend
            "comms_logger": {
                "enabled": False
            }
        }
        
        # Remove None values
        config = {k: v for k, v in config.items() if v is not None}
        
        return config
    
    def _create_optimizer(self):
        """Create optimizer for DDP training."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in ['bias', 'layer_norm', 'ln_']):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': self.training_config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        return torch.optim.AdamW(
            param_groups,
            lr=self.training_config.learning_rate,
            betas=(self.training_config.beta1, self.training_config.beta2),
            eps=self.training_config.epsilon
        )
    
    def create_distributed_dataloader(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        collate_fn=None
    ) -> DataLoader:
        """Create distributed dataloader."""
        
        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle
            )
            shuffle = False  # Sampler handles shuffling
        else:
            sampler = None
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            sampler=sampler,
            drop_last=True
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with DeepSpeed or DDP."""
        
        if self.use_deepspeed:
            return self._deepspeed_train_step(batch)
        else:
            return self._ddp_train_step(batch)
    
    def _deepspeed_train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Training step with DeepSpeed."""
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs['loss']
        
        # Backward pass (DeepSpeed handles everything)
        self.model.backward(loss)
        self.model.step()
        
        return {'loss': loss.item()}
    
    def _ddp_train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Training step with DDP."""
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs['loss']
        
        # Scale loss for gradient accumulation
        loss = loss / self.training_config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        return {'loss': loss.item() * self.training_config.gradient_accumulation_steps}
    
    def optimizer_step(self):
        """Optimizer step for DDP (DeepSpeed handles this automatically)."""
        if not self.use_deepspeed:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.training_config.max_grad_norm
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Zero gradients
            self.optimizer.zero_grad()
    
    def save_checkpoint(self, save_path: str):
        """Save distributed checkpoint."""
        if self.use_deepspeed:
            # DeepSpeed checkpoint
            self.model.save_checkpoint(save_path)
        else:
            # DDP checkpoint
            if self.rank == 0:  # Only save from rank 0
                model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
                
                checkpoint = {
                    'model_state_dict': model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'global_step': self.global_step,
                    'model_config': self.model_config.to_dict(),
                    'training_config': self.training_config.to_dict()
                }
                
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(checkpoint, save_path)
        
        # Synchronize all processes
        if self.world_size > 1:
            dist.barrier()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load distributed checkpoint."""
        if self.use_deepspeed:
            # DeepSpeed checkpoint loading
            _, client_state = self.model.load_checkpoint(checkpoint_path)
            if client_state:
                self.global_step = client_state.get('global_step', 0)
        else:
            # DDP checkpoint loading
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            model = self.model.module if hasattr(self.model, 'module') else self.model
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.global_step = checkpoint.get('global_step', 0)
    
    def cleanup(self):
        """Clean up distributed training."""
        if self.world_size > 1:
            dist.destroy_process_group()

def setup_distributed_training():
    """Setup distributed training environment variables."""
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    
    if local_rank != -1:
        # Running with torch.distributed.launch or deepspeed
        torch.cuda.set_device(local_rank)
        
        # Set environment variables for distributed training
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = '127.0.0.1'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '12355'
    
    return local_rank

if __name__ == "__main__":
    # Example usage
    from model.config import ModelConfig, TrainingConfig
    from transformers import GPT2TokenizerFast
    
    # Setup
    local_rank = setup_distributed_training()
    
    model_config = ModelConfig()
    training_config = TrainingConfig(use_deepspeed=True)
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    # Initialize distributed trainer
    trainer = DistributedTrainer(
        model_config=model_config,
        training_config=training_config,
        tokenizer=tokenizer,
        local_rank=local_rank
    )
    
    print(f"Distributed trainer initialized on device: {trainer.device}")
    print(f"World size: {trainer.world_size}, Rank: {trainer.rank}")
    print(f"Using DeepSpeed: {trainer.use_deepspeed}")
    
    # Cleanup
    trainer.cleanup()