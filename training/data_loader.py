import torch
import json
import os
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Dict, List, Iterator, Optional, Any
import logging
from transformers import PreTrainedTokenizerFast
import random
from datasets import load_dataset
import numpy as np

logger = logging.getLogger(__name__)

class CodeTextDataset(Dataset):
    """Dataset for loading and processing code and text data."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 4096,
        min_length: int = 10,
        data_mixing_ratios: Optional[Dict[str, float]] = None
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        self.data_mixing_ratios = data_mixing_ratios or {}
        
        # Load and process data
        self.samples = self._load_and_process_data()
        
        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def _load_and_process_data(self) -> List[Dict[str, Any]]:
        """Load and process all data files."""
        samples = []
        
        if os.path.isfile(self.data_path) and self.data_path.endswith('.jsonl'):
            # Single JSONL file
            samples.extend(self._load_jsonl_file(self.data_path))
        
        elif os.path.isdir(self.data_path):
            # Directory containing multiple data files
            for root, dirs, files in os.walk(self.data_path):
                for file in files:
                    if file.endswith('.jsonl'):
                        file_path = os.path.join(root, file)
                        file_samples = self._load_jsonl_file(file_path)
                        
                        # Apply data mixing ratios if specified
                        source_type = self._get_source_type(file)
                        if source_type in self.data_mixing_ratios:
                            ratio = self.data_mixing_ratios[source_type]
                            target_count = int(len(file_samples) * ratio)
                            if target_count < len(file_samples):
                                file_samples = random.sample(file_samples, target_count)
                        
                        samples.extend(file_samples)
                        logger.info(f"Loaded {len(file_samples)} samples from {file}")
        
        else:
            raise ValueError(f"Data path {self.data_path} is neither a file nor a directory")
        
        # Shuffle samples
        random.shuffle(samples)
        return samples
    
    def _get_source_type(self, filename: str) -> str:
        """Determine source type from filename."""
        if 'github' in filename.lower():
            return 'github'
        elif 'stackoverflow' in filename.lower():
            return 'stackoverflow'
        elif 'python_docs' in filename.lower():
            return 'python_docs'
        elif 'pypi' in filename.lower():
            return 'pypi'
        else:
            return 'unknown'
    
    def _load_jsonl_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load samples from a JSONL file."""
        samples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    if line.strip():
                        item = json.loads(line)
                        processed_samples = self._process_item(item)
                        samples.extend(processed_samples)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num} in {file_path}: {e}")
                except Exception as e:
                    logger.warning(f"Error processing line {line_num} in {file_path}: {e}")
        
        return samples
    
    def _process_item(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a single data item into training samples."""
        samples = []
        
        # Extract text content
        text_content = self._extract_text_content(item)
        
        if text_content and len(text_content.strip()) >= self.min_length:
            # Create training sample
            sample = {
                'text': text_content,
                'source': item.get('source', 'unknown'),
                'metadata': {
                    'original_item_id': item.get('id'),
                    'title': item.get('title', ''),
                    'url': item.get('url', ''),
                }
            }
            samples.append(sample)
        
        # For repositories with multiple files, process each file separately
        if 'files' in item:
            for file_info in item['files']:
                if file_info.get('type') == 'code' and file_info.get('content'):
                    content = file_info['content']
                    if len(content.strip()) >= self.min_length:
                        file_sample = {
                            'text': self._format_code_content(content, file_info.get('path', '')),
                            'source': 'github_code',
                            'metadata': {
                                'repo_name': item.get('repo_name', ''),
                                'file_path': file_info.get('path', ''),
                                'repo_stars': item.get('stars', 0),
                            }
                        }
                        samples.append(file_sample)
        
        # For Stack Overflow Q&A, create formatted samples
        if 'answers' in item:
            qa_text = self._format_stackoverflow_qa(item)
            if qa_text and len(qa_text.strip()) >= self.min_length:
                qa_sample = {
                    'text': qa_text,
                    'source': 'stackoverflow_qa',
                    'metadata': {
                        'question_id': item.get('question_id'),
                        'question_score': item.get('question_score', 0),
                        'tags': item.get('tags', []),
                    }
                }
                samples.append(qa_sample)
        
        return samples
    
    def _extract_text_content(self, item: Dict[str, Any]) -> str:
        """Extract main text content from an item."""
        # Try different field names for content
        content_fields = ['content', 'body', 'description', 'text']
        
        for field in content_fields:
            if field in item and item[field]:
                return str(item[field]).strip()
        
        return ""
    
    def _format_code_content(self, content: str, file_path: str) -> str:
        """Format code content with context."""
        # Add file path as comment for context
        if file_path and not file_path.startswith('#'):
            formatted = f"# File: {file_path}\n{content}"
        else:
            formatted = content
        
        # Ensure proper formatting
        return formatted.strip()
    
    def _format_stackoverflow_qa(self, item: Dict[str, Any]) -> str:
        """Format Stack Overflow Q&A into training text."""
        title = item.get('title', '')
        question = item.get('question_body', '')
        answers = item.get('answers', [])
        
        if not question or not answers:
            return ""
        
        # Format as: Title + Question + Best Answer
        formatted = f"Question: {title}\n\n{question}\n\n"
        
        # Find the best answer (accepted or highest scored)
        best_answer = None
        for answer in answers:
            if answer.get('is_accepted'):
                best_answer = answer
                break
            elif not best_answer or answer.get('score', 0) > best_answer.get('score', 0):
                best_answer = answer
        
        if best_answer:
            formatted += f"Answer:\n{best_answer.get('body', '')}"
        
        return formatted.strip()
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            sample['text'],
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0).clone(),  # For language modeling
            'source': sample['source'],
            'metadata': sample['metadata']
        }

class StreamingCodeTextDataset(IterableDataset):
    """Streaming dataset for very large datasets that don't fit in memory."""
    
    def __init__(
        self,
        data_paths: List[str],
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 4096,
        buffer_size: int = 10000,
        shuffle_buffer: bool = True
    ):
        self.data_paths = data_paths
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.shuffle_buffer = shuffle_buffer
    
    def _sample_generator(self) -> Iterator[Dict[str, Any]]:
        """Generate samples from all data files."""
        dataset = CodeTextDataset(
            data_path=self.data_paths[0],  # Simplified for demo
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        indices = list(range(len(dataset)))
        if self.shuffle_buffer:
            random.shuffle(indices)
        
        for idx in indices:
            yield dataset[idx]
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return self._sample_generator()

class DataCollatorForLanguageModeling:
    """Data collator for language modeling with dynamic padding."""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 4096,
        pad_to_multiple_of: Optional[int] = 8,
        return_tensors: str = "pt"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract input_ids and labels
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # Pad sequences
        max_len = min(max(len(seq) for seq in input_ids), self.max_length)
        
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        padded_input_ids = []
        padded_labels = []
        attention_masks = []
        
        for input_seq, label_seq in zip(input_ids, labels):
            seq_len = len(input_seq)
            
            if seq_len > max_len:
                # Truncate
                input_seq = input_seq[:max_len]
                label_seq = label_seq[:max_len]
                seq_len = max_len
            
            # Pad
            pad_length = max_len - seq_len
            if pad_length > 0:
                pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                
                input_seq = torch.cat([input_seq, torch.full((pad_length,), pad_token_id, dtype=input_seq.dtype)])
                label_seq = torch.cat([label_seq, torch.full((pad_length,), -100, dtype=label_seq.dtype)])  # -100 is ignored by loss
            
            # Attention mask (1 for real tokens, 0 for padding)
            attention_mask = torch.cat([
                torch.ones(seq_len, dtype=torch.bool),
                torch.zeros(pad_length, dtype=torch.bool)
            ])
            
            padded_input_ids.append(input_seq)
            padded_labels.append(label_seq)
            attention_masks.append(attention_mask)
        
        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(attention_masks),
            'labels': torch.stack(padded_labels)
        }

def create_dataloaders(
    train_data_path: str,
    eval_data_path: str,
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int = 8,
    max_length: int = 4096,
    num_workers: int = 4,
    data_mixing_ratios: Optional[Dict[str, float]] = None
) -> tuple[DataLoader, DataLoader]:
    """Create train and eval dataloaders."""
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        max_length=max_length,
        pad_to_multiple_of=8
    )
    
    # Training dataset
    train_dataset = CodeTextDataset(
        data_path=train_data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        data_mixing_ratios=data_mixing_ratios
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=data_collator,
        pin_memory=True,
        drop_last=True
    )
    
    # Evaluation dataset
    eval_dataset = CodeTextDataset(
        data_path=eval_data_path,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=data_collator,
        pin_memory=True,
        drop_last=False
    )
    
    logger.info(f"Created dataloaders - Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")
    
    return train_dataloader, eval_dataloader

if __name__ == "__main__":
    # Test the dataset
    from transformers import GPT2TokenizerFast
    
    # Create a dummy tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dummy data
    os.makedirs('test_data', exist_ok=True)
    with open('test_data/test.jsonl', 'w') as f:
        f.write('{"content": "def hello():\\n    print(\\"Hello, World!\\")\\n    return True", "source": "test"}\\n')
        f.write('{"content": "import numpy as np\\ndata = np.array([1, 2, 3])", "source": "test"}\\n')
    
    # Test dataset
    dataset = CodeTextDataset(
        data_path='test_data/test.jsonl',
        tokenizer=tokenizer,
        max_length=512
    )
    
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Input shape: {sample['input_ids'].shape}")
    print(f"Decoded text: {tokenizer.decode(sample['input_ids'][:50])}")
    
    # Test dataloader
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=data_collator)
    
    batch = next(iter(dataloader))
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")
    
    # Cleanup
    import shutil
    shutil.rmtree('test_data')