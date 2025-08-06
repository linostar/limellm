import json
import os
from typing import List, Dict, Any
import argparse
import torch
import logging
from transformers import GPT2TokenizerFast

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.architecture import LimeLLMForCausalLM
from evaluation.code_eval import HumanEvalBenchmark

logger = logging.getLogger(__name__)

class HumanEvalRunner:
    """Runner for Human-Eval benchmark evaluation."""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        self.model_path = model_path
        self.device = self._setup_device(device)
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        
        # Initialize benchmark
        self.benchmark = HumanEvalBenchmark()
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup compute device."""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        return torch.device(device)
    
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer from checkpoint."""
        logger.info(f"Loading model from {self.model_path}")
        
        # Load tokenizer
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        if os.path.isdir(self.model_path):
            # HuggingFace format
            model = LimeLLMForCausalLM.from_pretrained(self.model_path)
        else:
            # PyTorch checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            from model.config import ModelConfig
            model_config = ModelConfig.from_dict(checkpoint['model_config'])
            model = LimeLLMForCausalLM(model_config)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Model parameters: {model.get_num_params():,}")
        
        return model, tokenizer
    
    def run_evaluation(
        self,
        problems_file: str = None,
        output_file: str = None,
        max_length: int = 512,
        temperature: float = 0.2,
        num_samples: int = 1,
        batch_size: int = 1
    ) -> Dict[str, Any]:
        """Run Human-Eval evaluation."""
        
        # Load problems
        if problems_file and os.path.exists(problems_file):
            problems = self.benchmark.load_humaneval_problems(problems_file)
            logger.info(f"Loaded {len(problems)} problems from {problems_file}")
        else:
            logger.warning("Using example problems (Human-Eval dataset not found)")
            problems = self.benchmark._create_example_problems()
        
        # Run evaluation
        results = self.benchmark.evaluate_model_on_humaneval(
            model=self.model,
            tokenizer=self.tokenizer,
            problems=problems,
            max_length=max_length,
            temperature=temperature,
            num_samples=num_samples
        )
        
        # Add metadata
        results['evaluation_config'] = {
            'model_path': self.model_path,
            'max_length': max_length,
            'temperature': temperature,
            'num_samples': num_samples,
            'device': str(self.device),
            'problems_file': problems_file
        }
        
        # Save results
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Results saved to {output_file}")
        
        return results
    
    def generate_sample_completions(
        self,
        prompts: List[str],
        max_length: int = 256,
        temperature: float = 0.7,
        num_samples: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate sample completions for analysis."""
        
        results = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Generating completions for prompt {i+1}/{len(prompts)}")
            
            # Generate completions
            completions = self.benchmark._generate_completions(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                num_samples=num_samples
            )
            
            results.append({
                'prompt': prompt,
                'completions': completions
            })
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Run Human-Eval benchmark on LimeLLM')
    parser.add_argument('--model-path', required=True, help='Path to model checkpoint or directory')
    parser.add_argument('--problems-file', help='Path to Human-Eval problems JSONL file')
    parser.add_argument('--output-file', help='Path to save results JSON file')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=0.2, help='Generation temperature')
    parser.add_argument('--num-samples', type=int, default=1, help='Number of samples per problem')
    parser.add_argument('--device', default='auto', help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize runner
    runner = HumanEvalRunner(
        model_path=args.model_path,
        device=args.device
    )
    
    # Run evaluation
    try:
        results = runner.run_evaluation(
            problems_file=args.problems_file,
            output_file=args.output_file,
            max_length=args.max_length,
            temperature=args.temperature,
            num_samples=args.num_samples
        )
        
        # Print summary
        print("\n" + "="*50)
        print("HUMAN-EVAL EVALUATION RESULTS")
        print("="*50)
        print(f"Total problems: {results['total_problems']}")
        print(f"Problems solved: {results['problems_solved']}")
        print(f"Pass@1 score: {results['pass_at_1']:.2%}")
        print("="*50)
        
        # Show some examples
        if args.verbose and results['detailed_results']:
            print("\nDetailed Results (first 3):")
            for i, result in enumerate(results['detailed_results'][:3]):
                print(f"\nTask {result.task_id}:")
                print(f"  Passed: {result.passed}")
                print(f"  Syntax valid: {result.syntax_valid}")
                if result.error:
                    print(f"  Error: {result.error}")
                if hasattr(result, 'completion'):
                    print(f"  Generated code preview: {result.completion[:100]}...")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()