import torch
import ast
import sys
import subprocess
import tempfile
import os
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time
import re
import difflib

logger = logging.getLogger(__name__)

@dataclass
class CodeEvaluationResult:
    """Result of code evaluation."""
    task_id: str
    generated_code: str
    expected_output: Optional[str] = None
    actual_output: Optional[str] = None
    passed: bool = False
    error: Optional[str] = None
    execution_time: float = 0.0
    syntax_valid: bool = False
    test_cases_passed: int = 0
    total_test_cases: int = 0

class PythonCodeEvaluator:
    """Evaluator for Python code generation and refactoring tasks."""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def evaluate_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Check if code has valid Python syntax."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
    
    def execute_code_safely(self, code: str, test_input: str = "") -> Tuple[Optional[str], Optional[str], float]:
        """Execute code safely in a subprocess with timeout."""
        start_time = time.time()
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Execute with timeout
                process = subprocess.Popen(
                    [sys.executable, temp_file],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout
                )
                
                stdout, stderr = process.communicate(input=test_input, timeout=self.timeout)
                execution_time = time.time() - start_time
                
                if process.returncode == 0:
                    return stdout.strip(), None, execution_time
                else:
                    return None, stderr.strip(), execution_time
                    
            finally:
                # Clean up temporary file
                os.unlink(temp_file)
                
        except subprocess.TimeoutExpired:
            return None, f"Execution timeout ({self.timeout}s)", time.time() - start_time
        except Exception as e:
            return None, str(e), time.time() - start_time
    
    def evaluate_single_task(self, task: Dict[str, Any]) -> CodeEvaluationResult:
        """Evaluate a single code generation/refactoring task."""
        task_id = task.get('task_id', 'unknown')
        generated_code = task.get('generated_code', '')
        expected_output = task.get('expected_output')
        test_cases = task.get('test_cases', [])
        
        result = CodeEvaluationResult(
            task_id=task_id,
            generated_code=generated_code
        )
        
        # Check syntax
        syntax_valid, syntax_error = self.evaluate_syntax(generated_code)
        result.syntax_valid = syntax_valid
        
        if not syntax_valid:
            result.error = f"Syntax error: {syntax_error}"
            return result
        
        # Execute code and run test cases
        if test_cases:
            passed_tests = 0
            total_tests = len(test_cases)
            
            for i, test_case in enumerate(test_cases):
                test_input = test_case.get('input', '')
                expected = test_case.get('expected', '')
                
                # Execute code with test input
                output, error, exec_time = self.execute_code_safely(generated_code, test_input)
                result.execution_time += exec_time
                
                if error:
                    result.error = f"Test case {i+1} failed: {error}"
                    break
                
                # Compare output
                if output is not None and self._compare_outputs(output, expected):
                    passed_tests += 1
            
            result.test_cases_passed = passed_tests
            result.total_test_cases = total_tests
            result.passed = passed_tests == total_tests
            
        elif expected_output is not None:
            # Single expected output
            output, error, exec_time = self.execute_code_safely(generated_code)
            result.execution_time = exec_time
            result.actual_output = output
            result.expected_output = expected_output
            
            if error:
                result.error = error
            else:
                result.passed = self._compare_outputs(output, expected_output)
        
        else:
            # Just check if code runs without error
            output, error, exec_time = self.execute_code_safely(generated_code)
            result.execution_time = exec_time
            result.actual_output = output
            result.passed = error is None
            if error:
                result.error = error
        
        return result
    
    def _compare_outputs(self, actual: str, expected: str) -> bool:
        """Compare actual and expected outputs with some tolerance."""
        if actual is None:
            return False
        
        # Normalize whitespace
        actual = actual.strip()
        expected = expected.strip()
        
        # Exact match
        if actual == expected:
            return True
        
        # Try to parse as numbers and compare with tolerance
        try:
            actual_num = float(actual)
            expected_num = float(expected)
            return abs(actual_num - expected_num) < 1e-6
        except ValueError:
            pass
        
        # Try line-by-line comparison (ignoring extra whitespace)
        actual_lines = [line.strip() for line in actual.split('\n') if line.strip()]
        expected_lines = [line.strip() for line in expected.split('\n') if line.strip()]
        
        return actual_lines == expected_lines

class CodeRefactoringEvaluator:
    """Evaluator for code refactoring tasks."""
    
    def __init__(self):
        self.code_evaluator = PythonCodeEvaluator()
    
    def evaluate_refactoring(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a code refactoring task."""
        original_code = task.get('original_code', '')
        refactored_code = task.get('refactored_code', '')
        test_cases = task.get('test_cases', [])
        
        result = {
            'task_id': task.get('task_id', 'unknown'),
            'original_code': original_code,
            'refactored_code': refactored_code,
            'syntax_valid': False,
            'functionality_preserved': False,
            'improvements': {},
            'error': None
        }
        
        # Check syntax of refactored code
        syntax_valid, syntax_error = self.code_evaluator.evaluate_syntax(refactored_code)
        result['syntax_valid'] = syntax_valid
        
        if not syntax_valid:
            result['error'] = f"Syntax error in refactored code: {syntax_error}"
            return result
        
        # Test functionality preservation
        if test_cases:
            original_results = []
            refactored_results = []
            
            for test_case in test_cases:
                test_input = test_case.get('input', '')
                
                # Test original code
                orig_output, orig_error, _ = self.code_evaluator.execute_code_safely(original_code, test_input)
                original_results.append((orig_output, orig_error))
                
                # Test refactored code
                ref_output, ref_error, _ = self.code_evaluator.execute_code_safely(refactored_code, test_input)
                refactored_results.append((ref_output, ref_error))
            
            # Check if results are equivalent
            functionality_preserved = True
            for (orig_out, orig_err), (ref_out, ref_err) in zip(original_results, refactored_results):
                if orig_err is None and ref_err is None:
                    # Both succeeded, compare outputs
                    if not self.code_evaluator._compare_outputs(orig_out, ref_out):
                        functionality_preserved = False
                        break
                elif orig_err is not None and ref_err is not None:
                    # Both failed, that's okay for refactoring eval
                    continue
                else:
                    # One succeeded, one failed
                    functionality_preserved = False
                    break
            
            result['functionality_preserved'] = functionality_preserved
        
        # Analyze improvements
        result['improvements'] = self._analyze_improvements(original_code, refactored_code)
        
        return result
    
    def _analyze_improvements(self, original: str, refactored: str) -> Dict[str, Any]:
        """Analyze improvements in refactored code."""
        improvements = {
            'length_reduction': 0,
            'complexity_reduction': False,
            'readability_improved': False,
            'added_comments': False,
            'added_type_hints': False,
            'removed_duplicates': False
        }
        
        # Length comparison
        orig_lines = len([l for l in original.split('\n') if l.strip()])
        ref_lines = len([l for l in refactored.split('\n') if l.strip()])
        improvements['length_reduction'] = orig_lines - ref_lines
        
        # Check for added comments
        orig_comments = len(re.findall(r'#.*', original))
        ref_comments = len(re.findall(r'#.*', refactored))
        improvements['added_comments'] = ref_comments > orig_comments
        
        # Check for type hints
        orig_hints = len(re.findall(r':\s*\w+', original))
        ref_hints = len(re.findall(r':\s*\w+', refactored))
        improvements['added_type_hints'] = ref_hints > orig_hints
        
        # Basic complexity metrics
        orig_complexity = self._calculate_complexity(original)
        ref_complexity = self._calculate_complexity(refactored)
        improvements['complexity_reduction'] = ref_complexity < orig_complexity
        
        return improvements
    
    def _calculate_complexity(self, code: str) -> int:
        """Simple complexity calculation based on control flow statements."""
        complexity = 1  # Base complexity
        
        # Count control flow statements
        control_flow_patterns = [
            r'\bif\b', r'\belif\b', r'\belse\b',
            r'\bfor\b', r'\bwhile\b',
            r'\btry\b', r'\bexcept\b', r'\bfinally\b',
            r'\bwith\b'
        ]
        
        for pattern in control_flow_patterns:
            complexity += len(re.findall(pattern, code))
        
        return complexity

class HumanEvalBenchmark:
    """Human-Eval benchmark for code generation evaluation."""
    
    def __init__(self):
        self.evaluator = PythonCodeEvaluator()
    
    def load_humaneval_problems(self, file_path: str) -> List[Dict[str, Any]]:
        """Load Human-Eval problems from JSONL file."""
        problems = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        problem = json.loads(line)
                        problems.append(problem)
        except FileNotFoundError:
            logger.warning(f"Human-Eval file not found: {file_path}")
            # Create some example problems for demonstration
            problems = self._create_example_problems()
        
        return problems
    
    def _create_example_problems(self) -> List[Dict[str, Any]]:
        """Create example problems for testing."""
        return [
            {
                "task_id": "test_1",
                "prompt": "def add_two_numbers(a, b):\n    \"\"\"Add two numbers and return the result.\"\"\"\n    ",
                "canonical_solution": "return a + b",
                "test": "assert add_two_numbers(2, 3) == 5\nassert add_two_numbers(-1, 1) == 0\nassert add_two_numbers(0, 0) == 0",
                "entry_point": "add_two_numbers"
            },
            {
                "task_id": "test_2",
                "prompt": "def factorial(n):\n    \"\"\"Calculate factorial of n.\"\"\"\n    ",
                "canonical_solution": "if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
                "test": "assert factorial(0) == 1\nassert factorial(1) == 1\nassert factorial(5) == 120\nassert factorial(3) == 6",
                "entry_point": "factorial"
            }
        ]
    
    def evaluate_model_on_humaneval(
        self, 
        model, 
        tokenizer, 
        problems: List[Dict[str, Any]], 
        max_length: int = 512,
        temperature: float = 0.2,
        num_samples: int = 1
    ) -> Dict[str, Any]:
        """Evaluate model on Human-Eval benchmark."""
        
        results = []
        passed_total = 0
        
        for problem in problems:
            task_id = problem['task_id']
            prompt = problem['prompt']
            test_code = problem['test']
            
            logger.info(f"Evaluating task: {task_id}")
            
            # Generate code completion
            generated_codes = self._generate_completions(
                model, tokenizer, prompt, max_length, temperature, num_samples
            )
            
            # Evaluate each generated completion
            task_passed = False
            for i, completion in enumerate(generated_codes):
                full_code = prompt + completion
                
                # Create test case
                test_task = {
                    'task_id': f"{task_id}_sample_{i}",
                    'generated_code': full_code + "\n\n" + test_code,
                    'test_cases': []
                }
                
                # Evaluate
                result = self.evaluator.evaluate_single_task(test_task)
                result.problem_id = task_id
                result.completion = completion
                
                results.append(result)
                
                if result.passed:
                    task_passed = True
                    break  # At least one sample passed
            
            if task_passed:
                passed_total += 1
        
        # Calculate metrics
        pass_at_k = passed_total / len(problems) if problems else 0
        
        evaluation_summary = {
            'total_problems': len(problems),
            'problems_solved': passed_total,
            'pass_at_1': pass_at_k,
            'detailed_results': results
        }
        
        logger.info(f"Human-Eval Results: {passed_total}/{len(problems)} problems solved (Pass@1: {pass_at_k:.2%})")
        
        return evaluation_summary
    
    def _generate_completions(
        self, 
        model, 
        tokenizer, 
        prompt: str, 
        max_length: int, 
        temperature: float,
        num_samples: int
    ) -> List[str]:
        """Generate code completions using the model."""
        
        completions = []
        
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids']
        
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            model = model.cuda()
        
        model.eval()
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Generate completion
                generated = model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + max_length,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.95 if temperature > 0 else None,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                # Decode and extract completion
                generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                completion = generated_text[len(prompt):].strip()
                
                # Stop at function end or next function definition
                lines = completion.split('\n')
                clean_completion = []
                
                for line in lines:
                    if line.strip().startswith('def ') and clean_completion:
                        break  # Stop at next function
                    clean_completion.append(line)
                
                completion = '\n'.join(clean_completion).strip()
                completions.append(completion)
        
        return completions

def run_evaluation_suite(
    model,
    tokenizer,
    eval_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run comprehensive evaluation suite."""
    
    results = {
        'timestamp': time.time(),
        'model_info': {
            'parameters': model.get_num_params() if hasattr(model, 'get_num_params') else None,
            'config': model.config.to_dict() if hasattr(model, 'config') else None
        },
        'evaluations': {}
    }
    
    # Human-Eval benchmark
    if eval_config.get('run_humaneval', True):
        logger.info("Running Human-Eval benchmark...")
        humaneval = HumanEvalBenchmark()
        problems = humaneval.load_humaneval_problems(eval_config.get('humaneval_path', 'data/humaneval.jsonl'))
        
        humaneval_results = humaneval.evaluate_model_on_humaneval(
            model=model,
            tokenizer=tokenizer,
            problems=problems,
            max_length=eval_config.get('max_length', 512),
            temperature=eval_config.get('temperature', 0.2),
            num_samples=eval_config.get('num_samples', 1)
        )
        
        results['evaluations']['humaneval'] = humaneval_results
    
    return results

if __name__ == "__main__":
    # Test the evaluator
    evaluator = PythonCodeEvaluator()
    
    # Test syntax validation
    valid_code = "def hello():\n    print('Hello, World!')\n    return True"
    is_valid, error = evaluator.evaluate_syntax(valid_code)
    print(f"Valid code syntax check: {is_valid}, Error: {error}")
    
    # Test code execution
    test_task = {
        'task_id': 'test_1',
        'generated_code': 'print("Hello, World!")',
        'expected_output': 'Hello, World!'
    }
    
    result = evaluator.evaluate_single_task(test_task)
    print(f"Execution test passed: {result.passed}")
    print(f"Actual output: {result.actual_output}")
    print(f"Expected output: {result.expected_output}")
    
    # Test Human-Eval benchmark
    humaneval = HumanEvalBenchmark()
    problems = humaneval._create_example_problems()
    print(f"Created {len(problems)} example problems for testing")