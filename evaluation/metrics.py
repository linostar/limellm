import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import ast
import re
from collections import Counter
from dataclasses import dataclass
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import difflib

logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    
    # Language modeling metrics
    perplexity: float = 0.0
    loss: float = 0.0
    
    # Code-specific metrics
    syntax_accuracy: float = 0.0
    code_compilation_rate: float = 0.0
    functional_correctness: float = 0.0
    
    # Generation quality metrics
    bleu_score: Optional[float] = None
    rouge_scores: Optional[Dict[str, float]] = None
    edit_distance: Optional[float] = None
    
    # Performance metrics
    generation_time: float = 0.0
    tokens_per_second: float = 0.0
    
    # Human evaluation metrics
    readability_score: Optional[float] = None
    maintainability_score: Optional[float] = None

class CodeQualityAnalyzer:
    """Analyzer for code quality metrics."""
    
    def __init__(self):
        self.python_keywords = {
            'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
            'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
            'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
            'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return',
            'try', 'while', 'with', 'yield'
        }
    
    def analyze_syntax_correctness(self, code: str) -> Dict[str, Any]:
        """Analyze Python syntax correctness."""
        try:
            ast.parse(code)
            return {
                'is_valid': True,
                'error': None,
                'error_type': None
            }
        except SyntaxError as e:
            return {
                'is_valid': False,
                'error': str(e),
                'error_type': 'syntax_error',
                'line_number': e.lineno,
                'offset': e.offset
            }
        except Exception as e:
            return {
                'is_valid': False,
                'error': str(e),
                'error_type': 'parsing_error'
            }
    
    def analyze_code_structure(self, code: str) -> Dict[str, Any]:
        """Analyze code structure and complexity."""
        try:
            tree = ast.parse(code)
            
            analyzer = CodeStructureVisitor()
            analyzer.visit(tree)
            
            return {
                'num_functions': analyzer.num_functions,
                'num_classes': analyzer.num_classes,
                'num_imports': analyzer.num_imports,
                'cyclomatic_complexity': analyzer.cyclomatic_complexity,
                'max_nesting_depth': analyzer.max_nesting_depth,
                'num_comments': len(re.findall(r'#.*', code)),
                'num_docstrings': analyzer.num_docstrings,
                'lines_of_code': len([line for line in code.split('\n') if line.strip()]),
                'has_type_hints': analyzer.has_type_hints
            }
        except Exception as e:
            logger.warning(f"Error analyzing code structure: {e}")
            return {
                'num_functions': 0,
                'num_classes': 0,
                'error': str(e)
            }
    
    def calculate_readability_score(self, code: str) -> float:
        """Calculate readability score based on various factors."""
        if not code.strip():
            return 0.0
        
        score = 0.0
        max_score = 100.0
        
        # Factor 1: Comment density (0-20 points)
        lines = code.split('\n')
        code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        comment_lines = [line for line in lines if line.strip().startswith('#')]
        
        if code_lines:
            comment_ratio = len(comment_lines) / len(code_lines)
            score += min(20.0, comment_ratio * 100)  # 20% comments = full points
        
        # Factor 2: Average line length (0-20 points)
        if code_lines:
            avg_line_length = sum(len(line) for line in code_lines) / len(code_lines)
            # Optimal range: 50-80 characters
            if 50 <= avg_line_length <= 80:
                score += 20.0
            elif 30 <= avg_line_length < 50 or 80 < avg_line_length <= 100:
                score += 15.0
            elif 20 <= avg_line_length < 30 or 100 < avg_line_length <= 120:
                score += 10.0
            else:
                score += 5.0
        
        # Factor 3: Proper naming conventions (0-20 points)
        naming_score = self._analyze_naming_conventions(code)
        score += naming_score * 20
        
        # Factor 4: Function/class structure (0-20 points)
        structure = self.analyze_code_structure(code)
        if structure.get('num_functions', 0) > 0:
            score += 10.0  # Has functions
        if structure.get('num_docstrings', 0) > 0:
            score += 10.0  # Has docstrings
        
        # Factor 5: Complexity (0-20 points)
        complexity = structure.get('cyclomatic_complexity', 0)
        if complexity <= 5:
            score += 20.0  # Low complexity
        elif complexity <= 10:
            score += 15.0
        elif complexity <= 15:
            score += 10.0
        else:
            score += 5.0
        
        return min(score, max_score)
    
    def _analyze_naming_conventions(self, code: str) -> float:
        """Analyze adherence to Python naming conventions."""
        try:
            tree = ast.parse(code)
            visitor = NamingConventionVisitor()
            visitor.visit(tree)
            
            total_names = visitor.total_names
            correct_names = visitor.correct_names
            
            return correct_names / total_names if total_names > 0 else 1.0
        
        except Exception:
            return 0.5  # Default score if analysis fails

class CodeStructureVisitor(ast.NodeVisitor):
    """AST visitor to analyze code structure."""
    
    def __init__(self):
        self.num_functions = 0
        self.num_classes = 0
        self.num_imports = 0
        self.cyclomatic_complexity = 1  # Base complexity
        self.max_nesting_depth = 0
        self.current_depth = 0
        self.num_docstrings = 0
        self.has_type_hints = False
    
    def visit_FunctionDef(self, node):
        self.num_functions += 1
        
        # Check for docstring
        if (node.body and 
            isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Str)):
            self.num_docstrings += 1
        
        # Check for type hints
        if node.returns or any(arg.annotation for arg in node.args.args):
            self.has_type_hints = True
        
        self.current_depth += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1
    
    def visit_ClassDef(self, node):
        self.num_classes += 1
        
        # Check for class docstring
        if (node.body and 
            isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Str)):
            self.num_docstrings += 1
        
        self.current_depth += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1
    
    def visit_Import(self, node):
        self.num_imports += len(node.names)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        self.num_imports += len(node.names)
        self.generic_visit(node)
    
    def visit_If(self, node):
        self.cyclomatic_complexity += 1
        self.generic_visit(node)
    
    def visit_While(self, node):
        self.cyclomatic_complexity += 1
        self.generic_visit(node)
    
    def visit_For(self, node):
        self.cyclomatic_complexity += 1
        self.generic_visit(node)
    
    def visit_ExceptHandler(self, node):
        self.cyclomatic_complexity += 1
        self.generic_visit(node)

class NamingConventionVisitor(ast.NodeVisitor):
    """AST visitor to check naming conventions."""
    
    def __init__(self):
        self.total_names = 0
        self.correct_names = 0
    
    def visit_FunctionDef(self, node):
        self.total_names += 1
        # Functions should be snake_case
        if re.match(r'^[a-z_][a-z0-9_]*$', node.name):
            self.correct_names += 1
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        self.total_names += 1
        # Classes should be PascalCase
        if re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
            self.correct_names += 1
        self.generic_visit(node)
    
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):  # Variable assignment
            self.total_names += 1
            # Variables should be snake_case
            if re.match(r'^[a-z_][a-z0-9_]*$', node.id):
                self.correct_names += 1

class GenerationMetricsCalculator:
    """Calculate metrics for text generation quality."""
    
    def __init__(self):
        pass
    
    def calculate_bleu_score(self, generated: str, reference: str) -> float:
        """Calculate BLEU score between generated and reference text."""
        try:
            from nltk.translate.bleu_score import sentence_bleu
            
            reference_tokens = reference.split()
            generated_tokens = generated.split()
            
            return sentence_bleu([reference_tokens], generated_tokens)
        
        except ImportError:
            # Fallback to simple token overlap if NLTK not available
            ref_tokens = set(reference.split())
            gen_tokens = set(generated.split())
            
            if not ref_tokens:
                return 1.0 if not gen_tokens else 0.0
            
            overlap = len(ref_tokens & gen_tokens)
            return overlap / len(ref_tokens)
    
    def calculate_rouge_scores(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference, generated)
            
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        
        except ImportError:
            # Simple fallback
            return {
                'rouge1': self._simple_rouge_1(generated, reference),
                'rouge2': 0.0,  # Not implemented in fallback
                'rougeL': self._simple_rouge_l(generated, reference)
            }
    
    def _simple_rouge_1(self, generated: str, reference: str) -> float:
        """Simple ROUGE-1 implementation."""
        gen_tokens = generated.split()
        ref_tokens = reference.split()
        
        if not ref_tokens:
            return 1.0 if not gen_tokens else 0.0
        
        overlap = len(set(gen_tokens) & set(ref_tokens))
        return overlap / len(set(ref_tokens))
    
    def _simple_rouge_l(self, generated: str, reference: str) -> float:
        """Simple ROUGE-L implementation using LCS."""
        gen_tokens = generated.split()
        ref_tokens = reference.split()
        
        lcs_length = self._lcs_length(gen_tokens, ref_tokens)
        
        if not ref_tokens:
            return 1.0 if not gen_tokens else 0.0
        
        return lcs_length / len(ref_tokens)
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def calculate_edit_distance(self, generated: str, reference: str) -> float:
        """Calculate normalized edit distance."""
        if not reference:
            return 0.0 if not generated else 1.0
        
        # Use difflib for sequence matching
        matcher = difflib.SequenceMatcher(None, generated, reference)
        return 1.0 - matcher.ratio()

class EvaluationMetricsCollector:
    """Collect and aggregate evaluation metrics."""
    
    def __init__(self):
        self.code_analyzer = CodeQualityAnalyzer()
        self.generation_calculator = GenerationMetricsCalculator()
        
    def evaluate_model_outputs(
        self,
        generated_texts: List[str],
        reference_texts: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate model outputs comprehensively."""
        
        results = {
            'total_samples': len(generated_texts),
            'code_quality': {},
            'generation_quality': {},
            'syntax_analysis': {}
        }
        
        # Code quality analysis
        syntax_correct = 0
        readability_scores = []
        structure_metrics = {
            'functions': [], 'classes': [], 'complexity': [],
            'comments': [], 'docstrings': [], 'type_hints': []
        }
        
        for text in generated_texts:
            # Syntax analysis
            syntax_result = self.code_analyzer.analyze_syntax_correctness(text)
            if syntax_result['is_valid']:
                syntax_correct += 1
            
            # Structure analysis
            structure = self.code_analyzer.analyze_code_structure(text)
            structure_metrics['functions'].append(structure.get('num_functions', 0))
            structure_metrics['classes'].append(structure.get('num_classes', 0))
            structure_metrics['complexity'].append(structure.get('cyclomatic_complexity', 0))
            structure_metrics['comments'].append(structure.get('num_comments', 0))
            structure_metrics['docstrings'].append(structure.get('num_docstrings', 0))
            structure_metrics['type_hints'].append(1 if structure.get('has_type_hints', False) else 0)
            
            # Readability
            readability = self.code_analyzer.calculate_readability_score(text)
            readability_scores.append(readability)
        
        results['code_quality'] = {
            'syntax_correctness_rate': syntax_correct / len(generated_texts),
            'average_readability': np.mean(readability_scores) if readability_scores else 0,
            'average_functions_per_sample': np.mean(structure_metrics['functions']),
            'average_classes_per_sample': np.mean(structure_metrics['classes']),
            'average_complexity': np.mean(structure_metrics['complexity']),
            'comment_usage_rate': np.mean([1 if c > 0 else 0 for c in structure_metrics['comments']]),
            'docstring_usage_rate': np.mean([1 if d > 0 else 0 for d in structure_metrics['docstrings']]),
            'type_hint_usage_rate': np.mean(structure_metrics['type_hints'])
        }
        
        # Generation quality (if references provided)
        if reference_texts and len(reference_texts) == len(generated_texts):
            bleu_scores = []
            rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
            edit_distances = []
            
            for gen, ref in zip(generated_texts, reference_texts):
                # BLEU
                bleu = self.generation_calculator.calculate_bleu_score(gen, ref)
                bleu_scores.append(bleu)
                
                # ROUGE
                rouge = self.generation_calculator.calculate_rouge_scores(gen, ref)
                for key in rouge_scores:
                    rouge_scores[key].append(rouge.get(key, 0))
                
                # Edit distance
                edit_dist = self.generation_calculator.calculate_edit_distance(gen, ref)
                edit_distances.append(edit_dist)
            
            results['generation_quality'] = {
                'average_bleu': np.mean(bleu_scores),
                'average_rouge1': np.mean(rouge_scores['rouge1']),
                'average_rouge2': np.mean(rouge_scores['rouge2']),
                'average_rougeL': np.mean(rouge_scores['rougeL']),
                'average_edit_distance': np.mean(edit_distances)
            }
        
        return results
    
    def create_evaluation_report(
        self,
        metrics: Dict[str, Any],
        output_file: Optional[str] = None
    ) -> str:
        """Create a formatted evaluation report."""
        
        report_lines = [
            "="*60,
            "LimeLLM MODEL EVALUATION REPORT",
            "="*60,
            "",
            f"Total samples evaluated: {metrics['total_samples']}",
            "",
            "CODE QUALITY METRICS:",
            "-" * 30
        ]
        
        code_quality = metrics.get('code_quality', {})
        if code_quality:
            report_lines.extend([
                f"Syntax correctness rate: {code_quality['syntax_correctness_rate']:.1%}",
                f"Average readability score: {code_quality['average_readability']:.1f}/100",
                f"Average functions per sample: {code_quality['average_functions_per_sample']:.1f}",
                f"Average classes per sample: {code_quality['average_classes_per_sample']:.1f}",
                f"Average cyclomatic complexity: {code_quality['average_complexity']:.1f}",
                f"Comment usage rate: {code_quality['comment_usage_rate']:.1%}",
                f"Docstring usage rate: {code_quality['docstring_usage_rate']:.1%}",
                f"Type hint usage rate: {code_quality['type_hint_usage_rate']:.1%}",
                ""
            ])
        
        gen_quality = metrics.get('generation_quality', {})
        if gen_quality:
            report_lines.extend([
                "GENERATION QUALITY METRICS:",
                "-" * 30,
                f"Average BLEU score: {gen_quality['average_bleu']:.3f}",
                f"Average ROUGE-1: {gen_quality['average_rouge1']:.3f}",
                f"Average ROUGE-2: {gen_quality['average_rouge2']:.3f}",
                f"Average ROUGE-L: {gen_quality['average_rougeL']:.3f}",
                f"Average edit distance: {gen_quality['average_edit_distance']:.3f}",
                ""
            ])
        
        report_lines.append("="*60)
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report

if __name__ == "__main__":
    # Test the metrics
    analyzer = CodeQualityAnalyzer()
    
    # Test code samples
    good_code = '''
def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number.
    
    Args:
        n: The position in the Fibonacci sequence.
        
    Returns:
        The nth Fibonacci number.
    """
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Test the function
result = fibonacci(10)
print(f"The 10th Fibonacci number is: {result}")
'''
    
    bad_code = '''
def f(x):
    if x<=1:return x
    return f(x-1)+f(x-2)
print(f(10))
'''
    
    # Analyze code quality
    good_syntax = analyzer.analyze_syntax_correctness(good_code)
    bad_syntax = analyzer.analyze_syntax_correctness(bad_code)
    
    print(f"Good code syntax valid: {good_syntax['is_valid']}")
    print(f"Bad code syntax valid: {bad_syntax['is_valid']}")
    
    good_readability = analyzer.calculate_readability_score(good_code)
    bad_readability = analyzer.calculate_readability_score(bad_code)
    
    print(f"Good code readability: {good_readability:.1f}/100")
    print(f"Bad code readability: {bad_readability:.1f}/100")
    
    # Test evaluation collector
    collector = EvaluationMetricsCollector()
    results = collector.evaluate_model_outputs([good_code, bad_code])
    
    report = collector.create_evaluation_report(results)
    print("\n" + report)