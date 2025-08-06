#!/usr/bin/env python3

"""
LimeLLM to Ollama Export Script

This script converts trained LimeLLM models to Ollama-compatible format
with various quantization options and custom configurations.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from export.ollama_converter import OllamaModelConverter

logger = logging.getLogger(__name__)

def validate_model_path(model_path: str) -> Path:
    """Validate that the model path exists and contains a model."""
    path = Path(model_path)
    
    if not path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")
    
    if path.is_dir():
        # Check for HuggingFace format
        required_files = ['config.json', 'pytorch_model.bin']
        if not all((path / file).exists() for file in required_files):
            raise ValueError(f"Directory does not contain a valid HuggingFace model: {model_path}")
    
    elif path.is_file():
        # Check for PyTorch checkpoint
        if not path.suffix in ['.pt', '.pth', '.bin']:
            raise ValueError(f"File does not appear to be a PyTorch checkpoint: {model_path}")
    
    else:
        raise ValueError(f"Invalid model path: {model_path}")
    
    return path

def main():
    parser = argparse.ArgumentParser(
        description='Convert LimeLLM model to Ollama format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with default settings
  python export_to_ollama.py --model-path outputs/limellm-2b --output-dir ollama-models/limellm

  # Convert with Q4 quantization
  python export_to_ollama.py \\
    --model-path outputs/limellm-2b \\
    --output-dir ollama-models/limellm-q4 \\
    --quantization q4_0

  # Convert with custom model name and system prompt
  python export_to_ollama.py \\
    --model-path outputs/limellm-2b \\
    --output-dir ollama-models/limellm-custom \\
    --model-name limellm-custom \\
    --system-prompt "You are a specialized Python coding assistant."

  # After conversion, create Ollama model:
  cd ollama-models/limellm
  ollama create limellm -f Modelfile
  ollama run limellm "Write a Python function to sort a list"
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--model-path',
        required=True,
        help='Path to the trained LimeLLM model (directory or .pt file)'
    )
    
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for Ollama model files'
    )
    
    # Model configuration
    parser.add_argument(
        '--model-name',
        default='limellm',
        help='Name for the Ollama model (default: limellm)'
    )
    
    parser.add_argument(
        '--quantization',
        choices=['f32', 'f16', 'q8_0', 'q4_0', 'q4_1', 'q5_0', 'q5_1'],
        default='f16',
        help='Quantization format (default: f16)'
    )
    
    # Customization
    parser.add_argument(
        '--system-prompt',
        help='Custom system prompt for the model'
    )
    
    parser.add_argument(
        '--template',
        help='Custom chat template (advanced users)'
    )
    
    parser.add_argument(
        '--template-file',
        help='File containing custom chat template'
    )
    
    # Behavior
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output directory'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually converting'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Validate inputs
        model_path = validate_model_path(args.model_path)
        output_dir = Path(args.output_dir)
        
        # Check output directory
        if output_dir.exists() and not args.overwrite:
            logger.error(f"Output directory exists: {output_dir}")
            logger.error("Use --overwrite to replace existing directory")
            sys.exit(1)
        
        # Load custom template if provided
        template = args.template
        if args.template_file:
            template_file = Path(args.template_file)
            if not template_file.exists():
                logger.error(f"Template file not found: {template_file}")
                sys.exit(1)
            
            with open(template_file, 'r', encoding='utf-8') as f:
                template = f.read()
            logger.info(f"Loaded template from: {template_file}")
        
        # Print configuration
        print("\n" + "="*60)
        print("🍋 LIMELLM TO OLLAMA CONVERTER")
        print("="*60)
        print(f"Model Path: {model_path}")
        print(f"Output Directory: {output_dir}")
        print(f"Model Name: {args.model_name}")
        print(f"Quantization: {args.quantization}")
        
        if args.system_prompt:
            print(f"Custom System Prompt: {args.system_prompt[:50]}...")
        
        if template:
            print("Custom Template: Provided")
        
        print("="*60)
        
        # Dry run
        if args.dry_run:
            print("🔍 DRY RUN - Configuration validated")
            print("\nConversion would:")
            print(f"1. Load model from: {model_path}")
            print(f"2. Apply {args.quantization} quantization")
            print(f"3. Create GGUF file")
            print(f"4. Generate Ollama Modelfile")
            print(f"5. Create documentation")
            print(f"6. Save everything to: {output_dir}")
            print("\nTo run conversion, remove --dry-run flag")
            return
        
        # Initialize converter
        converter = OllamaModelConverter()
        
        # Run conversion
        logger.info("Starting model conversion...")
        
        converted_dir = converter.convert_to_ollama(
            model_path=str(model_path),
            output_dir=str(output_dir),
            model_name=args.model_name,
            quantization=args.quantization,
            template=template,
            system_prompt=args.system_prompt
        )
        
        # Success message
        print(f"\n🎉 Conversion completed successfully!")
        print(f"📁 Ollama model created at: {converted_dir}")
        print(f"\n🚀 To use with Ollama:")
        print(f"   cd {converted_dir}")
        print(f"   ollama create {args.model_name} -f Modelfile")
        print(f"   ollama run {args.model_name}")
        print(f"\n💡 Example usage:")
        print(f"   ollama run {args.model_name} \"Write a Python function to calculate fibonacci\"")
        
        # Show next steps
        print(f"\n📚 Documentation:")
        readme_path = Path(converted_dir) / "README.md"
        examples_path = Path(converted_dir) / "EXAMPLES.md"
        
        if readme_path.exists():
            print(f"   README: {readme_path}")
        
        if examples_path.exists():
            print(f"   Examples: {examples_path}")
        
        print("\n" + "="*60)
        
    except KeyboardInterrupt:
        logger.info("Conversion interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()