#!/usr/bin/env python3
"""
Main prediction script for electrical symbol recognition using SAM3.
Uses HuggingFace Transformers API.

Usage:
    python scripts/predict.py --image path/to/image.jpg --prompt "object"
    python scripts/predict.py --image path/to/image.jpg --prompt "face" --show
    python scripts/predict.py --image path/to/image.jpg --prompt "object" --save-masks
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config, get_config
from src.inference.predictor import SymbolPredictor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Electrical Symbol Recognition with SAM3 (Transformers)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to input image"
    )
    
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="Text prompt describing objects to detect (e.g., 'resistor', 'capacitor', 'face')"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config YAML file (optional)"
    )
    
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for results (default: data/output)"
    )
    
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save individual mask files"
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show results in popup window"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save visualization (useful with --show)"
    )
    
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token (can also use HF_TOKEN env var)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate input image
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    # Load or create config
    if args.config:
        config = get_config(args.config)
    else:
        config = Config.default()
    
    # Override config with command line args
    config.confidence_threshold = args.threshold
    
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    
    if args.hf_token:
        config.hf_token = args.hf_token
    
    print("=" * 60)
    print("Electrical Symbol Recognition with SAM3")
    print("Using HuggingFace Transformers API")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Prompt: '{args.prompt}'")
    print(f"Threshold: {config.confidence_threshold}")
    print(f"Output: {config.output_dir}")
    print("=" * 60)
    
    # Initialize predictor
    print("\nInitializing model...")
    predictor = SymbolPredictor(config)
    
    # Run prediction
    print(f"\nProcessing image with prompt: '{args.prompt}'")
    results = predictor.process_image(
        image_path=image_path,
        text_prompt=args.prompt,
        save_visualization=not args.no_save,
        save_masks_separately=args.save_masks,
        show_popup=args.show,
    )
    
    # Print results
    masks = results.get("masks", [])
    scores = results.get("scores", [])
    
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Found {len(masks)} object(s)")
    
    for i, score in enumerate(scores):
        score_val = float(score) if hasattr(score, 'item') else score
        print(f"  Object {i+1}: score={score_val:.4f}")
    
    if not args.no_save:
        print(f"\nVisualization saved to: {config.output_dir / 'visualizations'}")
    
    if args.save_masks:
        print(f"Masks saved to: {config.output_dir / 'masks'}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
