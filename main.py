#!/usr/bin/env python3
"""
Electrical Symbol Recognition — SAM3 Inference Pipeline (CLI).

Usage:
    python main.py                  # Full pipeline
    python main.py --visualize_only # Re-draw from existing results.json
"""

import argparse

from src.config import (
    ANNOTATIONS_PATH,
    INPUT_DIR,
    OUTPUT_IMAGES_DIR,
    OUTPUT_JSON_PATH,
)
from src.inference_engine import InferenceEngine
from src.metrics import compute_metrics
from src.utils import visualize_coco_results


def run_visualize_only() -> None:
    """Re-generate annotated images and metrics from existing results.json."""
    print("=" * 60)
    print("  Visualize-Only Mode")
    print("=" * 60)

    if not OUTPUT_JSON_PATH.exists():
        print(f"[ERROR] Results file not found: {OUTPUT_JSON_PATH}")
        return

    print(f"\nLoading results from {OUTPUT_JSON_PATH}")
    visualize_coco_results(
        results_json_path=OUTPUT_JSON_PATH,
        images_dir=INPUT_DIR,
        output_images_dir=OUTPUT_IMAGES_DIR,
    )

    print("\nComputing detection metrics...")
    compute_metrics(ANNOTATIONS_PATH, OUTPUT_JSON_PATH)

    print(f"\n{'=' * 60}")
    print("  Visualize-only complete!")
    print(f"{'=' * 60}")


def run_full_pipeline() -> None:
    """Run the full SAM3 inference pipeline via InferenceEngine."""
    print("=" * 60)
    print("  Electrical Symbol Recognition — SAM3 Pipeline")
    print("=" * 60)

    engine = InferenceEngine()
    for _ in engine.process_batch(log=lambda msg: print(msg)):
        pass  # drain generator (prints are the side-effect)

    print("\nComputing detection metrics...")
    compute_metrics(ANNOTATIONS_PATH, OUTPUT_JSON_PATH)

    print(f"\n{'=' * 60}")
    print("  Pipeline complete!")
    print(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Electrical Symbol Recognition with SAM3",
    )
    parser.add_argument(
        "--visualize_only",
        action="store_true",
        help="Skip inference; re-draw from existing results.json.",
    )
    args = parser.parse_args()

    if args.visualize_only:
        run_visualize_only()
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()
