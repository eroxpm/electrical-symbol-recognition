#!/usr/bin/env python3
"""
Electrical Symbol Recognition – SAM3 Inference Pipeline.

Usage:
    python main.py                  # Full pipeline (inference + NMS + visualisation)
    python main.py --visualize_only # Re-draw images from existing results.json
"""

import argparse
import json
import random
import cv2
import numpy as np
from pathlib import Path

from src.config import (
    CLASSES,
    INPUT_DIR,
    REFERENCES_DIR,
    ANNOTATIONS_PATH,
    OUTPUT_DIR,
    OUTPUT_IMAGES_DIR,
    OUTPUT_JSON_PATH,
    REFERENCE_FILENAMES,
)
from src.metrics import compute_metrics
from src.utils import (
    load_coco_data,
    class_agnostic_nms,
    build_coco_results,
    visualize_coco_results,
)

# Reproducibility
random.seed(42)
np.random.seed(42)


def get_target_images(images_dir: Path) -> list[Path]:
    """Return target image paths, excluding reference images."""
    all_files = list(images_dir.glob("*.jpg"))
    return [
        f for f in all_files
        if f.name not in REFERENCE_FILENAMES
        and not f.name.startswith("reference_")
    ]


def run_visualize_only() -> None:
    """Re-generate annotated images from an existing results.json."""
    print("=" * 60)
    print("  Visualize-Only Mode")
    print("=" * 60)

    if not OUTPUT_JSON_PATH.exists():
        print(f"[ERROR] Results file not found: {OUTPUT_JSON_PATH}")
        print("  Run a full inference first, or check the path.")
        return

    print(f"\nLoading results from {OUTPUT_JSON_PATH}")
    visualize_coco_results(
        results_json_path=OUTPUT_JSON_PATH,
        images_dir=INPUT_DIR,
        output_images_dir=OUTPUT_IMAGES_DIR,
    )

    # ── Metrics ───────────────────────────────────────────────
    print("\nComputing detection metrics...")
    compute_metrics(ANNOTATIONS_PATH, OUTPUT_JSON_PATH)

    print(f"\n{'=' * 60}")
    print("  Visualize-only complete!")
    print(f"{'=' * 60}")


def run_full_pipeline() -> None:
    """Run the full SAM3 inference pipeline."""
    # Lazy import — avoid loading heavy SAM3 deps in visualize-only mode
    from src.detector import MatrixGenerator, run_inference_for_class

    print("=" * 60)
    print("  Electrical Symbol Recognition – SAM3 Pipeline")
    print("=" * 60)

    # ── Setup output directories ──────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load annotations ──────────────────────────────────────
    print(f"\nLoading COCO annotations from {ANNOTATIONS_PATH}")
    coco_data, coco_images, coco_anns, filename_to_id = load_coco_data(ANNOTATIONS_PATH)

    # ── Discover target images ────────────────────────────────
    targets = get_target_images(INPUT_DIR)
    print(f"Found {len(targets)} target images in {INPUT_DIR}")

    if not targets:
        print("[WARN] No target images found. Exiting.")
        return

    # ── Phase A: Build reference strips ───────────────────────
    print("\nPhase A: Generating reference strips...")
    gen = MatrixGenerator(INPUT_DIR, coco_anns, coco_images, filename_to_id)
    gen.extract_best_crops()

    precomputed_strips = {}
    for cls_id, cls_name in CLASSES.items():
        strip_img, strip_info = gen.generate_matrix(target_cls=cls_id)
        if not strip_info:
            print(f"  [WARN] No reference crops for {cls_name}. Skipping.")
            continue

        strip_path = REFERENCES_DIR / f"reference_{cls_name}.jpg"
        strip_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(strip_path), strip_img)
        print(f"  Saved reference strip → {strip_path}")

        precomputed_strips[cls_id] = (strip_img, strip_info)

    # ── Phase B: Sequential per-class inference ───────────────
    print("\nPhase B: Running inference...")
    out_images_dict: dict = {}
    all_raw_annotations: list = []
    ann_id = 1

    for cls_id in CLASSES:
        if cls_id not in precomputed_strips:
            continue

        strip_img, strip_info = precomputed_strips[cls_id]

        class_anns, ann_id = run_inference_for_class(
            target_cls=cls_id,
            strip_img=strip_img,
            strip_info=strip_info,
            target_paths=targets,
            filename_to_id=filename_to_id,
            out_images_dict=out_images_dict,
            start_ann_id=ann_id,
        )
        all_raw_annotations.extend(class_anns)

    # ── Phase C: Global Class-Agnostic NMS ────────────────────
    print(f"\nPhase C: Global NMS ({len(all_raw_annotations)} raw annotations)...")
    final_annotations = class_agnostic_nms(all_raw_annotations)
    print(f"  NMS: {len(all_raw_annotations)} → {len(final_annotations)} annotations")

    # ── Phase D: Save results ─────────────────────────────────
    for idx, ann in enumerate(final_annotations, start=1):
        ann["id"] = idx

    results = build_coco_results(out_images_dict, final_annotations)

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved results → {OUTPUT_JSON_PATH}")

    # ── Phase E: Visualisation (reuse shared function) ────────
    print("Generating annotated images...")
    visualize_coco_results(
        results_json_path=OUTPUT_JSON_PATH,
        images_dir=INPUT_DIR,
        output_images_dir=OUTPUT_IMAGES_DIR,
        apply_nms=False,  # Already NMS-filtered in Phase C
    )

    # ── Phase F: Detection Metrics ────────────────────────────
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
        help="Skip inference. Re-draw images from existing output/results.json.",
    )
    args = parser.parse_args()

    if args.visualize_only:
        run_visualize_only()
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()
