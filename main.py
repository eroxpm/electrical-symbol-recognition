#!/usr/bin/env python3
"""
Electrical Symbol Recognition – SAM3 Inference Pipeline.

Single entry point that:
  1. Loads COCO annotations.
  2. Extracts reference crops and builds visual-prompt strips.
  3. Iterates through each class sequentially, running SAM3 inference.
  4. Applies Global Class-Agnostic NMS across all detections.
  5. Saves results.json and annotated images to output/.
"""

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
from src.detector import MatrixGenerator, run_inference_for_class
from src.utils import load_coco_data, class_agnostic_nms, draw_final_boxes, build_coco_results

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


def main() -> None:
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

        # Save strip for debugging / reproducibility
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
    # Re-assign sequential IDs after NMS
    for idx, ann in enumerate(final_annotations, start=1):
        ann["id"] = idx

    results = build_coco_results(out_images_dict, final_annotations)

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved results → {OUTPUT_JSON_PATH}")

    # ── Phase E: Visualisation ────────────────────────────────
    print("Generating annotated images...")

    # Group annotations by image
    anns_by_img: dict = {}
    for ann in final_annotations:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    for img_id, img_info in out_images_dict.items():
        src_path = INPUT_DIR / img_info["file_name"]
        if not src_path.exists():
            continue

        img = cv2.imread(str(src_path))
        if img is None:
            continue

        img_anns = anns_by_img.get(img_id, [])
        vis = draw_final_boxes(img, img_anns)

        out_path = OUTPUT_IMAGES_DIR / img_info["file_name"]
        cv2.imwrite(str(out_path), vis)

    print(f"Saved annotated images → {OUTPUT_IMAGES_DIR}")
    print(f"\n{'=' * 60}")
    print("  Pipeline complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
