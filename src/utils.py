"""
Utility functions: NMS, Visualization, and COCO helpers.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

from src.config import CLASSES, CLASS_COLORS, NMS_IOU_THRESHOLD


# ============================================================
# COCO Data Loading
# ============================================================

def load_coco_data(json_path: Path) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Load COCO-format JSON and return lookup dictionaries.

    Returns:
        (raw_data, images_by_id, annotations_by_image_id, filename_to_id)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}

    anns: Dict[int, List] = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in anns:
            anns[img_id] = []
        anns[img_id].append(ann)

    filename_to_id = {img["file_name"]: img["id"] for img in data["images"]}

    return data, images, anns, filename_to_id


# ============================================================
# Global Class-Agnostic NMS (Coverage-Based)
# ============================================================

def class_agnostic_nms(
    annotations: List[Dict[str, Any]],
    coverage_threshold: float = NMS_IOU_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Apply class-agnostic NMS using **coverage** instead of IoU.

    Coverage = intersection / min(area_a, area_b)

    This catches nested boxes (a small detection inside a larger one)
    that standard IoU misses. When coverage > threshold, the box with
    the lower confidence score is suppressed — regardless of class.

    Args:
        annotations: list of annotation dicts with 'image_id', 'bbox' [x,y,w,h], 'score'.
        coverage_threshold: coverage ratio above which the lower-scored box is suppressed.

    Returns:
        Filtered list of annotation dicts.
    """
    # Group by image
    by_image: Dict[int, List[Dict]] = {}
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in by_image:
            by_image[img_id] = []
        by_image[img_id].append(ann)

    final: List[Dict] = []

    for _, anns in by_image.items():
        if not anns:
            continue

        # Sort by score descending
        anns.sort(key=lambda x: x["score"], reverse=True)

        keep = []
        while anns:
            best = anns.pop(0)
            keep.append(best)

            bx1, by1, bw, bh = best["bbox"]
            box1 = [bx1, by1, bx1 + bw, by1 + bh]
            area1 = bw * bh

            remaining = []
            for other in anns:
                ox1, oy1, ow, oh = other["bbox"]
                box2 = [ox1, oy1, ox1 + ow, oy1 + oh]
                area2 = ow * oh

                # Compute intersection
                ix1 = max(box1[0], box2[0])
                iy1 = max(box1[1], box2[1])
                ix2 = min(box1[2], box2[2])
                iy2 = min(box1[3], box2[3])

                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter = iw * ih

                # Coverage: how much of the smaller box is covered
                min_area = min(area1, area2)
                coverage = inter / min_area if min_area > 0 else 0

                if coverage < coverage_threshold:
                    remaining.append(other)

            anns = remaining

        final.extend(keep)

    return final


# ============================================================
# Visualization
# ============================================================

def draw_final_boxes(
    image: np.ndarray,
    annotations: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Draw bounding boxes on image with label format: 'ClassID : Confidence'.

    Args:
        image: BGR numpy array.
        annotations: list of dicts with 'bbox', 'category_id', 'score'.

    Returns:
        Copy of the image with drawn boxes.
    """
    vis = image.copy()

    for ann in annotations:
        x, y, w, h = map(int, ann["bbox"])
        cls_id = ann["category_id"]
        score = ann["score"]
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))

        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

        label = f"{cls_id} : {score:.2f}"
        cv2.putText(vis, label, (x, y - 5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return vis


def visualize_coco_results(
    results_json_path: Path,
    images_dir: Path,
    output_images_dir: Path,
    apply_nms: bool = True,
) -> None:
    """
    Load a COCO results JSON, optionally apply class-agnostic NMS,
    and save annotated images.

    The NMS filtering is applied only for drawing — the original JSON
    is never modified.

    Args:
        results_json_path: path to results.json (COCO format).
        images_dir: directory containing the source images.
        output_images_dir: directory to write annotated images.
    """
    with open(results_json_path, "r") as f:
        results = json.load(f)

    annotations = results.get("annotations", [])
    images_list = results.get("images", [])

    if not annotations:
        print("[WARN] No annotations found in results JSON.")
        return

    # Apply NMS for visualization (does not modify the saved JSON)
    if apply_nms:
        draw_anns = class_agnostic_nms(annotations)
        print(f"  NMS for visualization: {len(annotations)} → {len(draw_anns)} boxes")
    else:
        draw_anns = annotations

    # Group filtered annotations by image_id
    anns_by_img: Dict[int, List[Dict]] = {}
    for ann in draw_anns:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    output_images_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for img_info in images_list:
        img_id = img_info["id"]
        filename = img_info["file_name"]
        src_path = images_dir / filename

        if not src_path.exists():
            continue

        img = cv2.imread(str(src_path))
        if img is None:
            continue

        img_anns = anns_by_img.get(img_id, [])
        vis = draw_final_boxes(img, img_anns)

        out_path = output_images_dir / filename
        cv2.imwrite(str(out_path), vis)
        count += 1

    print(f"  Saved {count} annotated images → {output_images_dir}")


# ============================================================
# COCO Results Builder
# ============================================================

def build_coco_results(
    images_dict: Dict[int, Dict],
    annotations: List[Dict[str, Any]],
) -> Dict:
    """
    Build a COCO-format results dictionary.

    Args:
        images_dict: mapping image_id -> {id, file_name, height, width}.
        annotations: final list of annotation dicts.

    Returns:
        COCO-format dict ready for JSON serialisation.
    """
    return {
        "info": {
            "description": "SAM3 Inference Results (All Classes, NMS Filtered)",
            "date": datetime.now().isoformat(),
        },
        "images": list(images_dict.values()),
        "annotations": annotations,
        "categories": [{"id": k, "name": v} for k, v in CLASSES.items()],
    }
