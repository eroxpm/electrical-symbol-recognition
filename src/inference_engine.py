"""
Inference Engine — Reusable wrapper for the SAM3 detection pipeline.

Provides a lazy-loaded model + precomputed strips that can be shared
across both Single Image Debug and Batch Analysis modes.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config import (
    CLASSES,
    ANNOTATIONS_PATH,
    INPUT_DIR,
    OUTPUT_DIR,
    OUTPUT_IMAGES_DIR,
    OUTPUT_JSON_PATH,
    REFERENCES_DIR,
    REFERENCE_FILENAMES,
    MODEL_ID,
    MASK_THRESHOLD,
    DEVICE,
    HF_TOKEN,
    VISUALIZATION_THRESHOLDS,
    VISUALIZATION_DISABLED_CLASSES,
)
from src.utils import (
    load_coco_data,
    class_agnostic_nms,
    build_coco_results,
    draw_final_boxes,
    visualize_coco_results,
)



class InferenceEngine:
    """
    Lazy-loaded SAM3 inference engine.

    The heavy model and reference strips are initialized on first use
    and reused across subsequent calls.
    """

    def __init__(self):
        self._model = None
        self._strips: Dict[int, Tuple[np.ndarray, list]] = {}
        self._coco_data = None
        self._filename_to_id: Dict[str, int] = {}
        self._ready = False

    # ----------------------------------------------------------
    # Lazy initialization
    # ----------------------------------------------------------

    def _ensure_loaded(self, log: Optional[Callable] = None):
        """Load model + precompute strips on first call."""
        if self._ready:
            return

        _log = log or (lambda msg: print(msg))

        from src.detector import MatrixGenerator, SAM3Model

        # COCO annotations
        _log("Loading annotations...")
        coco_data, coco_images, coco_anns, filename_to_id = load_coco_data(
            ANNOTATIONS_PATH,
        )
        self._coco_data = coco_data
        self._filename_to_id = filename_to_id

        # Reference strips
        _log("Building reference strips...")
        gen = MatrixGenerator(INPUT_DIR, coco_anns, coco_images, filename_to_id)
        gen.extract_best_crops()

        for cls_id, cls_name in CLASSES.items():
            strip_img, strip_info = gen.generate_matrix(target_cls=cls_id)
            if strip_info:
                strip_path = REFERENCES_DIR / f"reference_{cls_name}.jpg"
                strip_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(strip_path), strip_img)
                self._strips[cls_id] = (strip_img, strip_info)
                _log(f"  ✅ {cls_name}")
            else:
                _log(f"  ⚠️ No crops for {cls_name}")

        # SAM3 model
        _log("Loading SAM3 model...")
        self._model = SAM3Model(
            model_id=MODEL_ID,
            confidence_threshold=0.3,
            mask_threshold=MASK_THRESHOLD,
            device=DEVICE,
            hf_token=HF_TOKEN,
        )
        self._model.initialize()
        _log("✅ Model ready")
        self._ready = True

    # ----------------------------------------------------------
    # Single Image
    # ----------------------------------------------------------

    def process_single_image(
        self,
        image_path: Path,
    ) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]], str]:
        """
        Run all classes on ONE image → (annotated_rgb, detections, debug_log).

        Returns:
            annotated_rgb: numpy RGB image with drawn boxes (or None).
            detections: list of {label, class_id, score, bbox}.
            debug_log: multi-line string with processing details.
        """
        debug_lines: List[str] = []

        def dlog(msg: str):
            debug_lines.append(msg)

        self._ensure_loaded(log=dlog)

        from src.detector import infer_image_all_classes

        img_id = self._filename_to_id.get(image_path.name, 0)
        dlog(f"Image: {image_path.name} (id={img_id})")

        raw_anns, _, _ = infer_image_all_classes(
            target_path=image_path,
            img_id=img_id,
            precomputed_strips=self._strips,
            sam_model=self._model,
        )
        dlog(f"Raw detections: {len(raw_anns)}")

        # Per-class breakdown
        by_class: Dict[str, int] = {}
        for ann in raw_anns:
            cls_name = CLASSES.get(ann["category_id"], "?")
            by_class[cls_name] = by_class.get(cls_name, 0) + 1
        for cls, cnt in sorted(by_class.items()):
            dlog(f"  {cls}: {cnt}")

        # NMS
        if raw_anns:
            final_anns = class_agnostic_nms(raw_anns)
            removed = len(raw_anns) - len(final_anns)
            dlog(f"NMS removed {removed} boxes → {len(final_anns)} kept")
        else:
            final_anns = []
            dlog("No detections to filter")

        # Apply per-class visualization thresholds (same as batch pipeline)
        filtered_anns = []
        for ann in final_anns:
            cls_id = ann["category_id"]
            if cls_id in VISUALIZATION_DISABLED_CLASSES:
                continue
            threshold = VISUALIZATION_THRESHOLDS.get(cls_id, 0.5)
            if ann.get("score", 1.0) < threshold:
                continue
            filtered_anns.append(ann)

        vis_removed = len(final_anns) - len(filtered_anns)
        if vis_removed > 0:
            dlog(f"Visualization filter removed {vis_removed} → {len(filtered_anns)} shown")
        final_anns = filtered_anns

        # Draw
        img = cv2.imread(str(image_path))
        if img is not None and final_anns:
            img = draw_final_boxes(img, final_anns)
        annotated_rgb = (
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
        )

        # JSON-friendly detection list
        detections = []
        for ann in final_anns:
            cls_id = ann["category_id"]
            detections.append({
                "label": CLASSES.get(cls_id, f"class_{cls_id}"),
                "score": round(ann["score"], 4),
                "bbox": [
                    round(ann["bbox"][0], 1),
                    round(ann["bbox"][1], 1),
                    round(ann["bbox"][2], 1),
                    round(ann["bbox"][3], 1),
                ],
            })

        return annotated_rgb, detections, "\n".join(debug_lines)

    # ----------------------------------------------------------
    # Batch
    # ----------------------------------------------------------

    def process_batch(
        self,
        log: Optional[Callable] = None,
    ):
        """
        Generator: run full pipeline, yielding after each image for streaming.

        Yields nothing (caller should yield its own UI updates).
        Final state: results.json + annotated images saved to disk.
        """
        _log = log or (lambda msg: print(msg))
        self._ensure_loaded(log=_log)
        yield  # after model load

        from src.detector import infer_image_all_classes

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

        targets = self._get_target_images()
        _log(f"\n🚀 Running inference on {len(targets)} images...")
        yield  # after setup

        out_images_dict: Dict[int, Dict] = {}
        all_annotations: List[Dict] = []
        ann_id = 1

        for idx, target_path in enumerate(targets, start=1):
            img_id = self._filename_to_id.get(target_path.name, 0)

            image_anns, image_meta, ann_id = infer_image_all_classes(
                target_path=target_path,
                img_id=img_id,
                precomputed_strips=self._strips,
                sam_model=self._model,
                start_ann_id=ann_id,
                log=_log,
            )

            if image_meta:
                out_images_dict[img_id] = image_meta
            all_annotations.extend(image_anns)

            _log(
                f"  📸 [{idx}/{len(targets)}] {target_path.name}: "
                f"{len(image_anns)} detections"
            )
            yield  # after each image

        # NMS
        raw_count = len(all_annotations)
        final_annotations = class_agnostic_nms(all_annotations)
        for i, ann in enumerate(final_annotations, start=1):
            ann["id"] = i
        _log(f"\n🧹 NMS: {raw_count} → {len(final_annotations)}")
        yield  # after NMS

        # Save JSON
        results = build_coco_results(out_images_dict, final_annotations)
        with open(OUTPUT_JSON_PATH, "w") as f:
            json.dump(results, f, indent=4)
        _log(f"💾 Saved → {OUTPUT_JSON_PATH}")

        # Annotated images
        visualize_coco_results(
            results_json_path=OUTPUT_JSON_PATH,
            images_dir=INPUT_DIR,
            output_images_dir=OUTPUT_IMAGES_DIR,
            apply_nms=False,
        )
        _log("🎨 Annotated images saved")
        yield  # done

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    @staticmethod
    def _get_target_images() -> List[Path]:
        return sorted([
            f for f in INPUT_DIR.glob("*.jpg")
            if f.name not in REFERENCE_FILENAMES
            and not f.name.startswith("reference_")
        ])

    @staticmethod
    def list_input_images() -> List[str]:
        """Return filenames available for the dropdown."""
        return [
            f.name for f in sorted(INPUT_DIR.glob("*.jpg"))
            if f.name not in REFERENCE_FILENAMES
            and not f.name.startswith("reference_")
        ]

    @staticmethod
    def load_reference_gallery() -> list:
        """Load reference mosaic strips as gallery items (one per class)."""
        gallery = []
        for p in sorted(REFERENCES_DIR.glob("reference_*.jpg")):
            img = cv2.imread(str(p))
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                label = p.stem.replace("reference_", "")
                gallery.append((rgb, label))
        return gallery

    @staticmethod
    def load_output_gallery() -> list:
        """Load all annotated images from output/images/ as gallery items."""
        gallery = []
        if OUTPUT_IMAGES_DIR.exists():
            for p in sorted(OUTPUT_IMAGES_DIR.glob("*.jpg")):
                img = cv2.imread(str(p))
                if img is not None:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    gallery.append((rgb, p.stem))
        return gallery
