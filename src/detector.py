"""
SAM3 Inference Logic: Model wrapper, Visual Prompt Matrix, and per-class inference.

All logic is preserved identically from scripts/mosaic_inference.py to ensure
bit-for-bit identical results.
"""

import gc
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any

from src.config import (
    CLASSES,
    CONFIDENCE_THRESHOLDS,
    DEFAULT_CONFIDENCE,
    MODEL_ID,
    MASK_THRESHOLD,
    DEVICE,
    HF_TOKEN,
    REFERENCES_DIR,
    INPUT_DIR,
    REFERENCE_FILENAMES,
    CUSTOM_REFERENCE_CROPS,
    CUSTOM_FA_DC_SOURCE,
    MATRIX_CROP_SIZE,
    MATRIX_PADDING,
    MATRIX_ROWS,
    CROP_MARGINS,
    SCALE_RANGES,
    CELL_SIZE_BUFFER,
    SIZE_FILTER_MULTIPLIER,
)


# ============================================================
# SAM3 Model Wrapper
# ============================================================

class SAM3Model:
    """Wrapper for the SAM3 model via HuggingFace Transformers API."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        confidence_threshold: float = DEFAULT_CONFIDENCE,
        mask_threshold: float = MASK_THRESHOLD,
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        self.model_id = model_id
        self.confidence_threshold = confidence_threshold
        self.mask_threshold = mask_threshold
        self.hf_token = hf_token
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._initialized = False

    def initialize(self) -> None:
        """Load model and processor from HuggingFace."""
        if self._initialized:
            return

        if self.hf_token:
            from huggingface_hub import login
            login(token=self.hf_token)

        from transformers import Sam3Model, Sam3Processor

        from src.config import MODELS_DIR
        cache_dir = MODELS_DIR / "huggingface"
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading SAM3 model from {self.model_id}...")
        print(f"Cache directory: {cache_dir}")
        self.model = Sam3Model.from_pretrained(
            self.model_id, cache_dir=str(cache_dir)
        ).to(self.device)
        self.processor = Sam3Processor.from_pretrained(
            self.model_id, cache_dir=str(cache_dir)
        )
        self._initialized = True
        print("Model loaded successfully!")

    def predict_with_boxes(
        self,
        image: Image.Image,
        boxes_xyxy: List[List[float]],
        labels: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Run inference using multiple bounding-box prompts."""
        if not self._initialized:
            self.initialize()

        if not boxes_xyxy:
            return {"masks": [], "boxes": [], "scores": []}

        input_boxes = [boxes_xyxy]
        if labels is not None:
            input_boxes_labels = [labels]
        else:
            input_boxes_labels = [[1] * len(boxes_xyxy)]

        inputs = self.processor(
            images=image,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.confidence_threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        return results


# ============================================================
# Visual-Prompt Matrix Generator
# ============================================================

class MatrixGenerator:
    """Generates the visual-prompt reference strip using COCO annotations."""

    def __init__(
        self,
        images_dir: Path,
        coco_anns: Dict,
        coco_images: Dict,
        filename_to_id: Dict,
    ):
        self.images_dir = images_dir
        self.coco_anns = coco_anns
        self.coco_images = coco_images
        self.filename_to_id = filename_to_id
        self.best_crops: Dict[int, np.ndarray] = {}

    # ----------------------------------------------------------
    # Crop Extraction
    # ----------------------------------------------------------

    def extract_best_crops(self) -> None:
        """Extract representative crops for each class."""
        print(f"Extracting crops from {REFERENCES_DIR}...")

        if not REFERENCES_DIR.exists():
            print(f"  [ERROR] References directory not found: {REFERENCES_DIR}")
            return

        needed_classes = set(CLASSES.keys())
        all_ref_files = list(REFERENCES_DIR.glob("*.jpg"))

        # --- Custom Capacitor / Inductor overrides ---
        for cls_id, filename in CUSTOM_REFERENCE_CROPS.items():
            if cls_id not in needed_classes:
                continue
            custom_path = REFERENCES_DIR / filename
            if custom_path.exists():
                cls_name = CLASSES.get(cls_id, str(cls_id))
                print(f"  [INFO] {cls_name} (Class {cls_id}): Using custom reference {filename}")
                img = cv2.imread(str(custom_path))
                if img is not None:
                    self.best_crops[cls_id] = img
                    needed_classes.discard(cls_id)
                else:
                    print(f"  [WARN] Failed to read custom image {filename}")

        # --- Custom FA DC: extracted from COCO annotation ---
        if 3 in needed_classes:
            target_img_id = self.filename_to_id.get(CUSTOM_FA_DC_SOURCE)
            if target_img_id:
                anns = self.coco_anns.get(target_img_id, [])
                dc_ann = next((a for a in anns if a["category_id"] == 3), None)
                if dc_ann:
                    print(f"  [INFO] FA DC (Class 3): Extracting custom reference from {CUSTOM_FA_DC_SOURCE}")
                    src_path = self.images_dir / CUSTOM_FA_DC_SOURCE
                    src_img = cv2.imread(str(src_path))
                    if src_img is not None:
                        x, y, w, h = map(int, dc_ann["bbox"])
                        # Tight crop (0% margin)
                        crop = src_img[y : y + h, x : x + w]
                        if crop.size > 0:
                            self.best_crops[3] = crop
                            needed_classes.discard(3)
                    else:
                        print(f"  [WARN] Could not read {src_path}")
                else:
                    print("  [WARN] Custom FA DC image found but no Class 3 annotation.")
            else:
                print(f"  [WARN] Custom FA DC filename not found in COCO: {CUSTOM_FA_DC_SOURCE}")

        # --- Fallback: debug_classes crops ---
        for cls_id in list(needed_classes):
            prefix = f"class_{cls_id}_"
            cls_images = [f for f in all_ref_files if f.name.startswith(prefix)]
            if not cls_images:
                continue

            debug_path = cls_images[0]
            debug_filename = debug_path.name

            try:
                parts = debug_filename.replace(".jpg", "").split("_")
                img_id = int(parts[-2])
                ann_id = int(parts[-1])

                if img_id not in self.coco_images:
                    continue

                img_info = self.coco_images[img_id]
                src_filename = img_info["file_name"]
                src_path = self.images_dir / src_filename
                src_img = cv2.imread(str(src_path))
                if src_img is None:
                    print(f"  [WARN] Failed to read source image {src_path}")
                    continue

                target_ann = None
                if img_id in self.coco_anns:
                    for ann in self.coco_anns[img_id]:
                        if ann["id"] == ann_id:
                            target_ann = ann
                            break

                if target_ann is None:
                    continue

                x, y, w, h = map(float, target_ann["bbox"])
                margin_pct = CROP_MARGINS.get(cls_id, 0.15)
                margin_w = w * margin_pct
                margin_h = h * margin_pct

                if cls_id == 0:
                    margin_w = margin_h = 0
                    print(f"  [INFO] Resistor (Class 0): Using TIGHT crop (0% expansion).")

                x1 = int(max(0, x - margin_w))
                y1 = int(max(0, y - margin_h))
                x2 = int(min(src_img.shape[1], x + w + margin_w))
                y2 = int(min(src_img.shape[0], y + h + margin_h))

                if x2 > x1 and y2 > y1:
                    crop = src_img[y1:y2, x1:x2]
                    self.best_crops[cls_id] = crop
                    needed_classes.discard(cls_id)
                    print(f"  Got Class {cls_id} from {src_filename} (Expanded Crop)")

            except Exception as e:
                print(f"  [WARN] Failed to parse/process {debug_filename}: {e}")

    # ----------------------------------------------------------
    # Matrix Generation
    # ----------------------------------------------------------

    def generate_matrix(self, target_cls: int) -> Tuple[np.ndarray, List[Dict]]:
        """
        Build the reference strip for a single class.

        Returns:
            (matrix_image, grid_info) where grid_info contains cell metadata
            with tight bounding boxes in matrix coordinates.
        """
        rows = MATRIX_ROWS
        cols = 2
        classes_to_render = [target_cls, target_cls]

        cw, ch = MATRIX_CROP_SIZE

        # Dynamic sizing based on actual crop dimensions
        if target_cls in self.best_crops:
            crop = self.best_crops[target_cls]
            crop_h, crop_w = crop.shape[:2]
            s = max(crop_h, crop_w)
            s = int(s * CELL_SIZE_BUFFER)
            cw, ch = s, s
            print(f"  [INFO] {CLASSES.get(target_cls)} Matrix: Using dynamic size {cw}x{ch}")

        pad = MATRIX_PADDING
        matrix_h = rows * (ch + pad) + pad
        matrix_w = cols * (cw + pad) + pad
        matrix_img = np.ones((matrix_h, matrix_w, 3), dtype=np.uint8) * 255

        grid_info: List[Dict] = []

        for col_idx, cls_id in enumerate(classes_to_render):
            if cls_id not in self.best_crops:
                continue

            base_crop = self.best_crops[cls_id]

            for r in range(rows):
                # ----- Multi-scale column (col 1) -----
                if col_idx == 1:
                    s_low, s_high = SCALE_RANGES.get(cls_id, (0.65, 1.35))

                    if r == 0:
                        scale_factor, rot_code = s_low, None
                    elif r == 1:
                        scale_factor, rot_code = s_low, cv2.ROTATE_90_CLOCKWISE
                    elif r == 2:
                        scale_factor, rot_code = s_high, None
                    else:
                        scale_factor, rot_code = s_high, cv2.ROTATE_90_CLOCKWISE

                    bh, bw = self.best_crops[cls_id].shape[:2]
                    new_h, new_w = int(bh * scale_factor), int(bw * scale_factor)
                    resized = cv2.resize(self.best_crops[cls_id], (new_w, new_h))

                    if rot_code is not None:
                        resized = cv2.rotate(resized, rot_code)

                    rh, rw = resized.shape[:2]
                    container = np.ones((ch, cw, 3), dtype=np.uint8) * 255

                    y_off = (ch - rh) // 2
                    x_off = (cw - rw) // 2

                    # Clipping logic
                    src_x1, src_y1 = 0, 0
                    src_x2, src_y2 = rw, rh
                    dst_x1, dst_y1 = x_off, y_off
                    dst_x2, dst_y2 = x_off + rw, y_off + rh

                    if dst_x1 < 0:
                        src_x1 -= dst_x1
                        dst_x1 = 0
                    if dst_y1 < 0:
                        src_y1 -= dst_y1
                        dst_y1 = 0
                    if dst_x2 > cw:
                        src_x2 -= (dst_x2 - cw)
                        dst_x2 = cw
                    if dst_y2 > ch:
                        src_y2 -= (dst_y2 - ch)
                        dst_y2 = ch

                    if src_x2 > src_x1 and src_y2 > src_y1:
                        container[dst_y1:dst_y2, dst_x1:dst_x2] = \
                            resized[src_y1:src_y2, src_x1:src_x2]

                    rotated = container
                    content_box_rel = [dst_x1, dst_y1, dst_x2, dst_y2]

                # ----- Standard column (col 0) -----
                else:
                    bh, bw = self.best_crops[cls_id].shape[:2]
                    container = np.ones((ch, cw, 3), dtype=np.uint8) * 255
                    y_off = (ch - bh) // 2
                    x_off = (cw - bw) // 2
                    h_fit = min(bh, ch)
                    w_fit = min(bw, cw)
                    container[y_off:y_off + h_fit, x_off:x_off + w_fit] = \
                        self.best_crops[cls_id][:h_fit, :w_fit]

                    content_box_rel = [x_off, y_off, x_off + w_fit, y_off + h_fit]

                    if r == 0:
                        rotated = container
                    elif r == 1:
                        rotated = cv2.rotate(container, cv2.ROTATE_90_CLOCKWISE)
                    elif r == 2:
                        rotated = cv2.rotate(container, cv2.ROTATE_180)
                    else:
                        rotated = cv2.rotate(container, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Adjust content_box for rotation (standard column only)
                needs_swap = (col_idx != 1)
                final_rel_box = list(content_box_rel)

                if needs_swap:
                    curr_w = content_box_rel[2] - content_box_rel[0]
                    curr_h = content_box_rel[3] - content_box_rel[1]

                    if r in [1, 3]:
                        curr_w, curr_h = curr_h, curr_w

                    new_x_off = (cw - curr_w) // 2
                    new_y_off = (ch - curr_h) // 2
                    final_rel_box = [new_x_off, new_y_off, new_x_off + curr_w, new_y_off + curr_h]

                # Place cell in matrix
                x_offset = pad + col_idx * (cw + pad)
                y_offset = pad + r * (ch + pad)
                matrix_img[y_offset:y_offset + ch, x_offset:x_offset + cw] = rotated

                # Absolute box in matrix coordinates
                abs_x1 = x_offset + final_rel_box[0]
                abs_y1 = y_offset + final_rel_box[1]
                abs_x2 = x_offset + final_rel_box[2]
                abs_y2 = y_offset + final_rel_box[3]

                grid_info.append({
                    "class_id": cls_id,
                    "box": [abs_x1, abs_y1, abs_x2, abs_y2],
                    "rotation": r * 90,
                })

        return matrix_img, grid_info


# ============================================================
# Per-Image All-Classes Inference
# ============================================================


def infer_image_all_classes(
    target_path: Path,
    img_id: int,
    precomputed_strips: Dict[int, Tuple[np.ndarray, List[Dict]]],
    sam_model: "SAM3Model",
    start_ann_id: int = 1,
    log=None,
) -> Tuple[List[Dict], Dict, int]:
    """
    Run SAM3 inference on ONE image for ALL classes.

    Loads the image once, then iterates over each class strip.
    The SAM3 model must already be initialized.

    Args:
        target_path: path to the target image.
        img_id: COCO image ID.
        precomputed_strips: {cls_id: (strip_img, strip_info)}.
        sam_model: pre-initialized SAM3Model instance.
        start_ann_id: starting annotation ID counter.
        log: optional logging callback.

    Returns:
        (annotations, image_meta, next_ann_id)
    """
    _log = log or (lambda msg: print(msg))
    target_img = cv2.imread(str(target_path))
    if target_img is None:
        return [], {}, start_ann_id

    t_h, t_w = target_img.shape[:2]
    image_meta = {
        "id": img_id,
        "file_name": target_path.name,
        "height": t_h,
        "width": t_w,
    }

    annotations: List[Dict] = []
    ann_id = start_ann_id

    for cls_id, (strip_img, strip_info) in precomputed_strips.items():
        cls_name = CLASSES.get(cls_id, str(cls_id))
        conf = CONFIDENCE_THRESHOLDS.get(cls_id, DEFAULT_CONFIDENCE)
        sam_model.confidence_threshold = conf
        ann_id_before = ann_id

        # Scale strip to match target height
        m_h_base, m_w_base = strip_img.shape[:2]
        target_strip_h = min(int(t_h * 1.0), t_h)
        scale = target_strip_h / m_h_base
        new_m_w = int(m_w_base * scale)

        matrix_content = cv2.resize(
            strip_img, (new_m_w, target_strip_h),
            interpolation=cv2.INTER_LINEAR,
        )

        # Pad to full target height (center vertically)
        matrix_scaled = np.ones((t_h, new_m_w, 3), dtype=np.uint8) * 255
        y_offset = (t_h - target_strip_h) // 2
        matrix_scaled[y_offset:y_offset + target_strip_h, :] = matrix_content

        # Scale cell metadata
        scaled_info = []
        for cell in strip_info:
            x1, y1, x2, y2 = cell["box"]
            nx1 = max(0, int(x1 * scale))
            ny1 = max(0, int(y1 * scale) + y_offset)
            nx2 = min(new_m_w, int(x2 * scale))
            ny2 = min(target_strip_h + y_offset, int(y2 * scale) + y_offset)
            scaled_info.append({"class_id": cell["class_id"], "box": [nx1, ny1, nx2, ny2]})

        # Build canvas: [reference strip | target image]
        canvas = np.zeros((t_h, new_m_w + t_w, 3), dtype=np.uint8)
        canvas[:, :new_m_w, :] = matrix_scaled
        canvas[:, new_m_w:, :] = target_img
        canvas_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

        # Prepare prompts
        active_prompts = [cell["box"] for cell in scaled_info]
        if not active_prompts:
            continue

        active_labels = [1] * len(active_prompts)
        last = active_prompts[-1]
        prompt_ref_size = (last[2] - last[0], last[3] - last[1])

        # Inference
        try:
            res = sam_model.predict_with_boxes(
                image=canvas_pil,
                boxes_xyxy=active_prompts,
                labels=active_labels,
            )
        except Exception as e:
            _log(f"    ⚠️ Inference failed for {cls_name}: {e}")
            continue

        d_boxes = res.get("boxes", [])
        scores = res.get("scores", [])

        if torch.is_tensor(scores):
            scores = scores.tolist() if scores.numel() > 0 else []
        if not scores:
            scores = [1.0] * len(d_boxes)

        # Filter detections
        for i, box in enumerate(d_boxes):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            if cx > new_m_w:
                nx1 = max(0, x1 - new_m_w)
                nx2 = max(0, x2 - new_m_w)
                ny1 = max(0, y1)
                ny2 = max(0, y2)
                nw = nx2 - nx1
                nh = ny2 - ny1

                ref_w, ref_h = prompt_ref_size
                if nw > ref_w * SIZE_FILTER_MULTIPLIER or nh > ref_h * SIZE_FILTER_MULTIPLIER:
                    continue

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [float(nx1), float(ny1), float(nw), float(nh)],
                    "area": float(nw * nh),
                    "iscrowd": 0,
                    "score": float(scores[i]),
                })
                ann_id += 1

        cls_det = ann_id - ann_id_before
        if cls_det > 0:
            _log(f"    {cls_name}: {cls_det} det.")

        del canvas, canvas_pil

    del target_img
    return annotations, image_meta, ann_id

