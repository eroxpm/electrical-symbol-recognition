
import os
import cv2
import json
import shutil
import random
import gc
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

# ==========================================
# 1. CONFIGURATION
# ==========================================

import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import Config, get_config
from src.model.sam3_wrapper import SAM3Model

# Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "input" / "raw" # LEGACY
INPUT_DIR = DATA_DIR / "input"
TARGET_DIR = INPUT_DIR / "target"
REF_DIR = INPUT_DIR / "reference"
DEBUG_CLASSES_DIR = DATA_DIR / "debug_classes"
IMAGES_DIR = TARGET_DIR # Source images for Inference
ANNOTATIONS_PATH = DATA_DIR / "_annotations.coco.json"

OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
OUTPUT_JSON_PATH = OUTPUT_DIR / "_results.coco.json"

# Clean Setup
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Application Parameters
MATRIX_CROP_SIZE = (64, 64)
MATRIX_PADDING = 10 

# CLASS MAPPING (To be verified)
# CLASS MAPPING (Corrected by User)
CLASSES = {
    0: "Resistor",
    1: "Capacitor",
    2: "Inductor",
    3: "FA DC",
    4: "FA AC"
}

# Manual list of Reference images
REFERENCE_FILENAMES = [
    "354_png.rf.c40016ba7c239b8979addb26ae9b90ed.jpg", # Resistor, Capacitor, FA DC
    "electrical_213_png.rf.af3dc4e0c9d9a75a6f6a84d3a8f12fbf.jpg", # Inductor
    "346_png.rf.8e1a2047a66b822e791ffc4a9fe249a4.jpg" # FA AC
]

random.seed(42)
np.random.seed(42)

# ==========================================
# 2. HELPER CLASSES & FUNCTIONS
# ==========================================

def load_coco_data(json_path: Path) -> Tuple[Dict, Dict]:
    """Loads COCO JSON and returns lookup dicts."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    images = {img['id']: img for img in data['images']}
    anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in anns: anns[img_id] = []
        anns[img_id].append(ann)
        
    # Filename to Image ID lookup
    filename_to_id = {img['file_name']: img['id'] for img in data['images']}
    
    return data, images, anns, filename_to_id

def draw_boxes(image: np.ndarray, boxes: List[List[float]], color=(0, 255, 0), thickness=2) -> np.ndarray:
    img_copy = image.copy()
    for box in boxes:
        cls_id = int(box[0])
        x1, y1, x2, y2 = map(int, box[1:5])
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img_copy, str(cls_id), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img_copy

class MatrixGenerator:
    """Generates the Visual Prompt Matrix using COCO data."""
    
    def __init__(self, images_dir: Path, coco_anns: Dict, filename_to_id: Dict, ref_filenames: List[str]):
        self.images_dir = images_dir
        self.coco_anns = coco_anns
        self.filename_to_id = filename_to_id
        self.ref_filenames = ref_filenames
        self.best_crops: Dict[int, np.ndarray] = {}

    def extract_best_crops(self):
        """Extracts representative crops from the debug_classes directory."""
        print(f"Extracting crops from {DEBUG_CLASSES_DIR}...")
        
        if not DEBUG_CLASSES_DIR.exists():
            print(f"  [ERROR] Debug classes directory not found: {DEBUG_CLASSES_DIR}")
            return

        needed_classes = set(CLASSES.keys())
        
        # We look for files named class_{id}_*.jpg
        all_debug_files = list(DEBUG_CLASSES_DIR.glob("*.jpg"))
        
        for cls_id in list(needed_classes):
            # Find all images for this class
            prefix = f"class_{cls_id}_"
            cls_images = [f for f in all_debug_files if f.name.startswith(prefix)]
            
            if cls_images:
                # Use the first one found
                img_path = cls_images[0]
                img = cv2.imread(str(img_path))
                if img is not None:
                    self.best_crops[cls_id] = img
                    needed_classes.remove(cls_id)
                    print(f"  Got Class {cls_id} from {img_path.name}")
                else:
                    print(f"  [WARN] Failed to read debug image {img_path}")
            else:
                print(f"  [WARN] No debug image found for Class {cls_id} with prefix {prefix}")
        
        print(f"  Final crops for classes: {list(self.best_crops.keys())}")

    def generate_matrix(self, target_cls: Optional[int] = None) -> Tuple[np.ndarray, List[List[Any]]]:
        """Creates reference strip. If target_cls provided, 1 col. Else 5 cols."""
        rows = 4 
        
        if target_cls is not None:
             cols = 1
             classes_to_render = [target_cls]
        else:
             cols = 5
             classes_to_render = range(5)
             
        cw, ch = MATRIX_CROP_SIZE
        pad = MATRIX_PADDING
        
        matrix_h = rows * (ch + pad) + pad
        matrix_w = cols * (cw + pad) + pad
        
        matrix_img = np.ones((matrix_h, matrix_w, 3), dtype=np.uint8) * 255
        
        grid_info = []

        for col_idx, cls_id in enumerate(classes_to_render): # Columns
            if cls_id not in self.best_crops:
                continue
                
            base_crop = self.best_crops[cls_id]
            base_crop = cv2.resize(base_crop, MATRIX_CROP_SIZE)
            
            for r in range(4): # Rows
                if r == 0: rotated = base_crop
                elif r == 1: rotated = cv2.rotate(base_crop, cv2.ROTATE_90_CLOCKWISE)
                elif r == 2: rotated = cv2.rotate(base_crop, cv2.ROTATE_180)
                elif r == 3: rotated = cv2.rotate(base_crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Fixed x_offset calculation based on column index (0 for single col)
                x_offset = pad + col_idx * (cw + pad)
                y_offset = pad + r * (ch + pad)
                
                matrix_img[y_offset:y_offset+ch, x_offset:x_offset+cw] = rotated
                
                grid_info.append({
                    'class_id': cls_id,
                    'box': [x_offset, y_offset, x_offset+cw, y_offset+ch], 
                    'rotation': r * 90
                })
                
        return matrix_img, grid_info

# ==========================================
# 3. MAIN SCRIPT
# ==========================================

def main():
    print("Starting Matrix Prompt Inference (Sequential + Cleanup + NMS Final)...")
    
    # 0. Setup Directories & Migration
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    REF_DIR.mkdir(parents=True, exist_ok=True)
    
    if RAW_DIR.exists():
        print(f"Found legacy RAW directory: {RAW_DIR}. Migrating content to {TARGET_DIR}...")
        for item in RAW_DIR.iterdir():
            if item.is_file():
                dest = TARGET_DIR / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
        # Remove empty raw dir if possible
        try:
            RAW_DIR.rmdir()
            print("Removed empty raw directory.")
        except:
            pass

    # 0. Cleanup Output
    if OUTPUT_DIR.exists():
        print(f"Cleaning existing output directory: {OUTPUT_DIR}")
        try:
            shutil.rmtree(OUTPUT_DIR)
        except PermissionError:
            print("  [WARN] Could not clean output directory due to permissions (Docker owned?). Proceeding...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 0.1 Load Data
    print(f"Loading COCO annotations from {ANNOTATIONS_PATH}")
    coco_data, coco_images, coco_anns_lookup, filename_to_id = load_coco_data(ANNOTATIONS_PATH)
    
    # 0.2 Prepare Output JSON Structure
    out_images_dict = {} # Key: img_id, Value: dict
    all_raw_annotations = [] # Accumulate ALL detections here first
    out_ann_id = 1
    
    # 1. Load Model
    config = get_config()
    print(f"Initializing SAM3 Model: {config.model_id}")
    sam_model = SAM3Model(
        model_id=config.model_id,
        confidence_threshold=config.confidence_threshold,
        mask_threshold=config.mask_threshold,
        device=config.device,
        hf_token=config.hf_token
    )
    sam_model.initialize()

    # 2. Phase A: Extract crops for Reference Strips
    gen = MatrixGenerator(IMAGES_DIR, coco_anns_lookup, filename_to_id, REFERENCE_FILENAMES)
    gen.extract_best_crops()
    
    # 3. Phase B: Sequential Inference Loop
    all_files = list(IMAGES_DIR.glob("*.jpg"))
    targets = [
        f for f in all_files 
        if f.name not in REFERENCE_FILENAMES 
        and not f.name.startswith("reference_")
    ]
    print(f"Found {len(targets)} Target images.")

    # Enable ALL Classes
    TARGET_CLASSES = list(CLASSES.keys())
    
    # SEQUENTIAL LOOP: Class -> Image
    for target_cls in TARGET_CLASSES:
        cls_name = CLASSES.get(target_cls, f"class_{target_cls}")
        print(f"\n=== Processing Class: {cls_name} ({target_cls}) ===")
        
        cls_output_dir = OUTPUT_DIR / cls_name
        cls_process_dir = cls_output_dir / "process"
        cls_result_dir = cls_output_dir / "result"
        
        cls_output_dir.mkdir(parents=True, exist_ok=True)
        cls_process_dir.mkdir(parents=True, exist_ok=True)
        cls_result_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Generator Single Column for this class
            matrix_img_strip, matrix_info_strip = gen.generate_matrix(target_cls=target_cls)
            
            # Helper: Check if strip is valid (not empty/white)
            if not matrix_info_strip:
                 print(f"  [WARN] No reference crops found for {cls_name}. Skipping generation of reference strip.")
            else:
                 # User Request: Save the reference strip to reference directory
                 strip_save_path = REF_DIR / f"reference_{cls_name}.jpg"
                 cv2.imwrite(str(strip_save_path), matrix_img_strip)
                 print(f"  Saved Reference Strip to {strip_save_path}")

            for target_path in targets:
                target_filename = target_path.name
                
                img_id = filename_to_id.get(target_filename, 0)
                
                target_img = cv2.imread(str(target_path))
                if target_img is None: 
                    print(f"  Failed to read image {target_path}")
                    continue
                
                t_h, t_w = target_img.shape[:2]

                # Add to Output JSON Images (Deduplicated)
                if img_id not in out_images_dict:
                    out_images_dict[img_id] = {
                        "id": img_id,
                        "file_name": target_filename,
                        "height": t_h,
                        "width": t_w
                    }

                # Dynamic Scaling of the Strip
                m_h_base, m_w_base = matrix_img_strip.shape[:2]
                scale = t_h / m_h_base
                new_m_w = int(m_w_base * scale)
                new_m_h = t_h
                
                matrix_img_scaled = cv2.resize(matrix_img_strip, (new_m_w, new_m_h), interpolation=cv2.INTER_LINEAR)
                
                matrix_info_scaled = []
                for cell in matrix_info_strip:
                    x1, y1, x2, y2 = cell['box']
                    nx1, ny1, nx2, ny2 = int(x1*scale), int(y1*scale), int(x2*scale), int(y2*scale)
                    matrix_info_scaled.append({
                        'class_id': cell['class_id'],
                        'box': [nx1, ny1, nx2, ny2]
                    })

                canvas = np.zeros((t_h, new_m_w + t_w, 3), dtype=np.uint8)
                canvas[:, :new_m_w, :] = matrix_img_scaled
                canvas[:, new_m_w:, :] = target_img
                canvas_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                
                # Prepare Prompts: All cells in the strip are Positive
                active_prompts = []
                active_labels = [] 
                prompt_ref_size = (0, 0)
                prompt_vis = matrix_img_scaled.copy()
                
                for cell in matrix_info_scaled:
                    # All cells in this strip belong to target_cls
                    x1, y1, x2, y2 = cell['box']
                    active_prompts.append(cell['box'])
                    active_labels.append(1) # Positive
                    cv2.rectangle(prompt_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    prompt_ref_size = (x2 - x1, y2 - y1)
                
                if not active_prompts: 
                    del target_img, canvas, canvas_pil
                    continue

                # Run SAM with no_grad explicit wrapper (just in case model wrapper doesn't cover everything)
                with torch.no_grad():
                     res = sam_model.predict_with_boxes(
                        image=canvas_pil,
                        boxes_xyxy=active_prompts,
                        labels=active_labels
                    )
                
                d_boxes = res.get('boxes', [])
                scores = res.get('scores', []) 
                
                if torch.is_tensor(d_boxes): d_boxes = d_boxes.cpu().numpy().tolist()
                if torch.is_tensor(scores): scores = scores.cpu().numpy().tolist()
                
                if len(scores) != len(d_boxes):
                    scores = [1.0] * len(d_boxes)

                # Process Results (RAW)
                result_vis = target_img.copy()
                valid_count = 0
                
                for i, box in enumerate(d_boxes):
                    x1, y1, x2, y2 = box
                    score = scores[i]
                    cx = (x1 + x2) / 2
                    
                    if cx > new_m_w:
                        nx1 = max(0, x1 - new_m_w)
                        nx2 = max(0, x2 - new_m_w)
                        ny1 = max(0, y1)
                        ny2 = max(0, y2)
                        
                        nw = nx2 - nx1
                        nh = ny2 - ny1
                        
                        # Size Filtering (2x)
                        ref_w, ref_h = prompt_ref_size
                        max_w = ref_w * 2
                        max_h = ref_h * 2
                        
                        if nw > max_w or nh > max_h:
                            continue
                        
                        # Store Raw Detection
                        ann_entry = {
                            "id": out_ann_id,
                            "image_id": img_id,
                            "category_id": target_cls,
                            "bbox": [float(nx1), float(ny1), float(nw), float(nh)],
                            "area": float(nw * nh),
                            "iscrowd": 0,
                            "score": float(score)
                        }

                        all_raw_annotations.append(ann_entry)
                        out_ann_id += 1
                        valid_count += 1
                        
                        # Draw detection on result_vis
                        label_text = f"{cls_name} {score:.2f}"
                        cv2.rectangle(result_vis, (int(nx1), int(ny1)), (int(nx2), int(ny2)), (0, 255, 0), 2)
                        cv2.putText(result_vis, label_text, (int(nx1), int(ny1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Save Input Visualization (Process) - Always
                process_view = canvas.copy()
                process_view[:, :new_m_w, :] = prompt_vis
                cv2.imwrite(str(cls_process_dir / f"{target_filename}_input.jpg"), process_view)
                
                # Save Result Visualization (Result) - Always
                cv2.imwrite(str(cls_result_dir / f"{target_filename}_result.jpg"), result_vis)
                
                if valid_count > 0:
                    print(f"    {target_filename}: Saved {valid_count} detections.")
                
                # Cleanup per image
                del target_img, canvas, canvas_pil, res, d_boxes, scores, process_view, result_vis
        
        except Exception as e:
            print(f"Error processing class {cls_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Force cleanup after class
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # End of Image Loop for this Class -> Save Per-Class JSON
        cls_json_path = cls_result_dir / f"_results_{cls_name}.json"
        
        # Filter annotations for this class to save intermediate json
        cls_specific_anns = [a for a in all_raw_annotations if a['category_id'] == target_cls]
        
        cls_json = {
            "info": {"description": f"SAM3 Results for {cls_name}", "date": datetime.now().isoformat()},
            "images": list(out_images_dict.values()), 
            "annotations": cls_specific_anns,
            "categories": [{"id": target_cls, "name": cls_name}]
        }
        
        with open(cls_json_path, 'w') as f:
            json.dump(cls_json, f, indent=4)
        print(f"  Saved per-class JSON to {cls_json_path}")

    # 4. Global NMS (Optional / Just for Final JSON)
    print("Applying Global Cross-Class NMS for final global JSON...")
    
    final_annotations = []
    
    # Group by Image ID
    anns_by_img_global = {}
    for ann in all_raw_annotations:
        img_id = ann['image_id']
        if img_id not in anns_by_img_global: anns_by_img_global[img_id] = []
        anns_by_img_global[img_id].append(ann)
        
    for img_id, anns in anns_by_img_global.items():
        if not anns: continue
        
        # Sort by Score Descending
        anns.sort(key=lambda x: x['score'], reverse=True)
        
        keep_anns = []
        while anns:
            best = anns.pop(0)
            keep_anns.append(best)
            
            # Compare with rest
            bx1, by1, bw, bh = best['bbox']
            box1 = [bx1, by1, bx1+bw, by1+bh]
            area1 = bw * bh
            
            remaining = []
            for other in anns:
                ox1, oy1, ow, oh = other['bbox']
                box2 = [ox1, oy1, ox1+ow, oy1+oh]
                area2 = ow * oh
                
                # IoU
                ix1 = max(box1[0], box2[0])
                iy1 = max(box1[1], box2[1])
                ix2 = min(box1[2], box2[2])
                iy2 = min(box1[3], box2[3])
                
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                
                inter = iw * ih
                union = area1 + area2 - inter
                iou = inter / union if union > 0 else 0
                
                if iou < 0.5: # Keep if not overlapping too much
                    remaining.append(other)
            
            anns = remaining
            
        final_annotations.extend(keep_anns)
        
    print(f"NMS reduced annotations from {len(all_raw_annotations)} to {len(final_annotations)}")

    # 5. Save Final Global JSON
    final_json = {
        "info": {"description": "SAM3 Inference Results (All Classes, NMS Filtered)", "date": datetime.now().isoformat()},
        "images": list(out_images_dict.values()),
        "annotations": final_annotations,
        "categories": [{"id": k, "name": v} for k, v in CLASSES.items()]
    }
    
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(final_json, f, indent=4)
        
    print(f"Inference Complete. All results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
