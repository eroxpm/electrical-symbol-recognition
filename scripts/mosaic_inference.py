
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
    
    def __init__(self, images_dir: Path, coco_anns: Dict, coco_images: Dict, filename_to_id: Dict, ref_filenames: List[str]):
        self.images_dir = images_dir
        self.coco_anns = coco_anns
        self.coco_images = coco_images # ID -> Image Info
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
        
        # We look for files named class_{cls_id}_{cls_name}_{img_id}_{ann_id}.jpg
        # Example: class_0_Resistor_1_2.jpg -> cls=0, img_id=1, ann_id=2
        all_debug_files = list(DEBUG_CLASSES_DIR.glob("*.jpg"))
        
        all_debug_files = list(DEBUG_CLASSES_DIR.glob("*.jpg"))
        
        # --- CUSTOM CAPACITOR OVERRIDE ---
        # User requested specific file: data/input/reference_images/class_1_Capacitor_1_1.jpg
        custom_cap_path = PROJECT_ROOT / "data/input/reference_images/class_1_Capacitor_1_1.jpg"
        if 1 in needed_classes and custom_cap_path.exists():
            print(f"  [INFO] Capacitor (Class 1): Using custom reference {custom_cap_path.name}")
            custom_img = cv2.imread(str(custom_cap_path))
            if custom_img is not None:
                self.best_crops[1] = custom_img
                needed_classes.remove(1)
            else:
                print(f"  [WARN] Failed to read custom capacitor image")
        # ---------------------------------

        # --- CUSTOM INDUCTOR OVERRIDE ---
        # User requested specific file logic similar to capacitor
        # Candidates: class_2_Inductor_1_3.jpg
        custom_ind_path = PROJECT_ROOT / "data/input/reference_images/class_2_Inductor_1_3.jpg"
        if 2 in needed_classes and custom_ind_path.exists():
            print(f"  [INFO] Inductor (Class 2): Using custom reference {custom_ind_path.name}")
            custom_img_ind = cv2.imread(str(custom_ind_path))
            if custom_img_ind is not None:
                self.best_crops[2] = custom_img_ind
                needed_classes.remove(2)
            else:
                print(f"  [WARN] Failed to read custom inductor image")
        # --------------------------------

        # --- CUSTOM FA DC OVERRIDE ---
        # User requested specific file: electrical_75... (Horizontal)
        custom_dc_filename = "electrical_75_png.rf.5d96d5206833ea346077c38cc7950c68.jpg"
        
        if 3 in needed_classes:
             target_img_id = self.filename_to_id.get(custom_dc_filename)
             if target_img_id:
                  anns = self.coco_anns.get(target_img_id, [])
                  # Find Class 3
                  dc_ann = next((ann for ann in anns if ann['category_id'] == 3), None)
                  
                  if dc_ann:
                      print(f"  [INFO] FA DC (Class 3): Extracting custom reference from {custom_dc_filename}")
                      src_path = self.images_dir / custom_dc_filename
                      src_img = cv2.imread(str(src_path))
                      
                      if src_img is not None:
                          x, y, w, h = map(int, dc_ann['bbox'])
                          # Use TIGHT crop (0% Margin) as per user request
                          margin_w = 0 
                          margin_h = 0
                          x1 = max(0, x - margin_w); y1 = max(0, y - margin_h)
                          x2 = min(src_img.shape[1], x + w + margin_w)
                          y2 = min(src_img.shape[0], y + h + margin_h)
                          crop = src_img[y1:y2, x1:x2]
                          
                          if crop.size > 0:
                              self.best_crops[3] = crop
                              needed_classes.remove(3)
                              
                  else:
                       print(f"  [WARN] Custom FA DC image found but no Class 3 annotation present.")
             else:
                  print(f"  [WARN] Custom FA DC filename not found in COCO: {custom_dc_filename}")
        # --------------------------------

        for cls_id in list(needed_classes):
            prefix = f"class_{cls_id}_"
            cls_images = [f for f in all_debug_files if f.name.startswith(prefix)]
            
            if cls_images:
                # Use the first one found
                debug_path = cls_images[0]
                debug_filename = debug_path.name
                
                try:
                    parts = debug_filename.replace(".jpg", "").split("_")
                    # parts: ['class', '0', 'Resistor', '1', '2']
                    # We need img_id (index -2) and ann_id (index -1)
                    img_id = int(parts[-2])
                    ann_id = int(parts[-1])
                    
                    # Look up original image filename
                    if img_id in self.coco_images:
                         img_info = self.coco_images[img_id]
                         src_filename = img_info['file_name']
                         src_path = self.images_dir / src_filename
                         
                         src_img = cv2.imread(str(src_path))
                         if src_img is None:
                             print(f"  [WARN] Failed to read source image {src_path}")
                             continue
                             
                         # Find the specific annotation
                         target_ann = None
                         if img_id in self.coco_anns:
                             for ann in self.coco_anns[img_id]:
                                 if ann['id'] == ann_id:
                                     target_ann = ann
                                     break
                         
                         if target_ann:
                             x, y, w, h = map(float, target_ann['bbox'])
                             
                             # 15% Expansion Logic (EXCEPT for Resistor - Class 0)
                             if cls_id == 0:
                                 margin_w = 0
                                 margin_h = 0
                                 print(f"  [INFO] Resistor (Class 0): Using TIGHT crop (0% expansion).")
                             else:
                                 margin_w = w * 0.15
                                 margin_h = h * 0.15
                             
                             x1 = int(max(0, x - margin_w))
                             y1 = int(max(0, y - margin_h))
                             x2 = int(min(src_img.shape[1], x + w + margin_w))
                             y2 = int(min(src_img.shape[0], y + h + margin_h))
                             
                             if x2 > x1 and y2 > y1:
                                 crop = src_img[y1:y2, x1:x2]
                                 self.best_crops[cls_id] = crop
                                 needed_classes.remove(cls_id)
                                 print(f"  Got Class {cls_id} from {src_filename} (Expanded Crop)")
                                 continue
                except Exception as e:
                     print(f"  [WARN] Failed to parse/process {debug_filename}: {e}")
                
    def generate_matrix(self, target_cls: Optional[int] = None) -> Tuple[np.ndarray, List[List[Any]]]:
        """Creates reference strip. If target_cls provided, 1 col. Else 5 cols."""
        rows = 4 
        
        if target_cls is not None:
             cols = 1
             classes_to_render = [target_cls]
             
             # Resistor(0), Cap(1), Ind(2), FA DC(3), FA AC(4): 2 Cols
             if target_cls in [0, 1, 2, 3, 4]:
                 cols = 2
                 classes_to_render = [target_cls, target_cls] # Render Class twice
        else:
             cols = 5
             classes_to_render = range(5)
             
        cw, ch = MATRIX_CROP_SIZE
        
        # Dynamic Size for All Classes (0-4)
        if target_cls in [0, 1, 2, 3, 4] and target_cls in self.best_crops:
             crop = self.best_crops[target_cls]
             h, w = crop.shape[:2]
             s = max(h, w)
             
             # For All Classes (0-4), we need extra space for scaled variations
             if target_cls in [0, 1, 2, 3, 4]:
                  s = int(s * 1.40) # Larger buffer to prevent rounding errors (was 1.35)
                 
             cw, ch = s, s # Use native max dim (square)
             print(f"  [INFO] {CLASSES.get(target_cls)} Matrix: Using dynamic size {cw}x{ch}")
        
        pad = MATRIX_PADDING
        
        matrix_h = rows * (ch + pad) + pad
        matrix_w = cols * (cw + pad) + pad
        
        matrix_img = np.ones((matrix_h, matrix_w, 3), dtype=np.uint8) * 255
        
        grid_info = []

        for col_idx, cls_id in enumerate(classes_to_render): # Columns
            if cls_id not in self.best_crops:
                continue
                
            base_crop = self.best_crops[cls_id]
            # Special resize logic for All Classes (0-4)
            if cls_id in [0, 1, 2, 3, 4]:
                
                # Check for Multi-Scale Column (Column 1)
                is_multiscale_col = (cls_id in [0, 1, 2, 3, 4] and col_idx == 1)
                
                # Expand base crop for 1.25x scale if needed? 
                # Actually, we resize/place into container below.
                pass 
            else:
                base_crop = cv2.resize(base_crop, (cw, ch))
            
            for r in range(4): # Rows
                img_to_rotate = base_crop # Default
                
                # ALL CLASSES MULTI-SCALE LOGIC (Column 1)
                if cls_id in [0, 1, 2, 3, 4] and col_idx == 1:
                    # Row 0: 0.75x (0 deg)
                    # Row 1: 0.75x (90 deg)
                    # Row 2: 1.25x (0 deg)
                    # Row 3: 1.25x (90 deg)
                    
                    scale_factor = 1.0
                    rot_code = None
                    
                    # Default Scales
                    s_low, s_high = 0.75, 1.25
                    
                    # All Classes Tuning: +/- 35% (User Request)
                    if cls_id in [0, 1, 2, 3, 4]:
                        s_low, s_high = 0.65, 1.35
                    
                    # FA DC (Class 3) Special Tuning: +/- 50%
                    if cls_id == 3:
                        s_low, s_high = 0.50, 1.50
                    
                    if r == 0: scale_factor = s_low; rot_code = None
                    elif r == 1: scale_factor = s_low; rot_code = cv2.ROTATE_90_CLOCKWISE
                    elif r == 2: scale_factor = s_high; rot_code = None
                    elif r == 3: scale_factor = s_high; rot_code = cv2.ROTATE_90_CLOCKWISE
                    
                    # Resize Base Crop
                    h, w = self.best_crops[cls_id].shape[:2]
                    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                    resized_crop = cv2.resize(self.best_crops[cls_id], (new_w, new_h))
                    
                    if rot_code is not None:
                        resized_crop = cv2.rotate(resized_crop, rot_code)
                        
                    # Place in Container (cw, ch) - Center
                    rh, rw = resized_crop.shape[:2]
                    container = np.ones((ch, cw, 3), dtype=np.uint8) * 255
                    
                    y_off = (ch - rh) // 2
                    x_off = (cw - rw) // 2
                    
                    # Coordinate Calculation with Clipping
                    src_x1, src_y1 = 0, 0
                    src_x2, src_y2 = rw, rh
                    dst_x1, dst_y1 = x_off, y_off
                    dst_x2, dst_y2 = x_off + rw, y_off + rh
                    
                    # Handle Negative Start (Source larger than Dest)
                    if dst_x1 < 0:
                        src_x1 -= dst_x1
                        dst_x1 = 0
                    if dst_y1 < 0:
                        src_y1 -= dst_y1
                        dst_y1 = 0
                        
                    # Handle Exceeding End
                    if dst_x2 > cw:
                        src_x2 -= (dst_x2 - cw)
                        dst_x2 = cw
                    if dst_y2 > ch:
                        src_y2 -= (dst_y2 - ch)
                        dst_y2 = ch
                        
                    # Copy if valid
                    if src_x2 > src_x1 and src_y2 > src_y1:
                        container[dst_y1:dst_y2, dst_x1:dst_x2] = resized_crop[src_y1:src_y2, src_x1:src_x2]

                    rotated = container
                    
                    # Track tight content box (relative to container) - Use Clipped Coords
                    content_box_rel = [dst_x1, dst_y1, dst_x2, dst_y2]

                elif cls_id in [0, 1, 2, 3, 4]:
                     # Standard Logic for All Classes (Column 0 or Single)
                     # Centered in square container, NO SCALING (1.0x)
                     
                     h, w = self.best_crops[cls_id].shape[:2]
                     container = np.ones((ch, cw, 3), dtype=np.uint8) * 255
                     y_off = (ch - h) // 2
                     x_off = (cw - w) // 2
                     h_fit = min(h, ch)
                     w_fit = min(w, cw)
                     container[y_off:y_off+h_fit, x_off:x_off+w_fit] = self.best_crops[cls_id][:h_fit, :w_fit]
                     
                     container[y_off:y_off+h_fit, x_off:x_off+w_fit] = self.best_crops[cls_id][:h_fit, :w_fit]
                     
                     img_to_rotate = container
                     
                     # Track tight content box (relative to container)
                     # Note: This is pre-rotation. If we rotate 90/270, we might need to swap dims.
                     # But here the container is square (cw=ch), and we center.
                     # The rotated image is the container rotated against its center.
                     # So the content box also rotates.
                     
                     # Let's simplify: 
                     # For standard/centered content, the box is centered.
                     # If we rotate the container, the content rotates with it.
                     # Calculating the new box after rotation is complex if we just rotate the image.
                     # BUT: standard rotations of a centered object in a square container...
                     # 0/180: w, h are same dims.
                     # 90/270: w becomes h, h becomes w.
                     # Offsets change to re-center.
                     
                     # Logic below handles rotation of `rotated` image.
                     # To get tight box, we should probably calculate it AFTER rotation?
                     # OR just rely on the fact that we know the size and it's centered.
                     pass
                     
                     content_box_rel = [x_off, y_off, x_off+w_fit, y_off+h_fit] # Initial
                     
                     if r == 0: rotated = img_to_rotate
                     elif r == 1: rotated = cv2.rotate(img_to_rotate, cv2.ROTATE_90_CLOCKWISE)
                     elif r == 2: rotated = cv2.rotate(img_to_rotate, cv2.ROTATE_180)
                     elif r == 3: rotated = cv2.rotate(img_to_rotate, cv2.ROTATE_90_COUNTERCLOCKWISE)

                else:
                    # Standard logic for other classes
                    if r == 0: rotated = base_crop
                    elif r == 1: rotated = cv2.rotate(base_crop, cv2.ROTATE_90_CLOCKWISE)
                    elif r == 2: rotated = cv2.rotate(base_crop, cv2.ROTATE_180)
                    elif r == 3: rotated = cv2.rotate(base_crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
                    # For standard resizing (else block), the content fills the cell (cw, ch)
                    content_box_rel = [0, 0, cw, ch]
                
                # Calculate Final Tight Box in Matrix Coordinates
                # content_box_rel is [bx1, by1, bx2, by2] inside the cell (cw x ch)
                
                # If we rotated the container/image, we need to adjust content_box_rel?
                # For Inductor/Capacitor:
                # We essentially re-centered or the rotation handled it.
                # If we use cv2.rotate on the container, the content rotates.
                # For 90/270, w and h swap.
                
                final_rel_box = list(content_box_rel)
                
                # Apply rotation/swap logic ONLY for Standard Column (which processes unrotated container)
                # Multi-Scale Column already has fully rotated content_box_rel.
                
                needs_swap = (cls_id in [0, 1, 2, 3, 4]) and not (cls_id in [0, 1, 2, 3, 4] and col_idx == 1)
                
                if needs_swap:
                     # Re-calculate based on rotated size if 90/270
                     # We know the content size (rh, rw) or (h, w).
                     # And it's always centered in (cw, ch).
                     
                     # Current content dims
                     curr_w = content_box_rel[2] - content_box_rel[0]
                     curr_h = content_box_rel[3] - content_box_rel[1]
                     
                     if r in [1, 3]: # 90 or 270
                         # Swap dims
                         tmp = curr_w
                         curr_w = curr_h
                         curr_h = tmp
                         
                     # Re-center
                     new_x_off = (cw - curr_w) // 2
                     new_y_off = (ch - curr_h) // 2
                     final_rel_box = [new_x_off, new_y_off, new_x_off+curr_w, new_y_off+curr_h]

                
                # Fixed x_offset calculation based on column index
                x_offset = pad + col_idx * (cw + pad)
                y_offset = pad + r * (ch + pad)
                
                matrix_img[y_offset:y_offset+ch, x_offset:x_offset+cw] = rotated
                
                # Absolute Box
                abs_x1 = x_offset + final_rel_box[0]
                abs_y1 = y_offset + final_rel_box[1]
                abs_x2 = x_offset + final_rel_box[2]
                abs_y2 = y_offset + final_rel_box[3]

                grid_info.append({
                    'class_id': cls_id,
                    'box': [abs_x1, abs_y1, abs_x2, abs_y2], 
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
    
    # 1. Phase A: Generate ALL Reference Strips (Before Inference)
    print("Phase A: Generating all reference strips...")
    gen = MatrixGenerator(IMAGES_DIR, coco_anns_lookup, coco_images, filename_to_id, REFERENCE_FILENAMES)
    gen.extract_best_crops()
    
    precomputed_strips = {}
    
    for cls_id, cls_name in CLASSES.items():
        # Generate Strip
        matrix_img_strip, matrix_info_strip = gen.generate_matrix(target_cls=cls_id)
        
        if not matrix_info_strip:
             print(f"  [WARN] No reference crops found for {cls_name}. Skipping.")
             continue
             
        # Save Strip
        strip_save_path = REF_DIR / f"reference_{cls_name}.jpg"
        cv2.imwrite(str(strip_save_path), matrix_img_strip)
        print(f"  Saved Reference Strip to {strip_save_path}")
        
        precomputed_strips[cls_id] = (matrix_img_strip, matrix_info_strip)
    
    # 3. Phase B: Sequential Inference Loop
    all_files = list(IMAGES_DIR.glob("*.jpg"))
    targets = [
        f for f in all_files 
        if f.name not in REFERENCE_FILENAMES 
        and not f.name.startswith("reference_")
    ]
    print(f"Found {len(targets)} Target images.")

    # Enable ALL Classes
    # TARGET_CLASSES = list(CLASSES.keys())
    
    # TARGET_CLASSES = list(CLASSES.keys())
    
    # Process ONLY FA DC (Class 3) for Verification
    TARGET_CLASSES = [3] 
    
    # SEQUENTIAL LOOP: Class -> Image
    for target_cls in TARGET_CLASSES:
        cls_name = CLASSES.get(target_cls, f"class_{target_cls}")
        


        print(f"\n=== Processing Class: {cls_name} ({target_cls}) ===")
        
        if target_cls not in precomputed_strips:
            print(f"Skipping {cls_name} (No reference strip)")
            continue
            
        matrix_img_strip, matrix_info_strip = precomputed_strips[target_cls]

        # Initialize Model Per Class
        config = get_config()
        
        # Per-Class Confidence Overrides
        current_conf = config.confidence_threshold
        if target_cls == 1: # Capacitor
            current_conf = 0.40
            print(f"  [INFO] Capacitor: Lowering confidence threshold to {current_conf}")
        elif target_cls == 2: # Inductor
            current_conf = 0.40
            print(f"  [INFO] Inductor: Lowering confidence threshold to {current_conf}")
        elif target_cls == 0: # Resistor
            current_conf = 0.60
            print(f"  [INFO] Resistor: Setting confidence threshold to {current_conf}")
        elif target_cls == 3: # FA DC
            current_conf = 0.15
            print(f"  [INFO] FA DC: Lowering confidence threshold to {current_conf}")
        elif target_cls == 4: # FA AC
            current_conf = 0.55
            print(f"  [INFO] FA AC: Setting confidence threshold to {current_conf}")
            
        print(f"Initializing SAM3 Model for {cls_name} (Conf: {current_conf})...")
        sam_model = SAM3Model(
            model_id=config.model_id,
            confidence_threshold=current_conf,
            mask_threshold=config.mask_threshold,
            device=config.device,
            hf_token=config.hf_token
        )
        sam_model.initialize()
        
        cls_output_dir = OUTPUT_DIR / cls_name
        cls_process_dir = cls_output_dir / "process"
        cls_result_dir = cls_output_dir / "result"
        
        cls_output_dir.mkdir(parents=True, exist_ok=True)
        cls_process_dir.mkdir(parents=True, exist_ok=True)
        cls_result_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            for target_path in targets:
                target_filename = target_path.name
                
                img_id = filename_to_id.get(target_filename, 0)
                
                # Filter removed for production run
                # if "electrical_242" not in target_filename:
                #      continue

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

                # Dynamic Scaling of the Strip with Padding
                m_h_base, m_w_base = matrix_img_strip.shape[:2]
                
                # Target Scale Ratio
                # Default: 1.0 (Match target height)
                # Capacitor (Class 1): 0.75 (User Request to reduce size relative to target)
                
                if target_cls in [0, 1, 2, 3, 4]:
                    # Adaptive Scaling: Match Target Height (User Request)
                    # Use 1.0 ratio to scale reference up/down with target image.
                    TARGET_REF_SCALE_RATIO = 1.0
                else:
                    TARGET_REF_SCALE_RATIO = 1.0
                    
                if TARGET_REF_SCALE_RATIO is None:
                    # Fixed Native Height
                    target_strip_h = m_h_base
                else:
                    # Relative Height
                    target_strip_h = int(t_h * TARGET_REF_SCALE_RATIO)
                
                # SAFETY CLAMP: Ensure reference strip does not exceed target image height
                # (Prevents broadcast errors on very small images)
                if target_strip_h > t_h:
                    print(f"  [WARN] constrained ref height {target_strip_h} to target height {t_h}")
                    target_strip_h = t_h

                scale = target_strip_h / m_h_base
                new_m_w = int(m_w_base * scale)
                new_m_h_content = target_strip_h
                
                # Resize Content
                matrix_img_content = cv2.resize(matrix_img_strip, (new_m_w, new_m_h_content), interpolation=cv2.INTER_LINEAR)
                
                # Create Padded Strip (Full Target Height)
                # Since we clamped target_strip_h <= t_h, this is safe.
                matrix_img_scaled = np.ones((t_h, new_m_w, 3), dtype=np.uint8) * 255
                
                # Center Vertically
                y_offset = (t_h - new_m_h_content) // 2
                matrix_img_scaled[y_offset:y_offset+new_m_h_content, :] = matrix_img_content
                
                matrix_info_scaled = []
                matrix_info_scaled = []
                # Margin for prompts to avoid being too tight after scaling
                # REVERT: Margin seemed to hurt performance? Testing with 0.
                PROMPT_MARGIN = 0 
                print(f"  [DEBUG] Scaling Reference: Ratio={target_strip_h/m_h_base:.3f} (Base: {m_h_base}x{m_w_base} -> New: {target_strip_h}x{new_m_w})")

                for cell in matrix_info_strip:
                    x1, y1, x2, y2 = cell['box']
                    nx1, ny1, nx2, ny2 = int(x1*scale), int(y1*scale), int(x2*scale), int(y2*scale)
                    
                    # Apply Vertical Offset
                    ny1 += y_offset
                    ny2 += y_offset
                    
                    # Relax the box slightly
                    nx1 = max(0, nx1 - PROMPT_MARGIN)
                    ny1 = max(0, ny1 - PROMPT_MARGIN)
                    nx2 = min(new_m_w, nx2 + PROMPT_MARGIN)
                    ny2 = min(target_strip_h + y_offset, ny2 + PROMPT_MARGIN) # Ensure inside content area

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
                
                # DEBUG: Save Prompt Visualization
                debug_prompt_path = cls_process_dir / f"{target_filename}_prompts.jpg"
                cv2.imwrite(str(debug_prompt_path), prompt_vis)
                print(f"  [DEBUG] Saved prompt visualization to {debug_prompt_path}")
                print(f"  [DEBUG] Active Prompts: {active_prompts}")
                
                if not active_prompts: 
                    del target_img, canvas, canvas_pil
                    continue

                # INFERENCE
                try:
                    res = sam_model.predict_with_boxes(
                        image=canvas_pil,
                        boxes_xyxy=active_prompts,
                        labels=active_labels
                    )
                except Exception as e:
                     print(f"Model Inference Failed for {target_filename}: {e}")
                     continue

                d_boxes = res.get('boxes', [])
                scores = res.get('scores', [])
                
                # Check Scores
                if torch.is_tensor(scores):
                    if scores.numel() == 0:
                        scores = []
                    else:
                        scores = scores.tolist()
                
                if not scores: 
                     scores = [1.0] * len(d_boxes)

                # Process Results (RAW)
                result_vis = target_img.copy()
                valid_count = 0
                
                for i, box in enumerate(d_boxes):
                    x1, y1, x2, y2 = box
                    score = scores[i]
                    cx = (x1 + x2) / 2
                    
                    # Coordinate Shift (new_m_w is already the scaled width from above)
                    # Do NOT redefine it using matrix_img_strip.shape[1]
                    
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
                        
                        # Draw detection
                        # Simplified Label: Score ONLY (User Request)
                        label_text = f"{score:.2f}"
                        cv2.rectangle(result_vis, (int(nx1), int(ny1)), (int(nx2), int(ny2)), (0, 255, 0), 2)
                        cv2.putText(result_vis, label_text, (int(nx1), int(ny1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Save Input Visualization (Process) - Always
                # Re-create process view (canvas from RGB to BGR)
                process_view = cv2.cvtColor(np.array(canvas_pil), cv2.COLOR_RGB2BGR)
                
                # Overlay the prompt elements (green boxes) on the left side
                if prompt_vis is not None and process_view.shape[0] == prompt_vis.shape[0]:
                     # Ensure width matches new_m_w
                     valid_w = min(new_m_w, prompt_vis.shape[1])
                     process_view[:, :valid_w, :] = prompt_vis[:, :valid_w, :]

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
            print(f"Cleaning up model for {cls_name}...")
            del sam_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # End of Image Loop for this Class -> Save Per-Class JSON

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
