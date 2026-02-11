
import json
import cv2
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Current assumed mapping (to be verified)
# 0: FA AC, 1: FA DC, 2: Inductor, 3: Capacitor, 4: Resistor
# Corrected Mapping
CLASSES = {
    0: "Resistor",
    1: "Capacitor",
    2: "Inductor",
    3: "FA DC",
    4: "FA AC"
}

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "input" / "raw"
LABEL_DIR = RAW_DIR / "labels"
OUTPUT_JSON_PATH = DATA_DIR / "_annotations.coco.json"
DEBUG_DIR = DATA_DIR / "debug_classes"

def main():
    print("Starting YOLO to COCO conversion...")
    
    # Setup Debug Directory
    if DEBUG_DIR.exists():
        shutil.rmtree(DEBUG_DIR)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    categories = [{"id": k, "name": v} for k, v in CLASSES.items()]
    
    ann_id_counter = 0
    img_id_counter = 0
    
    # Store one example crop per class for verification
    examples_saved = {k: 0 for k in CLASSES.keys()}
    MAX_EXAMPLES = 3

    # List all images
    image_files = list(RAW_DIR.glob("*.jpg"))
    print(f"Found {len(image_files)} images.")
    
    for img_path in image_files:
        filename = img_path.name
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Error reading {filename}")
            continue
            
        h, w = img.shape[:2]
        img_id_counter += 1
        
        # Add Image entry
        images.append({
            "id": img_id_counter,
            "file_name": filename,
            "height": h,
            "width": w,
            "date_captured": datetime.now().isoformat()
        })
        
        # Read Label file
        label_file = LABEL_DIR / filename.replace(".jpg", ".txt")
        if not label_file.exists():
            continue
            
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                
                cls_id = int(parts[0])
                xc, yc, nw, nh = map(float, parts[1:5])
                
                # Convert YOLO (norm center_x, center_y, w, h) to COCO (abs x, y, w, h)
                abs_w = nw * w
                abs_h = nh * h
                abs_x = (xc * w) - (abs_w / 2)
                abs_y = (yc * h) - (abs_h / 2)
                
                ann_id_counter += 1
                annotations.append({
                    "id": ann_id_counter,
                    "image_id": img_id_counter,
                    "category_id": cls_id,
                    "bbox": [abs_x, abs_y, abs_w, abs_h],
                    "area": abs_w * abs_h,
                    "iscrowd": 0
                })
                
                # Save Debug Crop
                if examples_saved.get(cls_id, 0) < MAX_EXAMPLES:
                    x1, y1 = int(abs_x), int(abs_y)
                    x2, y2 = int(abs_x + abs_w), int(abs_y + abs_h)
                    
                    # Clip to image bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        crop = img[y1:y2, x1:x2]
                        cls_name = CLASSES.get(cls_id, "Unknown")
                        debug_fname = f"class_{cls_id}_{cls_name}_{img_id_counter}_{ann_id_counter}.jpg"
                        cv2.imwrite(str(DEBUG_DIR / debug_fname), crop)
                        examples_saved[cls_id] += 1

    # Build COCO dict
    coco_output = {
        "info": {
            "year": datetime.now().year,
            "version": "1.0",
            "description": "Converted from YOLO format",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    # Save JSON
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(coco_output, f, indent=4)
        
    print(f"Conversion complete!")
    print(f"Saved {len(images)} images and {len(annotations)} annotations to {OUTPUT_JSON_PATH}")
    print(f"Debug crops saved to {DEBUG_DIR}")

if __name__ == "__main__":
    main()
