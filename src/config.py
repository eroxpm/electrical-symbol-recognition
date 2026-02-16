"""
Configuration constants for Electrical Symbol Recognition with SAM3.

All tuning parameters, class definitions, and path helpers are consolidated here.
"""

import os
from pathlib import Path
from typing import Dict, Optional


# ============================================================
# Project Paths (Docker-friendly: defaults to /app inside container)
# ============================================================

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).parent.parent))

DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
REFERENCES_DIR = DATA_DIR / "references"
ANNOTATIONS_PATH = DATA_DIR / "annotations.json"

OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_IMAGES_DIR = OUTPUT_DIR / "images"
OUTPUT_JSON_PATH = OUTPUT_DIR / "results.json"

MODELS_DIR = PROJECT_ROOT / "models"

# ============================================================
# Class Definitions
# ============================================================

CLASSES: Dict[int, str] = {
    0: "Resistor",
    1: "Capacitor",
    2: "Inductor",
    3: "FA DC",
    4: "FA AC",
}

# ============================================================
# Model Settings
# ============================================================

MODEL_ID: str = "facebook/sam3"
DEFAULT_CONFIDENCE: float = 0.50
MASK_THRESHOLD: float = 0.50
DEVICE: str = os.environ.get("DEVICE", "cuda")
HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN")

# Per-class confidence overrides
CONFIDENCE_THRESHOLDS: Dict[int, float] = {
    0: 0.30,   # Resistor
    1: 0.30,   # Capacitor
    2: 0.30,   # Inductor
    3: 0.15,   # FA DC
    4: 0.55,   # FA AC
}

# ============================================================
# Reference Image Configuration
# ============================================================

# Images used as reference sources (contain annotated symbols)
REFERENCE_FILENAMES = [
    "354_png.rf.c40016ba7c239b8979addb26ae9b90ed.jpg",        # Resistor, Capacitor, FA DC
    "electrical_213_png.rf.af3dc4e0c9d9a75a6f6a84d3a8f12fbf.jpg",  # Inductor
    "346_png.rf.8e1a2047a66b822e791ffc4a9fe249a4.jpg",        # FA AC
]

# Custom reference crop overrides: class_id -> filename in references/ dir
# These take priority over debug_classes auto-extraction
CUSTOM_REFERENCE_CROPS: Dict[int, str] = {
    1: "class_1_Capacitor_1_1.jpg",   # Capacitor
    2: "class_2_Inductor_1_3.jpg",    # Inductor
}

# Custom FA DC: extracted from COCO annotation on a specific image
CUSTOM_FA_DC_SOURCE = "electrical_75_png.rf.5d96d5206833ea346077c38cc7950c68.jpg"

# ============================================================
# Matrix / Visual Prompt Configuration
# ============================================================

MATRIX_CROP_SIZE = (64, 64)  # Default crop cell size (overridden dynamically)
MATRIX_PADDING = 10
MATRIX_ROWS = 4

# Per-class crop expansion margins (percentage of bbox dim)
# 0.0 = tight crop, 0.15 = 15% expansion
CROP_MARGINS: Dict[int, float] = {
    0: 0.0,    # Resistor: tight crop
    1: 0.15,   # Capacitor: 15%
    2: 0.15,   # Inductor: 15%
    3: 0.0,    # FA DC: tight crop (custom extraction)
    4: 0.15,   # FA AC: 15%
}

# Multi-scale factors for Column 1 of the reference strip
SCALE_RANGES: Dict[int, tuple] = {
    0: (0.65, 1.35),   # Resistor:  ±35%
    1: (0.65, 1.35),   # Capacitor: ±35%
    2: (0.65, 1.35),   # Inductor:  ±35%
    3: (0.50, 1.50),   # FA DC:     ±50%
    4: (0.65, 1.35),   # FA AC:     ±35%
}

# Dynamic cell size buffer multiplier (for scaled variations)
CELL_SIZE_BUFFER = 1.40

# ============================================================
# Post-Processing
# ============================================================

NMS_IOU_THRESHOLD = 0.5
SIZE_FILTER_MULTIPLIER = 2.0  # Max detection size = 2x reference size

# ============================================================
# Visualization
# ============================================================

VISUALIZATION_THRESHOLDS: Dict[int, float] = {
    0: 0.40,   # Resistor: 0.40
    1: 0.40,   # Capacitor: 0.40
    2: 0.45,   # Inductor: 0.45
    3: 1.10,   # FA DC: Deactivated
    4: 0.55,   # FA AC: Keep as is
}

# Explicitly disable some classes from visualization
VISUALIZATION_DISABLED_CLASSES: list[int] = [3]  # FA DC

CLASS_COLORS: Dict[int, tuple] = {
    0: (255, 0, 0),      # Resistor:  Blue
    1: (0, 255, 0),      # Capacitor: Green
    2: (0, 0, 255),      # Inductor:  Red
    3: (255, 255, 0),    # FA DC:     Cyan
    4: (0, 255, 255),    # FA AC:     Yellow
}
