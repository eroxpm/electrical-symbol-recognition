# Electrical Symbol Recognition — SAM3

Detection of electrical symbols (resistors, capacitors, inductors, etc.) in schematic images using **Meta's SAM3** (Segment Anything Model 3) with visual prompting.

## How It Works

1. **Reference crops** are extracted from COCO-annotated schematics
2. A **visual-prompt strip** is built per class (multi-scale reference matrix)
3. SAM3 runs inference on `[reference strip | target image]` canvases
4. **Class-agnostic NMS** (coverage-based) filters overlapping detections
5. Results are saved as COCO JSON + annotated images + metrics

## Quick Start (Docker)

### Prerequisites
- Docker with **NVIDIA GPU support** (`nvidia-container-toolkit`)
- HuggingFace token with access to `facebook/sam3`

### Setup
```bash
echo "HF_TOKEN=hf_your_token_here" > .env
docker-compose build sam3
```

### Execution Modes

#### 1. CLI — Full Pipeline
Runs inference → NMS → annotated images → metrics:
```bash
docker-compose run --rm sam3
```

#### 2. CLI — Visualize Only
Regenerates annotated images + metrics from existing `output/results.json`:
```bash
docker-compose run --rm sam3 --visualize_only
```

#### 3. Gradio Web UI
Interactive two-tab dashboard:
```bash
docker-compose up ui
# Open http://localhost:7860
```

**Tab 1 — Single Image Debug:**
- Select one image, run detection, see annotated result + raw JSON + debug log

**Tab 2 — Batch Analysis & Metrics:**
- Run full pipeline or load existing results
- View per-class Precision/Recall/F1 table + Confusion Matrix
- Browse all annotated output images + reference mosaics

## Output Files

| File | Description |
|------|-------------|
| `output/results.json` | COCO-format detection results |
| `output/images/*.jpg` | Annotated schematic images |
| `output/metrics.csv` | Per-class Precision, Recall, F1 |
| `output/confusion_matrix.png` | Visual confusion matrix |

## Project Structure

```
├── app.py                      # Gradio Web UI (two-tab dashboard)
├── main.py                     # CLI entry point
├── src/
│   ├── config.py               # Paths, thresholds, class definitions
│   ├── detector.py             # SAM3 model wrapper + inference logic
│   ├── inference_engine.py     # Reusable engine (shared by CLI + UI)
│   ├── metrics.py              # Detection metrics (sklearn + pandas)
│   └── utils.py                # NMS, visualization, COCO helpers
├── data/
│   ├── input/                  # Target schematic images
│   ├── references/             # Generated reference strips
│   └── annotations.json        # COCO ground truth
├── models/                     # Cached HuggingFace models
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Configuration

All tuning parameters are in [`src/config.py`](src/config.py):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `CONFIDENCE_THRESHOLDS` | Per-class detection confidence | 0.15–0.55 |
| `VISUALIZATION_THRESHOLDS` | Per-class drawing threshold | 0.40–0.55 |
| `NMS_IOU_THRESHOLD` | Coverage threshold for NMS | 0.5 |
| `SIZE_FILTER_MULTIPLIER` | Max detection = N × reference size | 2.0 |
| `VISUALIZATION_DISABLED_CLASSES` | Classes hidden from output | `[3]` (FA DC) |

## Detected Classes

| ID | Class | Color | Status |
|----|-------|-------|--------|
| 0 | Resistor | Blue | Active |
| 1 | Capacitor | Green | Active |
| 2 | Inductor | Red | Active |
| 3 | FA DC | Cyan | Disabled |
| 4 | FA AC | Yellow | Active |

## License

MIT License
