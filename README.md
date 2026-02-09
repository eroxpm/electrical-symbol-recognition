# Electrical Symbol Recognition with SAM3

A Python project for electrical symbol recognition using Meta's SAM3 (Segment Anything Model 3) via HuggingFace Transformers.

## Features

- **Text-based segmentation**: Describe objects using natural language
- **Box prompts**: Use bounding boxes to guide segmentation
- **Batch processing**: Process multiple images efficiently
- **Popup visualization**: View results interactively

## Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- HuggingFace account with access to [facebook/sam3](https://huggingface.co/facebook/sam3)

### Setup

1. Create a conda environment:
```bash
conda create -n sam3-env python=3.10
conda activate sam3-env
```

2. Install PyTorch with CUDA:
```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Login to HuggingFace:
```bash
huggingface-cli login
```

## Usage

### Command Line

```bash
# Basic prediction with popup display
python scripts/predict.py --image path/to/image.jpg --prompt "object" --show

# Save visualization (no popup)
python scripts/predict.py --image path/to/image.jpg --prompt "resistor" --threshold 0.5

# Save individual masks
python scripts/predict.py --image path/to/image.jpg --prompt "capacitor" --save-masks
```

### Python API

```python
from src.inference.predictor import SymbolPredictor

# Initialize predictor
predictor = SymbolPredictor()

# Process an image
results = predictor.process_image(
    image_path="path/to/image.jpg",
    text_prompt="resistor",
    show_popup=True
)

print(f"Found {len(results['masks'])} objects")
```

### Low-level API

```python
from src.model.sam3_wrapper import SAM3Model
from PIL import Image

# Initialize model
model = SAM3Model(confidence_threshold=0.5)

# Load and process image
image = Image.open("path/to/image.jpg")
results = model.predict_with_text(image, "electrical component")

# Results contain: masks, boxes, scores
```

## Project Structure

```
electrical-symbol-recognition/
├── configs/
│   └── config.yaml        # Configuration file
├── data/
│   ├── input/             # Input images
│   └── output/            # Generated outputs
├── models/
│   └── huggingface/       # Cached HuggingFace models
├── scripts/
│   └── predict.py         # CLI prediction script
├── src/
│   ├── config.py          # Configuration management
│   ├── model/
│   │   └── sam3_wrapper.py    # SAM3 model wrapper
│   ├── inference/
│   │   └── predictor.py       # High-level predictor
│   └── utils/
│       └── visualization.py   # Visualization utilities
├── tests/
│   └── test_basic.py      # Basic tests
├── requirements.txt
└── README.md
```

## Configuration

Edit `configs/config.yaml`:

```yaml
model_id: facebook/sam3
confidence_threshold: 0.5
mask_threshold: 0.5
device: cuda
```

Or use environment variables:
```bash
export HF_TOKEN="your_huggingface_token"
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--image`, `-i` | Path to input image (required) |
| `--prompt`, `-p` | Text prompt for detection (required) |
| `--threshold`, `-t` | Confidence threshold (default: 0.5) |
| `--show` | Show results in popup window |
| `--save-masks` | Save individual mask files |
| `--no-save` | Don't save visualization |
| `--output-dir`, `-o` | Custom output directory |

## License

MIT License
