# Electrical Symbol Recognition with SAM3

A Python project for detecting and segmenting electrical symbols using SAM3 (Segment Anything Model 3).

## Project Structure

```
├── configs/                 # Configuration files
│   └── config.yaml         
├── data/                    
│   ├── input/              # Input images
│   │   ├── raw/            
│   │   └── processed/      
│   └── output/             # Results
│       ├── masks/          
│       └── visualizations/ 
├── models/                  # Model checkpoints
├── src/                     # Source code
│   ├── config.py           # Configuration management
│   ├── model/              # Model wrappers
│   ├── inference/          # Prediction logic
│   └── utils/              # Utilities
├── scripts/                 # Executable scripts
│   └── predict.py          # Main prediction script
├── libs/                    # External libraries (SAM3)
└── notebooks/               # Jupyter notebooks
```

## Installation

1. Activate the conda environment:
```bash
conda activate sam3-env
```

2. Install additional dependencies:
```bash
pip install -r requirements.txt
```

3. Set your HuggingFace token (optional, can also pass via CLI):
```bash
export HF_TOKEN="your_token_here"
```

## Usage

### Command Line

Basic usage with text prompt:
```bash
python scripts/predict.py --image data/input/raw/test.jpg --prompt "person"
```

With visualization display:
```bash
python scripts/predict.py --image data/input/raw/test.jpg --prompt "face" --show
```

Save individual masks:
```bash
python scripts/predict.py --image data/input/raw/test.jpg --prompt "object" --save-masks
```

Custom threshold:
```bash
python scripts/predict.py --image data/input/raw/test.jpg --prompt "person" --threshold 0.7
```

### Python API

```python
from src.inference.predictor import SymbolPredictor
from src.config import Config

# Initialize with default config
predictor = SymbolPredictor()

# Or with custom config
config = Config.default()
config.confidence_threshold = 0.7
predictor = SymbolPredictor(config)

# Process an image
boxes, scores, masks = predictor.process_image(
    image_path="data/input/raw/test.jpg",
    text_prompt="person",
    save_visualization=True,
    show=False,
)

print(f"Found {len(boxes)} objects")
```

## Configuration

Edit `configs/config.yaml` to customize:

- `confidence_threshold`: Detection confidence (0.0-1.0)
- `device`: 'cuda' or 'cpu'
- `use_bfloat16`: Enable bfloat16 for faster inference
- Path configurations

## CLI Options

| Option | Description |
|--------|-------------|
| `--image, -i` | Path to input image (required) |
| `--prompt, -p` | Text prompt for detection (required) |
| `--config, -c` | Path to config YAML |
| `--threshold, -t` | Confidence threshold (default: 0.5) |
| `--output-dir, -o` | Output directory |
| `--save-masks` | Save individual mask files |
| `--show` | Display visualization |
| `--no-save` | Don't save visualization |
| `--hf-token` | HuggingFace token |

## License

MIT
