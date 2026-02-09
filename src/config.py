"""
Configuration management for electrical symbol recognition.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import yaml


@dataclass
class Config:
    """Configuration class for the project."""
    
    # Paths
    project_root: Path
    input_dir: Path
    output_dir: Path
    models_dir: Path
    
    # Model settings
    model_id: str = "facebook/sam3"
    confidence_threshold: float = 0.5
    mask_threshold: float = 0.5
    device: str = "cuda"
    
    # HuggingFace settings
    hf_token: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        project_root = Path(cfg.get('project_root', Path(__file__).parent.parent))
        
        return cls(
            project_root=project_root,
            input_dir=Path(cfg.get('input_dir', project_root / 'data' / 'input')),
            output_dir=Path(cfg.get('output_dir', project_root / 'data' / 'output')),
            models_dir=Path(cfg.get('models_dir', project_root / 'models')),
            model_id=cfg.get('model_id', 'facebook/sam3'),
            confidence_threshold=cfg.get('confidence_threshold', 0.5),
            mask_threshold=cfg.get('mask_threshold', 0.5),
            device=cfg.get('device', 'cuda'),
            hf_token=cfg.get('hf_token', os.environ.get('HF_TOKEN')),
        )
    
    @classmethod
    def default(cls) -> "Config":
        """Create default configuration."""
        project_root = Path(__file__).parent.parent
        
        return cls(
            project_root=project_root,
            input_dir=project_root / 'data' / 'input',
            output_dir=project_root / 'data' / 'output',
            models_dir=project_root / 'models',
        )


def get_config(config_path: Optional[str] = None) -> Config:
    """Get configuration, either from file or default."""
    if config_path and os.path.exists(config_path):
        return Config.from_yaml(config_path)
    return Config.default()
