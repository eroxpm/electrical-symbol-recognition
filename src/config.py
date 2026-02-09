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
    sam3_root: Path
    bpe_path: Path
    input_dir: Path
    output_dir: Path
    models_dir: Path
    
    # Model settings
    confidence_threshold: float = 0.5
    device: str = "cuda"
    use_bfloat16: bool = True
    
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
            sam3_root=Path(cfg.get('sam3_root', project_root / 'libs' / 'sam3')),
            bpe_path=Path(cfg.get('bpe_path', '')),
            input_dir=Path(cfg.get('input_dir', project_root / 'data' / 'input')),
            output_dir=Path(cfg.get('output_dir', project_root / 'data' / 'output')),
            models_dir=Path(cfg.get('models_dir', project_root / 'models')),
            confidence_threshold=cfg.get('confidence_threshold', 0.5),
            device=cfg.get('device', 'cuda'),
            use_bfloat16=cfg.get('use_bfloat16', True),
            hf_token=cfg.get('hf_token', os.environ.get('HF_TOKEN')),
        )
    
    @classmethod
    def default(cls) -> "Config":
        """Create default configuration."""
        project_root = Path(__file__).parent.parent
        sam3_root = project_root / 'libs' / 'sam3'
        
        return cls(
            project_root=project_root,
            sam3_root=sam3_root,
            bpe_path=sam3_root / 'assets' / 'bpe_simple_vocab_16e6.txt.gz',
            input_dir=project_root / 'data' / 'input',
            output_dir=project_root / 'data' / 'output',
            models_dir=project_root / 'models',
        )


def get_config(config_path: Optional[str] = None) -> Config:
    """Get configuration, either from file or default."""
    if config_path and os.path.exists(config_path):
        return Config.from_yaml(config_path)
    return Config.default()
