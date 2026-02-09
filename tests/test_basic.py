"""
Basic tests for the electrical symbol recognition project.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all modules can be imported."""
    from src.config import Config, get_config
    from src.model.sam3_wrapper import SAM3Model
    from src.inference.predictor import SymbolPredictor
    from src.utils.visualization import plot_results, save_masks, show_results_popup
    
    assert Config is not None
    assert SAM3Model is not None
    assert SymbolPredictor is not None
    assert plot_results is not None
    assert show_results_popup is not None
    print("✓ All imports successful!")


def test_config_default():
    """Test default configuration creation."""
    from src.config import Config
    
    config = Config.default()
    
    assert config.confidence_threshold == 0.5
    assert config.device == "cuda"
    assert config.model_id == "facebook/sam3"
    assert config.project_root.exists()
    print("✓ Default config test passed!")


def test_directory_structure():
    """Test that project directories exist."""
    dirs_to_check = [
        "configs",
        "data",
        "models",
        "src",
        "src/model",
        "src/inference",
        "src/utils",
        "scripts",
    ]
    
    for dir_path in dirs_to_check:
        full_path = project_root / dir_path
        assert full_path.exists(), f"Directory not found: {dir_path}"
    
    print("✓ Directory structure test passed!")


if __name__ == "__main__":
    test_imports()
    test_config_default()
    test_directory_structure()
    print("\n✅ All tests passed!")
