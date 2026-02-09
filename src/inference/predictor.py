"""
Predictor class for electrical symbol recognition.
High-level interface for running inference.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any

from PIL import Image

from src.config import Config, get_config
from src.model.sam3_wrapper import SAM3Model
from src.utils.visualization import (
    plot_results,
    show_results_popup,
    save_masks,
)


class SymbolPredictor:
    """High-level predictor for electrical symbol detection and segmentation."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the predictor.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or get_config()
        self.model = SAM3Model(
            model_id="facebook/sam3",
            confidence_threshold=self.config.confidence_threshold,
            hf_token=self.config.hf_token,
            device=self.config.device,
        )
        self._current_image = None
    
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load an image for processing.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded PIL Image
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        self._current_image = Image.open(image_path).convert("RGB")
        return self._current_image
    
    def predict(
        self,
        text_prompt: str,
        image: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:
        """
        Run prediction on the image.
        
        Args:
            text_prompt: Text description of objects to detect
            image: Optional image (uses loaded image if not provided)
            
        Returns:
            Dict with 'masks', 'boxes', 'scores'
        """
        if image is None:
            image = self._current_image
        
        if image is None:
            raise RuntimeError("No image loaded. Call load_image first or provide image.")
        
        results = self.model.predict_with_text(image, text_prompt)
        return results
    
    def predict_and_visualize(
        self,
        text_prompt: str,
        image: Optional[Image.Image] = None,
        save_visualization: bool = True,
        save_masks_separately: bool = False,
        output_name: Optional[str] = None,
        show_popup: bool = False,
    ) -> Dict[str, Any]:
        """
        Run prediction and generate visualizations.
        
        Args:
            text_prompt: Text description of objects to detect
            image: Optional image (uses loaded image if not provided)
            save_visualization: Whether to save the visualization plot
            save_masks_separately: Whether to save individual mask files
            output_name: Base name for output files
            show_popup: Whether to show results in a popup window
            
        Returns:
            Dict with 'masks', 'boxes', 'scores'
        """
        if image is None:
            image = self._current_image
        
        if image is None:
            raise RuntimeError("No image loaded. Provide image or call load_image first.")
        
        results = self.predict(text_prompt, image)
        
        n_objects = len(results.get("masks", []))
        print(f"Found {n_objects} object(s) for prompt: '{text_prompt}'")
        
        if output_name is None:
            output_name = text_prompt.replace(" ", "_").lower()
        
        if save_visualization:
            vis_path = self.config.output_dir / "visualizations" / f"{output_name}.png"
            plot_results(
                image,
                results,
                title=f"Detections: {text_prompt}",
                save_path=vis_path,
                show=False,
            )
        
        if save_masks_separately and n_objects > 0:
            masks_dir = self.config.output_dir / "masks" / output_name
            save_masks(results.get("masks"), masks_dir, base_name=output_name)
        
        if show_popup:
            show_results_popup(image, results, title=f"SAM3: {text_prompt}")
        
        return results
    
    def process_image(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        save_visualization: bool = True,
        save_masks_separately: bool = False,
        show_popup: bool = False,
    ) -> Dict[str, Any]:
        """
        Complete pipeline: load image, predict, and visualize.
        
        Args:
            image_path: Path to input image
            text_prompt: Text description of objects to detect
            save_visualization: Whether to save visualization
            save_masks_separately: Whether to save individual masks
            show_popup: Whether to show popup with results
            
        Returns:
            Dict with 'masks', 'boxes', 'scores'
        """
        image = self.load_image(image_path)
        
        output_name = Path(image_path).stem + "_" + text_prompt.replace(" ", "_").lower()
        
        return self.predict_and_visualize(
            text_prompt=text_prompt,
            image=image,
            save_visualization=save_visualization,
            save_masks_separately=save_masks_separately,
            output_name=output_name,
            show_popup=show_popup,
        )
