"""
Predictor class for electrical symbol recognition.
High-level interface for running inference.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Any, Union

from PIL import Image

from src.config import Config, get_config
from src.model.sam3_wrapper import SAM3Model
from src.utils.visualization import plot_results, save_masks


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
            sam3_root=self.config.sam3_root,
            bpe_path=self.config.bpe_path,
            confidence_threshold=self.config.confidence_threshold,
            device=self.config.device,
            use_bfloat16=self.config.use_bfloat16,
            hf_token=self.config.hf_token,
        )
        self._inference_state = None
        self._current_image = None
        
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load and set an image for processing.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded PIL Image
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        self._current_image = Image.open(image_path).convert("RGB")
        self._inference_state = self.model.set_image(self._current_image)
        
        return self._current_image
    
    def predict(
        self,
        text_prompt: str,
        box_prompt: Optional[List[float]] = None,
    ) -> Tuple[List[Any], List[float], List[Any]]:
        """
        Run prediction on the loaded image.
        
        Args:
            text_prompt: Text description of objects to detect
            box_prompt: Optional bounding box [x, y, w, h] to focus prediction
            
        Returns:
            Tuple of (boxes, scores, masks)
        """
        if self._inference_state is None:
            raise RuntimeError("No image loaded. Call load_image first.")
        
        if box_prompt:
            boxes, scores, masks = self.model.predict_with_box(
                self._inference_state,
                text_prompt=text_prompt,
                box_prompt=box_prompt,
            )
        else:
            boxes, scores, masks = self.model.predict_with_text(
                self._inference_state,
                text_prompt=text_prompt,
            )
        
        return boxes, scores, masks
    
    def predict_and_visualize(
        self,
        text_prompt: str,
        box_prompt: Optional[List[float]] = None,
        save_visualization: bool = True,
        save_masks_separately: bool = False,
        output_name: Optional[str] = None,
        show: bool = False,
    ) -> Tuple[List[Any], List[float], List[Any]]:
        """
        Run prediction and generate visualizations.
        
        Args:
            text_prompt: Text description of objects to detect
            box_prompt: Optional bounding box to focus prediction
            save_visualization: Whether to save the visualization plot
            save_masks_separately: Whether to save individual mask files
            output_name: Base name for output files
            show: Whether to display the plot
            
        Returns:
            Tuple of (boxes, scores, masks)
        """
        boxes, scores, masks = self.predict(text_prompt, box_prompt)
        
        print(f"Found {len(boxes)} object(s) for prompt: '{text_prompt}'")
        
        if output_name is None:
            output_name = text_prompt.replace(" ", "_").lower()
        
        if save_visualization:
            vis_path = self.config.output_dir / "visualizations" / f"{output_name}.png"
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            plot_results(
                self._current_image,
                boxes, scores, masks,
                title=f"Detections: {text_prompt}",
                save_path=vis_path,
                show=show,
            )
        
        if save_masks_separately:
            masks_dir = self.config.output_dir / "masks" / output_name
            save_masks(masks, masks_dir, base_name=output_name)
        
        return boxes, scores, masks
    
    def process_image(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        box_prompt: Optional[List[float]] = None,
        save_visualization: bool = True,
        save_masks_separately: bool = False,
        show: bool = False,
    ) -> Tuple[List[Any], List[float], List[Any]]:
        """
        Complete pipeline: load image, predict, and visualize.
        
        Args:
            image_path: Path to input image
            text_prompt: Text description of objects to detect
            box_prompt: Optional bounding box to focus prediction
            save_visualization: Whether to save visualization
            save_masks_separately: Whether to save individual masks
            show: Whether to display the plot
            
        Returns:
            Tuple of (boxes, scores, masks)
        """
        self.load_image(image_path)
        
        output_name = Path(image_path).stem + "_" + text_prompt.replace(" ", "_").lower()
        
        return self.predict_and_visualize(
            text_prompt=text_prompt,
            box_prompt=box_prompt,
            save_visualization=save_visualization,
            save_masks_separately=save_masks_separately,
            output_name=output_name,
            show=show,
        )
