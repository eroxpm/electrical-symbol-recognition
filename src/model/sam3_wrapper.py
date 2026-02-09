"""
SAM3 Model Wrapper for electrical symbol recognition.
Provides a clean interface to SAM3 functionality.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, List, Any

import torch
from PIL import Image


class SAM3Model:
    """Wrapper class for SAM3 image segmentation model."""
    
    def __init__(
        self,
        sam3_root: Path,
        bpe_path: Path,
        confidence_threshold: float = 0.5,
        device: str = "cuda",
        use_bfloat16: bool = True,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize SAM3 model.
        
        Args:
            sam3_root: Path to SAM3 library root
            bpe_path: Path to BPE vocabulary file
            confidence_threshold: Confidence threshold for predictions
            device: Device to run model on ('cuda' or 'cpu')
            use_bfloat16: Whether to use bfloat16 precision
            hf_token: HuggingFace token for model access
        """
        self.sam3_root = sam3_root
        self.bpe_path = bpe_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.use_bfloat16 = use_bfloat16
        self.hf_token = hf_token
        
        self.model = None
        self.processor = None
        self._initialized = False
        
    def initialize(self) -> None:
        """Initialize the model and processor."""
        if self._initialized:
            return
            
        # Add SAM3 to path
        sam3_path = str(self.sam3_root)
        if sam3_path not in sys.path:
            sys.path.insert(0, sam3_path)
        
        # Configure PyTorch for performance
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if self.use_bfloat16:
                torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        
        # Login to HuggingFace if token provided
        if self.hf_token:
            from huggingface_hub import login
            login(token=self.hf_token)
        
        # Import and build model
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        
        self.model = build_sam3_image_model(bpe_path=str(self.bpe_path))
        self.Sam3Processor = Sam3Processor
        self._initialized = True
        
    def set_image(self, image: Image.Image) -> Any:
        """
        Set the image for processing.
        
        Args:
            image: PIL Image to process
            
        Returns:
            Inference state for the image
        """
        if not self._initialized:
            self.initialize()
            
        self.processor = self.Sam3Processor(
            self.model,
            confidence_threshold=self.confidence_threshold
        )
        self.current_image = image
        self.image_size = image.size  # (width, height)
        
        return self.processor.set_image(image)
    
    def predict_with_text(
        self,
        inference_state: Any,
        text_prompt: str,
    ) -> Tuple[List[Any], List[float], List[Any]]:
        """
        Predict objects using text prompt.
        
        Args:
            inference_state: State from set_image
            text_prompt: Text description of objects to find
            
        Returns:
            Tuple of (boxes, scores, masks)
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call set_image first.")
            
        boxes, scores, masks = self.processor.predict(
            inference_state,
            text_prompt=text_prompt,
        )
        
        return boxes, scores, masks
    
    def predict_with_box(
        self,
        inference_state: Any,
        text_prompt: str,
        box_prompt: List[float],
    ) -> Tuple[List[Any], List[float], List[Any]]:
        """
        Predict objects using text and box prompt.
        
        Args:
            inference_state: State from set_image
            text_prompt: Text description of objects to find
            box_prompt: Bounding box [x, y, w, h] in normalized coordinates
            
        Returns:
            Tuple of (boxes, scores, masks)
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call set_image first.")
        
        from sam3.model.box_ops import box_xywh_to_cxcywh
        
        # Convert box coordinates
        width, height = self.image_size
        box_prompt_norm = [
            box_prompt[0] / width,
            box_prompt[1] / height,
            box_prompt[2] / width,
            box_prompt[3] / height,
        ]
        box_cxcywh = box_xywh_to_cxcywh(torch.tensor([box_prompt_norm]))
        
        boxes, scores, masks = self.processor.predict(
            inference_state,
            text_prompt=text_prompt,
            box_prompt=box_cxcywh,
        )
        
        return boxes, scores, masks
