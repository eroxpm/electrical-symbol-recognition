"""
SAM3 Model Wrapper using HuggingFace Transformers API.
Provides a clean interface to SAM3 functionality.
"""

import torch
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from PIL import Image


class SAM3Model:
    """Wrapper class for SAM3 using HuggingFace Transformers API."""
    
    def __init__(
        self,
        model_id: str = "facebook/sam3",
        confidence_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize SAM3 model.
        
        Args:
            model_id: HuggingFace model ID
            confidence_threshold: Confidence threshold for detections
            mask_threshold: Threshold for mask binarization
            device: Device to run model on ('cuda' or 'cpu')
            hf_token: HuggingFace token for model access
        """
        self.model_id = model_id
        self.confidence_threshold = confidence_threshold
        self.mask_threshold = mask_threshold
        self.hf_token = hf_token
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self.processor = None
        self._initialized = False
        
    def initialize(self) -> None:
        """Initialize the model and processor from HuggingFace."""
        if self._initialized:
            return
        
        # Login to HuggingFace if token provided
        if self.hf_token:
            from huggingface_hub import login
            login(token=self.hf_token)
        
        # Import and load model
        from transformers import Sam3Model, Sam3Processor
        
        # Use local cache directory in project
        from pathlib import Path
        cache_dir = Path(__file__).parent.parent.parent / "models" / "huggingface"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading SAM3 model from {self.model_id}...")
        print(f"Cache directory: {cache_dir}")
        self.model = Sam3Model.from_pretrained(
            self.model_id,
            cache_dir=str(cache_dir)
        ).to(self.device)
        self.processor = Sam3Processor.from_pretrained(
            self.model_id,
            cache_dir=str(cache_dir)
        )
        self._initialized = True
        print("Model loaded successfully!")
    
    def predict_with_text(
        self,
        image: Image.Image,
        text_prompt: str,
    ) -> Dict[str, Any]:
        """
        Predict objects using text prompt.
        
        Args:
            image: PIL Image to process
            text_prompt: Text description of objects to find
            
        Returns:
            Dict with 'masks', 'boxes', 'scores'
        """
        if not self._initialized:
            self.initialize()
        
        # Prepare inputs
        inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt"
        ).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.confidence_threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        
        return results
    
    def predict_with_box(
        self,
        image: Image.Image,
        box_xyxy: List[float],
        text_prompt: Optional[str] = None,
        is_positive: bool = True,
    ) -> Dict[str, Any]:
        """
        Predict objects using bounding box prompt.
        
        Args:
            image: PIL Image to process
            box_xyxy: Bounding box [x1, y1, x2, y2] in pixel coordinates
            text_prompt: Optional text prompt to combine with box
            is_positive: Whether box is positive (include) or negative (exclude)
            
        Returns:
            Dict with 'masks', 'boxes', 'scores'
        """
        if not self._initialized:
            self.initialize()
        
        # Prepare box inputs
        input_boxes = [[box_xyxy]]
        input_boxes_labels = [[1 if is_positive else 0]]
        
        # Prepare inputs
        inputs = self.processor(
            images=image,
            text=text_prompt,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
            return_tensors="pt"
        ).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.confidence_threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        
        return results
    
    def predict_batch(
        self,
        images: List[Image.Image],
        text_prompts: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Batch prediction for multiple images.
        
        Args:
            images: List of PIL Images
            text_prompts: List of text prompts (one per image)
            
        Returns:
            List of result dicts
        """
        if not self._initialized:
            self.initialize()
        
        # Prepare inputs
        inputs = self.processor(
            images=images,
            text=text_prompts,
            return_tensors="pt"
        ).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.confidence_threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist()
        )
        
        return results
