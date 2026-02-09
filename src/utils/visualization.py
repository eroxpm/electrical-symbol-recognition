"""
Visualization utilities for electrical symbol recognition.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Any

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_results(
    image: Image.Image,
    boxes: List[Any],
    scores: List[float],
    masks: List[Any],
    title: str = "Detection Results",
    save_path: Optional[Path] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot detection results with boxes and masks.
    
    Args:
        image: Original PIL Image
        boxes: List of detected bounding boxes
        scores: List of confidence scores
        masks: List of segmentation masks
        title: Plot title
        save_path: Path to save the figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure if not showing
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    ax.set_title(f"{title} - Found {len(boxes)} object(s)")
    ax.axis('off')
    
    # Generate colors for each detection
    colors = plt.cm.rainbow(np.linspace(0, 1, max(len(boxes), 1)))
    
    width, height = image.size
    
    for idx, (box, score, mask) in enumerate(zip(boxes, scores, masks)):
        color = colors[idx]
        
        # Draw mask with transparency
        if mask is not None:
            mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.array(mask)
            if mask_np.ndim == 3:
                mask_np = mask_np[0]
            
            # Create colored mask
            colored_mask = np.zeros((*mask_np.shape, 4))
            colored_mask[mask_np > 0.5] = [*color[:3], 0.4]
            ax.imshow(colored_mask)
        
        # Draw bounding box
        if box is not None:
            box_np = box.cpu().numpy() if hasattr(box, 'cpu') else np.array(box)
            if box_np.ndim > 1:
                box_np = box_np[0]
            
            # Convert from normalized to pixel coordinates if needed
            if np.max(box_np) <= 1.0:
                x1, y1, x2, y2 = box_np * np.array([width, height, width, height])
            else:
                x1, y1, x2, y2 = box_np
            
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add score label
            ax.text(
                x1, y1 - 5,
                f'{score:.2f}',
                color='white',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8)
            )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    if show:
        plt.show()
        return None
    
    return fig


def draw_box_on_image(
    image: Image.Image,
    box: List[float],
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> Image.Image:
    """
    Draw a bounding box on an image.
    
    Args:
        image: PIL Image
        box: Box coordinates [x1, y1, x2, y2]
        color: RGB color tuple
        thickness: Line thickness
        
    Returns:
        Image with box drawn
    """
    from PIL import ImageDraw
    
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    x1, y1, x2, y2 = box
    for i in range(thickness):
        draw.rectangle(
            [x1 - i, y1 - i, x2 + i, y2 + i],
            outline=color
        )
    
    return img_copy


def save_masks(
    masks: List[Any],
    output_dir: Path,
    base_name: str = "mask",
) -> List[Path]:
    """
    Save individual masks as images.
    
    Args:
        masks: List of segmentation masks
        output_dir: Directory to save masks
        base_name: Base name for mask files
        
    Returns:
        List of paths to saved mask files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for idx, mask in enumerate(masks):
        mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.array(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        
        # Convert to uint8
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_uint8)
        
        save_path = output_dir / f"{base_name}_{idx:03d}.png"
        mask_img.save(save_path)
        saved_paths.append(save_path)
        
    return saved_paths
