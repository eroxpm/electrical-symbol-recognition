"""
Visualization utilities for electrical symbol recognition.
Includes popup display functionality.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Any, Tuple

from PIL import Image


def overlay_masks(
    image: Image.Image,
    masks: Any,
    alpha: float = 0.5,
) -> Image.Image:
    """
    Overlay segmentation masks on image with different colors.
    
    Args:
        image: Original PIL Image
        masks: Tensor of masks [N, H, W]
        alpha: Transparency of masks (0-1)
        
    Returns:
        PIL Image with overlaid masks
    """
    image = image.convert("RGBA")
    
    # Handle empty masks
    if masks is None or len(masks) == 0:
        return image
    
    # Convert masks to numpy
    if hasattr(masks, 'cpu'):
        masks = masks.cpu().numpy()
    masks = (255 * masks).astype(np.uint8)
    
    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(max(n_masks, 1))
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]
    
    for mask, color in zip(masks, colors):
        mask_img = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha_mask = mask_img.point(lambda v: int(v * alpha))
        overlay.putalpha(alpha_mask)
        image = Image.alpha_composite(image, overlay)
    
    return image


def draw_boxes_on_image(
    image: Image.Image,
    boxes: Any,
    scores: Optional[Any] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
) -> Image.Image:
    """
    Draw bounding boxes on image.
    
    Args:
        image: PIL Image
        boxes: Tensor of boxes [N, 4] in xyxy format
        scores: Optional scores for each box
        color: RGB color tuple
        thickness: Line thickness
        
    Returns:
        PIL Image with drawn boxes
    """
    from PIL import ImageDraw, ImageFont
    
    image = image.copy()
    draw = ImageDraw.Draw(image)
    
    if boxes is None or len(boxes) == 0:
        return image
    
    # Convert to numpy if needed
    if hasattr(boxes, 'cpu'):
        boxes = boxes.cpu().numpy()
    if scores is not None and hasattr(scores, 'cpu'):
        scores = scores.cpu().numpy()
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        
        if scores is not None:
            score = float(scores[i])
            label = f"{score:.2f}"
            draw.text((x1, y1 - 15), label, fill=color)
    
    return image


def plot_results(
    image: Image.Image,
    results: dict,
    title: str = "Detection Results",
    save_path: Optional[Path] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 8),
) -> Optional[plt.Figure]:
    """
    Plot detection results with masks and boxes.
    
    Args:
        image: Original PIL Image
        results: Dict with 'masks', 'boxes', 'scores'
        title: Plot title
        save_path: Optional path to save figure
        show: Whether to display the plot
        figsize: Figure size
        
    Returns:
        matplotlib Figure if created
    """
    masks = results.get("masks", [])
    boxes = results.get("boxes", [])
    scores = results.get("scores", [])
    
    # Create visualization
    result_image = overlay_masks(image.copy(), masks)
    result_image = draw_boxes_on_image(result_image.convert("RGB"), boxes, scores)
    
    if show or save_path:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(result_image)
        ax.set_title(f"{title} - Found {len(masks)} object(s)")
        ax.axis("off")
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"Saved visualization to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    return None


def show_results_popup(
    image: Image.Image,
    results: dict,
    title: str = "SAM3 Detection Results",
) -> None:
    """
    Show results in a popup window (blocking).
    
    Args:
        image: Original PIL Image
        results: Dict with 'masks', 'boxes', 'scores'
        title: Window title
    """
    masks = results.get("masks", [])
    boxes = results.get("boxes", [])
    scores = results.get("scores", [])
    
    # Create visualization
    result_image = overlay_masks(image.copy(), masks)
    result_image = draw_boxes_on_image(result_image.convert("RGB"), boxes, scores)
    
    # Use matplotlib for popup
    plt.figure(figsize=(14, 10))
    plt.imshow(result_image)
    plt.title(f"{title}\nFound {len(masks)} object(s)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def save_masks(
    masks: Any,
    output_dir: Path,
    base_name: str = "mask",
) -> List[Path]:
    """
    Save individual masks as PNG files.
    
    Args:
        masks: Tensor of masks [N, H, W]
        output_dir: Directory to save masks
        base_name: Base name for mask files
        
    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    if masks is None or len(masks) == 0:
        return saved_paths
    
    # Convert to numpy if needed
    if hasattr(masks, 'cpu'):
        masks = masks.cpu().numpy()
    
    for i, mask in enumerate(masks):
        mask_img = (mask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_img)
        path = output_dir / f"{base_name}_{i:03d}.png"
        mask_pil.save(path)
        saved_paths.append(path)
    
    print(f"Saved {len(saved_paths)} masks to {output_dir}")
    return saved_paths
