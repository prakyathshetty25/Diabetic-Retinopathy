"""
Explainable AI (XAI) Engine via Grad-CAM (Gradient-Weighted Class Activation Mapping)
Extracts spatial feature maps from final convolutional layer to identify visual DR lesion indicators
(Microaneurysms, hemorrhages, hard exudates, cotton wool spots).
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Generates Grad-CAM activation heatmap normalized to [0, 1].
        """
        self.model.zero_grad()
        
        # Forward pass
        input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)
        
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]
        score.backward()

        # Compute weights \alpha_k^c
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)  # [C, 1, 1]

        # Weighted combination of feature maps
        cam = torch.sum(weights * activations, dim=0)  # [H, W]

        # Apply ReLU to keep only positive influence features
        cam = F.relu(cam)

        # Normalize to [0.0, 1.0]
        cam_np = cam.cpu().numpy()
        if cam_np.max() > 0:
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()


def generate_gradcam_overlay(
    model: nn.Module,
    target_layer: nn.Module,
    input_tensor: torch.Tensor,
    original_rgb: np.ndarray,
    target_class: int = None,
    alpha: float = 0.55
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generates high-resolution Grad-CAM heatmap overlay onto original fundus scan.
    Returns:
        (overlay_rgb, heatmap_jet_rgb, spatial_lesion_summary)
    """
    grad_cam = GradCAM(model, target_layer)
    try:
        raw_cam = grad_cam.generate_heatmap(input_tensor, target_class=target_class)
    finally:
        grad_cam.remove_hooks()

    h, w = original_rgb.shape[:2]
    # Resize activation map to original image resolution
    cam_resized = cv2.resize(raw_cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_uint8 = np.uint8(255 * cam_resized)

    # Apply Jet color map (Red = high activation/lesion area, Blue = normal background)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Alpha blending with original fundus scan
    overlay = cv2.addWeighted(original_rgb, 1 - alpha, heatmap_rgb, alpha, 0)

    # Compute spatial statistics for RAG report integration
    threshold_high = cam_resized > 0.6
    threshold_med = (cam_resized > 0.3) & (cam_resized <= 0.6)
    
    high_coverage_pct = float(np.sum(threshold_high) / (h * w) * 100)
    med_coverage_pct = float(np.sum(threshold_med) / (h * w) * 100)
    
    # Peak activation region centroid (e.g. Macula, Optic Disc, Quadrant)
    if np.any(threshold_high):
        y_indices, x_indices = np.where(threshold_high)
        centroid_x = float(np.mean(x_indices) / w)
        centroid_y = float(np.mean(y_indices) / h)
    else:
        centroid_x, centroid_y = 0.5, 0.5

    # Determine focal spatial distribution
    quadrant = "Central Macular Region"
    if centroid_x < 0.5 and centroid_y < 0.5:
        quadrant = "Nasal Superior Quadrant"
    elif centroid_x >= 0.5 and centroid_y < 0.5:
        quadrant = "Temporal Superior Quadrant"
    elif centroid_x < 0.5 and centroid_y >= 0.5:
        quadrant = "Nasal Inferior Quadrant"
    elif centroid_x >= 0.5 and centroid_y >= 0.5:
        quadrant = "Temporal Inferior Quadrant"

    spatial_summary = {
        "high_lesion_density_pct": round(high_coverage_pct, 2),
        "moderate_lesion_density_pct": round(med_coverage_pct, 2),
        "primary_lesion_quadrant": quadrant,
        "max_intensity": float(np.max(cam_resized)),
        "mean_intensity": float(np.mean(cam_resized))
    }

    return overlay, heatmap_rgb, spatial_summary
