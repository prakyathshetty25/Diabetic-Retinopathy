"""
Configuration parameters for Universal Retinal Screening framework.
"""

from typing import Dict, List, Tuple

# Severity scale mapping based on International Clinical Diabetic Retinopathy (ICDR) scale
DR_CLASSES: Dict[int, str] = {
    0: "No DR",
    1: "Mild Non-Proliferative DR",
    2: "Moderate Non-Proliferative DR",
    3: "Severe Non-Proliferative DR",
    4: "Proliferative DR"
}

DR_SEVERITY_COLORS: Dict[int, str] = {
    0: "#10B981",  # Green - Normal
    1: "#3B82F6",  # Blue - Mild
    2: "#F59E0B",  # Amber - Moderate
    3: "#EF4444",  # Red - Severe
    4: "#991B1B"   # Dark Red - Proliferative
}

DR_DESCRIPTIONS: Dict[int, str] = {
    0: "No retinal microvascular abnormalities detected.",
    1: "Microaneurysms present only.",
    2: "More than microaneurysms, but less than severe NPDR (e.g. hard exudates, retinal hemorrhages).",
    3: "Any of the following (4-2-1 rule): >20 intraretinal hemorrhages in 4 quadrants, venous beading in 2+ quadrants, or IRMA in 1+ quadrant.",
    4: "Neovascularization or vitreous/preretinal hemorrhage present."
}

# Image Preprocessing Setup
IMAGE_SIZE: Tuple[int, int] = (512, 512)
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

# Model Configuration
DEFAULT_BACKBONE: str = "resnet50"
NUM_CLASSES: int = 5

# API Configuration
HOST: str = "0.0.0.0"
PORT: int = 8000
