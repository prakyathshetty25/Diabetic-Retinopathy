"""
Configuration parameters for Universal Retinal Screening framework.
"""

from typing import Dict, List, Tuple

# Severity scale mapping based on International Clinical Diabetic Retinopathy (ICDR) scale
DR_CLASSES: Dict[int, str] = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR"
}

DR_PROGRESSION_RISK: Dict[int, str] = {
    0: "Low",
    1: "Mild",
    2: "Moderate",
    3: "High",
    4: "Critical"
}

DR_CLINICAL_RECOMMENDATIONS: Dict[int, str] = {
    0: "Routine annual eye screening.",
    1: "Follow-up screening in 6-12 months.",
    2: "Ophthalmologist referral within 3-6 months.",
    3: "Urgent ophthalmic consultation within 1 month.",
    4: "Immediate specialized laser/surgical intervention."
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

# Image Preprocessing Setup (224x224 standard input resolution for ResNet50)
IMAGE_SIZE: Tuple[int, int] = (224, 224)
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

# Model Configuration
DEFAULT_BACKBONE: str = "resnet50"
NUM_CLASSES: int = 5

# API Configuration
HOST: str = "0.0.0.0"
PORT: int = 8000

