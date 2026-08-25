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

DR_PROGRESSION_RISK: Dict[int, str] = {
    0: "Low (Annual screening)",
    1: "Mild (Monitor 6-12 months)",
    2: "Moderate (Referral within 3-6 months)",
    3: "High (Urgent ophthalmic referral)",
    4: "Critical (Immediate intervention required)"
}

DR_CLINICAL_RECOMMENDATIONS: Dict[int, str] = {
    0: "Routine annual dilated eye examination and glycemic control monitoring.",
    1: "Follow-up dilated fundus exam within 6 to 12 months. Optimize blood glucose and blood pressure.",
    2: "Follow-up dilated fundus exam within 3 to 6 months. Consider OCT imaging to evaluate macular edema.",
    3: "Urgent referral to Retina Specialist / Ophthalmologist within 2 to 4 weeks. Evaluate for PRP laser or Anti-VEGF.",
    4: "Emergency referral to Retina Specialist within 24-48 hours. Urgent Anti-VEGF, PRP photocoagulation, or vitrectomy evaluation."
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

# Image Preprocessing Setup (224x224 standard input resolution for ResNet50 / MobileNet)
IMAGE_SIZE: Tuple[int, int] = (224, 224)
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

# Model Configuration
DEFAULT_BACKBONE: str = "resnet50"
NUM_CLASSES: int = 5

# API Configuration
HOST: str = "0.0.0.0"
PORT: int = 8000

