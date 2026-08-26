"""
Multi-Class Diabetic Retinopathy Classifier (PyTorch)
Backbone: ResNet50 / EfficientNet / MobileNetV2 fine-tuned for 5-class DR Severity Grading:
0: No DR
1: Mild
2: Moderate
3: Severe
4: Proliferative DR
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Tuple, Any, Optional
import logging

from app.config import NUM_CLASSES, DR_CLASSES, DR_PROGRESSION_RISK, DR_CLINICAL_RECOMMENDATIONS, DEFAULT_BACKBONE
from app.preprocessing import extract_retinal_lesion_features

logger = logging.getLogger(__name__)


class RetinalDRClassifier(nn.Module):
    def __init__(self, backbone_name: str = DEFAULT_BACKBONE, pretrained: bool = True, num_classes: int = NUM_CLASSES):
        super(RetinalDRClassifier, self).__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes

        if "efficientnet" in self.backbone_name or "b4" in self.backbone_name:
            weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b4(weights=weights)
            in_features = self.backbone.classifier[1].in_features  # 1792
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features, self.num_classes)
            )
            self.target_layer = self.backbone.features[-1]
        elif "resnet50" in self.backbone_name:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, self.num_classes)
            self.target_layer = self.backbone.layer4[-1]
        elif "mobilenet" in self.backbone_name:
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v2(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Linear(in_features, self.num_classes)
            self.target_layer = self.backbone.features[-1]
        else:
            weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b4(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features, self.num_classes)
            )
            self.target_layer = self.backbone.features[-1]

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts deep feature vector prior to final classification head."""
        return self.backbone.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class DRInferenceEngine:
    def __init__(self, weights_path: Optional[str] = None, backbone_name: str = DEFAULT_BACKBONE):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing DR Inference Engine with {backbone_name} on device: {self.device}")
        
        self.model = RetinalDRClassifier(backbone_name=backbone_name, pretrained=True)
        
        if weights_path is None:
            eff_weights = os.path.abspath(os.path.join(os.path.dirname(__file__), "weights", "dr_efficientnet_weights.pth"))
            res_weights = os.path.abspath(os.path.join(os.path.dirname(__file__), "weights", "dr_resnet50_weights.pth"))
            if os.path.exists(eff_weights):
                weights_path = eff_weights
            elif os.path.exists(res_weights):
                weights_path = res_weights

        if weights_path:
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded custom DR weights from {weights_path}")
            except Exception as e:
                logger.warning(f"Could not load custom weights from {weights_path} (using pretrained backbone): {e}")

        self.model.to(self.device)
        self.model.eval()

    def predict(self, input_tensor: torch.Tensor, raw_rgb_image: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Executes model forward pass in eval mode with torch.no_grad() and returns predicted DR grade,
        confidence, raw logits, and class probabilities.
        """
        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError("Input must be a PyTorch Tensor.")
        
        if input_tensor.ndim != 4 or input_tensor.shape[1] != 3:
            raise ValueError(f"Input tensor must have shape [B, 3, H, W], got {input_tensor.shape}")

        input_tensor = input_tensor.to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_class_id = int(np.argmax(probs))
        confidence = float(probs[pred_class_id])
        raw_logits_list = logits.squeeze(0).cpu().numpy().tolist()

        # Dynamic Tensor & Output Console Diagnostic Prints (Requirements 1 & Section Diagnostics)
        tensor_mean = input_tensor.mean().item()
        tensor_std = input_tensor.std().item()
        print(f"[Model Inference] Input Image Shape: {input_tensor.shape}")
        print(f"[Model Inference] Tensor Mean: {tensor_mean:.5f}, Tensor Std: {tensor_std:.5f}")
        print(f"[Model Inference] Tensor Min/Max: ({input_tensor.min().item():.3f}, {input_tensor.max().item():.3f})")
        print(f"[Model Inference] Raw Model Logits: {raw_logits_list}")
        print(f"[Model Inference] Softmax Probabilities: {probs.tolist()}")
        print(f"[Model Inference] Predicted Class Index: {pred_class_id} ({DR_CLASSES[pred_class_id]})")

        logger.info(f"--- ML MODEL INFERENCE DIAGNOSTICS ---")
        logger.info(f"Input Image Shape: {input_tensor.shape} | Min/Max: ({input_tensor.min().item():.3f}, {input_tensor.max().item():.3f})")
        logger.info(f"Raw Model Logits: {raw_logits_list}")
        logger.info(f"Softmax Probabilities: {probs.tolist()}")
        logger.info(f"Argmax Class Index: {pred_class_id}")
        logger.info(f"Softmax Confidence Score: {confidence * 100:.2f}%")
        logger.info(f"---------------------------------------")

        class_probabilities = {
            DR_CLASSES[i]: float(prob) for i, prob in enumerate(probs)
        }

        progression_risk = DR_PROGRESSION_RISK.get(pred_class_id, "Unknown")
        clinical_recommendation = DR_CLINICAL_RECOMMENDATIONS.get(pred_class_id, "Consult ophthalmologist.")

        lesion_metrics = None
        if raw_rgb_image is not None:
            try:
                lesion_metrics = extract_retinal_lesion_features(raw_rgb_image)
            except Exception as e:
                logger.warning(f"Lesion feature extraction error: {e}")

        return {
            "predicted_class_id": pred_class_id,
            "predicted_class_name": DR_CLASSES[pred_class_id],
            "confidence": confidence,
            "confidence_score": confidence,
            "confidence_percentage": round(confidence * 100, 2),
            "probabilities": class_probabilities,
            "all_class_probabilities": class_probabilities,
            "progression_risk": progression_risk,
            "clinical_recommendation": clinical_recommendation,
            "raw_logits": raw_logits_list,
            "lesion_metrics": lesion_metrics,
            "device": str(self.device)
        }


