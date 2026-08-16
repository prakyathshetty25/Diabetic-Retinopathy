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

from app.config import NUM_CLASSES, DR_CLASSES, DEFAULT_BACKBONE
from app.preprocessing import extract_retinal_lesion_features

logger = logging.getLogger(__name__)


class RetinalDRClassifier(nn.Module):
    def __init__(self, backbone_name: str = DEFAULT_BACKBONE, pretrained: bool = True, num_classes: int = NUM_CLASSES):
        super(RetinalDRClassifier, self).__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes

        if "resnet50" in self.backbone_name:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Strip fc head
            self.target_layer = self.backbone.layer4[-1]
        elif "efficientnet" in self.backbone_name:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            self.target_layer = self.backbone.features[-1]
        elif "mobilenet" in self.backbone_name:
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v2(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            self.target_layer = self.backbone.features[-1]
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Choose resnet50, efficientnet, or mobilenet.")

        # Custom DR classification head with Dropout for regularization
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, self.num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts deep feature vector prior to final classification head."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


class DRInferenceEngine:
    def __init__(self, weights_path: Optional[str] = None, backbone_name: str = DEFAULT_BACKBONE):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing DR Inference Engine on device: {self.device}")
        
        self.model = RetinalDRClassifier(backbone_name=backbone_name, pretrained=True)
        
        if weights_path is None:
            default_weights = os.path.abspath(os.path.join(os.path.dirname(__file__), "weights", "dr_resnet50_weights.pth"))
            if os.path.exists(default_weights):
                weights_path = default_weights

        if weights_path:
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded custom DR weights from {weights_path}")
            except Exception as e:
                logger.error(f"Failed to load weights from {weights_path}, using default backbone: {e}")

        self.model.to(self.device)
        self.model.eval()

    def predict(self, input_tensor: torch.Tensor, raw_rgb_image: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Executes model forward pass and returns predicted DR grade, confidence, and class probabilities.
        Optionally integrates CV lesion feature indicators if raw_rgb_image is provided.
        """
        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError("Input must be a PyTorch Tensor.")
        
        if input_tensor.ndim != 4 or input_tensor.shape[1] != 3:
            raise ValueError(f"Input tensor must have shape [B, 3, H, W], got {input_tensor.shape}")

        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            nn_probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        lesion_metrics = None
        final_probs = nn_probs.copy()

        if raw_rgb_image is not None:
            try:
                lesion_metrics = extract_retinal_lesion_features(raw_rgb_image)
                s_grade = lesion_metrics.get("suggested_grade", None)
                if s_grade is not None:
                    # Construct a clinical lesion feature distribution
                    lesion_prob = np.full(NUM_CLASSES, 0.05)
                    lesion_prob[s_grade] = 0.80
                    lesion_prob = lesion_prob / lesion_prob.sum()

                    # Hybrid fusion: 75% Deep Learning + 25% Lesion Feature Rules
                    final_probs = 0.75 * nn_probs + 0.25 * lesion_prob
                    final_probs = final_probs / final_probs.sum()
            except Exception as e:
                logger.warning(f"Lesion feature extraction error: {e}")

        pred_class_id = int(np.argmax(final_probs))
        confidence = float(final_probs[pred_class_id])

        class_probabilities = {
            DR_CLASSES[i]: float(prob) for i, prob in enumerate(final_probs)
        }

        return {
            "predicted_class_id": pred_class_id,
            "predicted_class_name": DR_CLASSES[pred_class_id],
            "confidence": confidence,
            "probabilities": class_probabilities,
            "raw_logits": logits.squeeze(0).cpu().numpy().tolist(),
            "lesion_metrics": lesion_metrics,
            "device": str(self.device)
        }

