"""
Automated unit & integration verification script for Universal Retinal Screening backend.
"""

import os
import sys

# Add backend directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

import torch
import numpy as np
from PIL import Image

from app.preprocessing import prepare_tensor_from_image
from app.model import DRInferenceEngine
from app.gradcam import generate_gradcam_overlay
from app.rag_engine import RAGClinicalReportGenerator
from app.synthetic_data import create_synthetic_fundus


def run_backend_tests():
    print("==================================================")
    print("RUNNING RETINAL SCREENING BACKEND VERIFICATION")
    print("==================================================")

    # 1. Test Synthetic Image Generation
    print("\n[1/5] Testing Synthetic Fundus Image Generator...")
    synthetic_img = create_synthetic_fundus(grade=3)  # Severe DR
    assert isinstance(synthetic_img, Image.Image), "Synthetic fundus must be a PIL Image."
    assert synthetic_img.size == (512, 512), f"Expected (512, 512), got {synthetic_img.size}"
    print(" -> PASSED! Generated synthetic fundus scan.")

    # 2. Test Preprocessing & Green Channel CLAHE Pipeline
    print("\n[2/5] Testing Preprocessing & Green Channel CLAHE Pipeline...")
    tensor, preprocessed_np = prepare_tensor_from_image(synthetic_img)
    assert tensor.shape == (1, 3, 224, 224), f"Expected tensor shape (1, 3, 224, 224), got {tensor.shape}"
    assert preprocessed_np.shape == (224, 224, 3), f"Expected RGB shape (224, 224, 3), got {preprocessed_np.shape}"
    print(f" -> PASSED! Tensor shape: {tensor.shape}, Preprocessed RGB shape: {preprocessed_np.shape}")

    # 3. Test Model Inference Engine & Diagnostics
    print("\n[3/5] Testing PyTorch Model Inference Engine & Diagnostics across Grades 0-4...")
    engine = DRInferenceEngine()
    for test_g in range(5):
        sample_img = create_synthetic_fundus(grade=test_g)
        g_tensor, g_preprocessed = prepare_tensor_from_image(sample_img)
        print(f"\n--- Testing Grade {test_g} Image ---")
        g_pred = engine.predict(g_tensor, raw_rgb_image=g_preprocessed)
        assert 0 <= g_pred['predicted_class_id'] <= 4, "Class ID out of bounds [0-4]."
        assert len(g_pred['all_class_probabilities']) == 5, "Must return 5 class probabilities."
    
    # Store prediction for downstream test steps
    prediction = engine.predict(tensor)
    print("\n -> PASSED! PyTorch model forward pass & structured JSON verification succeeded.")

    # 4. Test Grad-CAM XAI Generator
    print("\n[4/5] Testing Grad-CAM Heatmap & Overlay Engine...")
    overlay, heatmap, spatial_summary = generate_gradcam_overlay(
        model=engine.model,
        target_layer=engine.model.target_layer,
        input_tensor=tensor,
        original_rgb=preprocessed_np,
        target_class=prediction['predicted_class_id']
    )
    assert overlay.shape == (224, 224, 3), f"Overlay shape mismatch: {overlay.shape}"
    assert heatmap.shape == (224, 224, 3), f"Heatmap shape mismatch: {heatmap.shape}"
    print(f" -> Primary Lesion Quadrant: {spatial_summary['primary_lesion_quadrant']}")
    print(f" -> High Lesion Density Coverage: {spatial_summary['high_lesion_density_pct']}%")
    print(" -> PASSED! Grad-CAM heatmap extraction and overlay succeeded.")

    # 5. Test RAG Clinical Report Generator
    print("\n[5/5] Testing RAG Clinical Report Generator...")
    rag = RAGClinicalReportGenerator()
    report = rag.generate_report(
        predicted_grade=prediction['predicted_class_id'],
        confidence=prediction['confidence_score'],
        probabilities=prediction['all_class_probabilities'],
        spatial_summary=spatial_summary
    )
    print(f" -> Diagnosis: {report['summary_header']['diagnosis']}")
    print(f" -> Urgency: {report['summary_header']['urgency_level']}")
    print(f" -> ICDR Criteria Matched: {report['diagnostic_reasoning']['icdr_criteria_matched']}")
    print(f" -> Referral Timeline: {report['clinical_recommendations']['referral_timeline']}")
    assert len(report['retrieved_guideline_citations']) > 0, "RAG must return guideline citations."
    print(" -> PASSED! RAG clinical explanation report generated successfully.")

    print("\n==================================================")
    print(" ALL 5 BACKEND VERIFICATION TESTS PASSED SUCCESSFULLY! ")
    print("==================================================")


if __name__ == "__main__":
    run_backend_tests()
