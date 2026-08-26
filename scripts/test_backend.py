"""
Automated unit & integration verification script for Universal Retinal Screening backend.
"""

import os
import sys
import io
import asyncio
import numpy as np
from PIL import Image

# Add backend directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

import torch
from app.preprocessing import preprocess_image, prepare_tensor_from_image
from app.model import DRInferenceEngine
from app.gradcam import generate_gradcam, generate_gradcam_overlay
from app.rag_engine import generate_clinical_report, RAGClinicalReportGenerator
from app.synthetic_data import create_synthetic_fundus
from app.main import _process_prediction


def run_backend_tests():
    print("==================================================")
    print("RUNNING RETINAL SCREENING BACKEND VERIFICATION")
    print("==================================================")

    # 1. Test Synthetic Image Generation
    print("\n[1/5] Testing Synthetic Fundus Image Generator...")
    synthetic_img = create_synthetic_fundus(grade=3)  # Severe DR
    assert isinstance(synthetic_img, Image.Image), "Synthetic fundus must be a PIL Image."
    assert synthetic_img.size == (512, 512), f"Expected (512, 512), got {synthetic_img.size}"
    
    img_byte_arr = io.BytesIO()
    synthetic_img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    print(" -> PASSED! Generated synthetic fundus scan.")

    # 2. Test Preprocessing & Green Channel CLAHE Pipeline (Requirement 1: 224x224)
    print("\n[2/5] Testing Preprocessing Pipeline (224x224)...")
    tensor, preprocessed_np = preprocess_image(img_bytes)
    assert tensor.shape == (1, 3, 224, 224), f"Expected tensor shape (1, 3, 224, 224), got {tensor.shape}"
    assert preprocessed_np.shape == (224, 224, 3), f"Expected RGB shape (224, 224, 3), got {preprocessed_np.shape}"
    print(f" -> PASSED! Tensor shape: {tensor.shape}, Preprocessed RGB shape: {preprocessed_np.shape}")

    # 3. Test Model Inference Engine & Diagnostics (Requirement 2 & 4)
    print("\n[3/5] Testing PyTorch Model Inference Engine (ResNet50)...")
    engine = DRInferenceEngine()
    for test_g in range(5):
        sample_img = create_synthetic_fundus(grade=test_g)
        s_byte_arr = io.BytesIO()
        sample_img.save(s_byte_arr, format='JPEG')
        g_tensor, g_preprocessed = preprocess_image(s_byte_arr.getvalue())
        print(f"\n--- Testing Grade {test_g} Image ---")
        g_pred = engine.predict(g_tensor, raw_rgb_image=g_preprocessed)
        assert 0 <= g_pred['predicted_class_id'] <= 4, "Class ID out of bounds [0-4]."
        assert len(g_pred['all_class_probabilities']) == 5, "Must return 5 class probabilities."
    
    prediction = engine.predict(tensor)
    print("\n -> PASSED! PyTorch model forward pass & structured JSON verification succeeded.")

    # 4. Test Grad-CAM XAI Generator (Requirement 3: Base64 string output)
    print("\n[4/5] Testing Grad-CAM Heatmap & Overlay Engine...")
    gradcam_b64 = generate_gradcam(
        input_tensor=tensor,
        original_image=preprocessed_np,
        predicted_class=prediction['predicted_class_id'],
        model=engine.model,
        target_layer=engine.model.target_layer
    )
    assert gradcam_b64.startswith("data:image/jpeg;base64,"), "Grad-CAM output must be a base64 JPEG data URI."
    print(" -> PASSED! Grad-CAM base64 heatmap extraction succeeded.")

    # 5. Test RAG Clinical Report Generator (Requirement 4 & Requirement 5 Endpoint Output)
    print("\n[5/5] Testing RAG Clinical Report & Full Predict Pipeline...")
    clinical_report = generate_clinical_report(
        predicted_class_id=prediction['predicted_class_id'],
        confidence_score=prediction['confidence_percentage']
    )
    assert "progression_risk" in clinical_report
    assert "assessment" in clinical_report
    assert "recommendation" in clinical_report

    # Test complete prediction function asynchronously
    res = asyncio.run(_process_prediction(image_base64=None, file=None, patient_id="TEST-001") if False else _process_prediction_test(img_bytes))
    assert "predicted_class_id" in res
    assert "predicted_class_name" in res
    assert "confidence_percentage" in res
    assert "gradcam_base64" in res
    assert "clinical_report" in res

    print("\n==================================================")
    print(" ALL 5 BACKEND VERIFICATION TESTS PASSED SUCCESSFULLY! ")
    print("==================================================")


async def _process_prediction_test(img_bytes):
    return await _process_prediction(image_base64=None, file=None, patient_id="TEST-001") if False else await _process_prediction_from_bytes(img_bytes)

async def _process_prediction_from_bytes(img_bytes):
    from fastapi import UploadFile
    fake_file = UploadFile(filename="test.jpg", file=io.BytesIO(img_bytes))
    return await _process_prediction(file=fake_file, patient_id="TEST-001")

if __name__ == "__main__":
    run_backend_tests()
