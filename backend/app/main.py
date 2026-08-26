"""
FastAPI Main Application Server for Universal Retinal Screening Framework.
Provides RESTful APIs for:
- DR Severity Prediction & Class Probability Breakdown
- Grad-CAM Spatial Lesion Heatmap Overlay Generation
- RAG-based ICDR Clinical Decision Support Report Generation
- Synthetic Fundus Dataset Samples for Live Clinical Demo
"""

import io
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging

from app.config import DR_CLASSES, DR_DESCRIPTIONS, DR_SEVERITY_COLORS
from app.preprocessing import preprocess_image
from app.model import DRInferenceEngine
from app.gradcam import generate_gradcam, generate_gradcam_overlay
from app.rag_engine import generate_clinical_report, RAGClinicalReportGenerator, ICDR_KNOWLEDGE_BASE
from app.synthetic_data import create_synthetic_fundus, image_to_base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RetinalScreeningAPI")

app = FastAPI(
    title="Universal Retinal Screening API",
    description="Deep Learning DR Detection with EfficientNet-B4, Grad-CAM XAI, and RAG Clinical Report Generator",
    version="1.0.0"
)

# Enable CORS for local cross-origin frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize deep learning engine and RAG pipeline
try:
    engine = DRInferenceEngine()
    rag_generator = RAGClinicalReportGenerator()
    logger.info("Retinal Screening backend modules successfully initialized.")
except Exception as e:
    logger.error(f"Failed to initialize deep learning components: {e}")
    raise e


class PredictBase64Request(BaseModel):
    image_base64: str
    patient_id: Optional[str] = "PATIENT-9921"


@app.get("/api/health")
@app.get("/health")
def health_check():
    """System health check endpoint."""
    return {
        "status": "healthy",
        "device": str(engine.device),
        "backbone": engine.model.backbone_name,
        "num_classes": engine.model.num_classes,
        "version": "1.0.0"
    }


@app.get("/api/guidelines")
@app.get("/guidelines")
def get_icdr_guidelines():
    """Returns International Clinical Diabetic Retinopathy (ICDR) guidelines."""
    return {
        "guidelines": ICDR_KNOWLEDGE_BASE,
        "class_descriptions": DR_DESCRIPTIONS,
        "severity_colors": DR_SEVERITY_COLORS
    }


@app.get("/api/samples")
@app.get("/samples")
def get_sample_fundus_scans():
    """
    Generates synthetic fundus scan samples for all 5 DR grades for instant clinical testing.
    """
    samples = []
    for grade in range(5):
        img = create_synthetic_fundus(grade=grade)
        samples.append({
            "grade": grade,
            "title": f"Grade {grade}: {DR_CLASSES[grade]}",
            "description": DR_DESCRIPTIONS[grade],
            "image_base64": image_to_base64(img),
            "color": DR_SEVERITY_COLORS[grade]
        })
    return {"samples": samples}


async def _process_prediction(
    file: Optional[UploadFile] = None,
    image_base64: Optional[str] = None,
    patient_id: Optional[str] = "PATIENT-1001"
):
    image_bytes = None

    if file is not None:
        try:
            image_bytes = await file.read()
            logger.info(f"[API Endpoint] Received uploaded file: {file.filename}, {len(image_bytes)} bytes")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file upload: {str(e)}")
    elif image_base64:
        try:
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]
            image_bytes = base64.b64decode(image_base64)
            logger.info(f"[API Endpoint] Received Base64 image payload, {len(image_bytes)} decoded bytes")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid Base64 image payload: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Either image file or image_base64 string must be provided.")

    if not image_bytes or len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Corrupt or empty image data payload.")

    try:
        # 1. Simple Preprocessing
        tensor, preprocessed_np = preprocess_image(image_bytes)

        # 2. Simple Model Inference (prints Input Tensor Mean, Raw Probabilities, Predicted Class Index)
        pred_result = engine.predict(tensor, raw_rgb_image=preprocessed_np)
        pred_grade = pred_result["predicted_class_id"]
        class_name = pred_result["predicted_class_name"]
        confidence_pct = float(pred_result["confidence_percentage"])
        risk_level = pred_result["progression_risk"]
        recommendation_text = pred_result["recommendation"]

        # 3. Grad-CAM XAI spatial lesion map extraction
        gradcam_b64 = generate_gradcam(
            input_tensor=tensor,
            original_image=preprocessed_np,
            predicted_class=pred_grade,
            model=engine.model,
            target_layer=engine.model.target_layer
        )

        overlay_rgb, heatmap_rgb, spatial_summary = generate_gradcam_overlay(
            model=engine.model,
            target_layer=engine.model.target_layer,
            input_tensor=tensor,
            original_rgb=preprocessed_np,
            target_class=pred_grade
        )

        # 4. RAG Clinical Decision Report generation
        clinical_report = generate_clinical_report(
            predicted_class_id=pred_grade,
            confidence_score=confidence_pct,
            probabilities=pred_result["probabilities"],
            spatial_summary=spatial_summary
        )

        # Base64 output images
        preprocessed_b64 = image_to_base64(Image.fromarray(preprocessed_np))
        heatmap_b64 = image_to_base64(Image.fromarray(heatmap_rgb))

        # 5. Requirement 5 JSON Response + UI dashboard compatibility
        return {
            "predicted_class_id": pred_grade,
            "predicted_class_name": class_name,
            "confidence_percentage": confidence_pct,
            "progression_risk": risk_level,
            "recommendation": recommendation_text,
            # Additional UI dashboard compatibility fields:
            "confidence_score": pred_result["confidence"],
            "all_class_probabilities": pred_result["probabilities"],
            "clinical_recommendation": recommendation_text,
            "gradcam_base64": gradcam_b64,
            "clinical_report": clinical_report,
            "patient_id": patient_id,
            "prediction": pred_result,
            "spatial_summary": spatial_summary,
            "images": {
                "preprocessed": preprocessed_b64,
                "gradcam_overlay": gradcam_b64,
                "heatmap_only": heatmap_b64
            }
        }

    except Exception as e:
        logger.error(f"Error executing DR screening prediction pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal screening engine error: {str(e)}")


@app.post("/predict")
async def predict_endpoint(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    patient_id: Optional[str] = Form("PATIENT-1001")
):
    """Retinal Screening Analysis Endpoint (/predict)."""
    return await _process_prediction(file=file, image_base64=image_base64, patient_id=patient_id)


@app.post("/api/predict")
async def api_predict_endpoint(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    patient_id: Optional[str] = Form("PATIENT-1001")
):
    """Retinal Screening Analysis Endpoint (/api/predict)."""
    return await _process_prediction(file=file, image_base64=image_base64, patient_id=patient_id)

