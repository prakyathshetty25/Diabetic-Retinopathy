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
from app.preprocessing import prepare_tensor_from_image, preprocess_fundus_image
from app.model import DRInferenceEngine
from app.gradcam import generate_gradcam_overlay
from app.rag_engine import RAGClinicalReportGenerator, ICDR_KNOWLEDGE_BASE
from app.synthetic_data import create_synthetic_fundus, image_to_base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RetinalScreeningAPI")

app = FastAPI(
    title="Universal Retinal Screening API",
    description="Deep Learning DR Detection, Grad-CAM XAI, and RAG Clinical Report Generator",
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
def get_icdr_guidelines():
    """Returns International Clinical Diabetic Retinopathy (ICDR) guidelines."""
    return {
        "guidelines": ICDR_KNOWLEDGE_BASE,
        "class_descriptions": DR_DESCRIPTIONS,
        "severity_colors": DR_SEVERITY_COLORS
    }


@app.get("/api/samples")
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


@app.post("/api/predict")
async def predict_retinal_scan(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    patient_id: Optional[str] = Form("PATIENT-1001")
):
    """
    Primary Retinal Screening Analysis Endpoint.
    Executes:
    1. Input image validation & corrupt scan protection
    2. Preprocessing & CLAHE contrast enhancement
    3. Multi-Class DR Severity Inference (PyTorch)
    4. Grad-CAM XAI spatial lesion map extraction
    5. RAG Clinical Decision Report generation
    """
    pil_image = None

    if file is not None:
        try:
            contents = await file.read()
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid or corrupt image file: {str(e)}")
    elif image_base64:
        try:
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]
            image_bytes = base64.b64decode(image_base64)
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Either image file or image_base64 string must be provided.")

    if pil_image is None or pil_image.size[0] == 0 or pil_image.size[1] == 0:
        raise HTTPException(status_code=400, detail="Corrupt image: zero dimension detected.")

    try:
        # 1. Preprocessing and PyTorch tensor preparation
        tensor, preprocessed_np = prepare_tensor_from_image(pil_image)

        # 2. PyTorch Model Prediction
        pred_result = engine.predict(tensor)
        pred_grade = pred_result["predicted_class_id"]
        confidence = pred_result["confidence"]
        probabilities = pred_result["probabilities"]

        # 3. Grad-CAM Heatmap Extraction & Overlay
        overlay_rgb, heatmap_rgb, spatial_summary = generate_gradcam_overlay(
            model=engine.model,
            target_layer=engine.model.target_layer,
            input_tensor=tensor,
            original_rgb=preprocessed_np,
            target_class=pred_grade
        )

        # 4. RAG Clinical Report Generation
        clinical_report = rag_generator.generate_report(
            predicted_grade=pred_grade,
            confidence=confidence,
            probabilities=probabilities,
            spatial_summary=spatial_summary
        )

        # Convert output images to Base64 for clean JSON response
        preprocessed_b64 = image_to_base64(Image.fromarray(preprocessed_np))
        gradcam_overlay_b64 = image_to_base64(Image.fromarray(overlay_rgb))
        heatmap_b64 = image_to_base64(Image.fromarray(heatmap_rgb))

        return {
            "patient_id": patient_id,
            "prediction": pred_result,
            "spatial_summary": spatial_summary,
            "clinical_report": clinical_report,
            "images": {
                "preprocessed": preprocessed_b64,
                "gradcam_overlay": gradcam_overlay_b64,
                "heatmap_only": heatmap_b64
            }
        }

    except Exception as e:
        logger.error(f"Error processing screening request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal screening engine error: {str(e)}")
