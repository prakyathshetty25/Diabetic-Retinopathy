"""
Clinical Explanation Generator (RAG Pipeline)
Retrieval-Augmented Generation Module connecting predicted DR severity grade + Grad-CAM spatial lesion metrics
to ICDR (International Clinical Diabetic Retinopathy) guidelines and patient care protocols.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

from app.config import DR_CLASSES, DR_DESCRIPTIONS, DR_SEVERITY_COLORS

logger = logging.getLogger(__name__)

# International Clinical Diabetic Retinopathy (ICDR) Guidelines Knowledge Base
ICDR_KNOWLEDGE_BASE: List[Dict[str, Any]] = [
    {
        "id": "icdr_grade_0",
        "grade": 0,
        "title": "No Diabetic Retinopathy (ICDR Grade 0)",
        "criteria": "No microvascular abnormalities present upon fundus examination.",
        "lesion_indicators": ["No microaneurysms", "No hemorrhages", "No exudates", "Clean vascular arch"],
        "follow_up_recommendation": "Annual dilated eye examination by an eye care specialist.",
        "patient_counseling": "Maintain tight glycemic control (HbA1c < 7.0%), monitor blood pressure and lipid profiles.",
        "urgency_level": "Routine"
    },
    {
        "id": "icdr_grade_1",
        "grade": 1,
        "title": "Mild Non-Proliferative Diabetic Retinopathy (ICDR Grade 1)",
        "criteria": "Presence of microaneurysms ONLY.",
        "lesion_indicators": ["Isolated microaneurysms", "Tiny red dots in capillary beds", "No hard exudates"],
        "follow_up_recommendation": "Follow-up dilated fundus exam within 6 to 12 months.",
        "patient_counseling": "Optimize blood glucose, blood pressure, and renal function. Schedule regular screening.",
        "urgency_level": "Low"
    },
    {
        "id": "icdr_grade_2",
        "grade": 2,
        "title": "Moderate Non-Proliferative Diabetic Retinopathy (ICDR Grade 2)",
        "criteria": "More than microaneurysms, but less than severe NPDR (e.g. hard exudates, cotton wool spots, intraretinal hemorrhages).",
        "lesion_indicators": ["Multiple microaneurysms", "Hard exudates (waxy yellow deposits)", "Cotton wool spots (nerve fiber layer micro-infarcts)", "Retinal hemorrhages in < 4 quadrants"],
        "follow_up_recommendation": "Follow-up dilated fundus exam within 3 to 6 months. Consider OCT imaging to evaluate macular edema.",
        "patient_counseling": "Strict metabolic control required. Prompt evaluation if visual acuity decreases or floaters appear.",
        "urgency_level": "Moderate"
    },
    {
        "id": "icdr_grade_3",
        "grade": 3,
        "title": "Severe Non-Proliferative Diabetic Retinopathy (ICDR Grade 3)",
        "criteria": "Meets any one of the 4-2-1 ICDR criteria: >20 intraretinal hemorrhages in 4 quadrants, prominent venous beading in 2+ quadrants, or intraretinal microvascular abnormalities (IRMA) in 1+ quadrant.",
        "lesion_indicators": ["Widespread intraretinal hemorrhages", "Venous beading / sausage-like vessels", "IRMA (Intraretinal Microvascular Abnormalities)", "Extensive cotton wool spots"],
        "follow_up_recommendation": "Urgent referral to an Retina Specialist / Ophthalmologist within 2 to 4 weeks. High risk of conversion to PDR.",
        "patient_counseling": "High risk of vision loss. Anti-VEGF or Panretinal Photocoagulation (PRP) evaluation required.",
        "urgency_level": "High - Urgent Referral"
    },
    {
        "id": "icdr_grade_4",
        "grade": 4,
        "title": "Proliferative Diabetic Retinopathy (ICDR Grade 4)",
        "criteria": "Neovascularization of the disc (NVD), neovascularization elsewhere (NVE), vitreous hemorrhage, or preretinal hemorrhage.",
        "lesion_indicators": ["Fragile new blood vessel growth (Neovascularization)", "Vitreous hemorrhage", "Preretinal membrane formation", "Fibrovascular proliferation", "Tractional retinal detachment risk"],
        "follow_up_recommendation": "Immediate emergency referral to Retina Specialist within 24-48 hours. Consider urgent Anti-VEGF therapy, PRP laser, or vitrectomy.",
        "patient_counseling": "Vision-threatening medical emergency. Avoid strenuous physical exertion and heavy lifting until evaluated.",
        "urgency_level": "Critical - Emergency Referral"
    }
]


class ClinicalVectorStore:
    """
    Lightweight vector store using TF-IDF and Cosine Similarity for fast, deterministic RAG retrieval.
    """
    def __init__(self, knowledge_base: List[Dict[str, Any]] = ICDR_KNOWLEDGE_BASE):
        self.documents = knowledge_base
        self.vectorizer = TfidfVectorizer(stop_words="english")
        
        # Build text corpus for embedding
        self.corpus = [
            f"{doc['title']} {doc['criteria']} {' '.join(doc['lesion_indicators'])} {doc['follow_up_recommendation']} {doc['patient_counseling']}"
            for doc in self.documents
        ]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

    def retrieve_guidelines(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Retrieves top-k relevant clinical guideline documents given a diagnostic text query.
        """
        query_vec = self.vectorizer.transform([query])
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        top_indices = np.argsort(sim_scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc["similarity_score"] = float(sim_scores[idx])
            results.append(doc)
        return results


class RAGClinicalReportGenerator:
    def __init__(self):
        self.vector_store = ClinicalVectorStore()

    def generate_report(
        self,
        predicted_grade: int,
        confidence: float,
        probabilities: Dict[str, float],
        spatial_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive, structured clinical decision support report.
        """
        class_name = DR_CLASSES.get(predicted_grade, "Unknown")
        
        # Construct RAG query from prediction & spatial lesion indicators
        query = f"Diabetic Retinopathy grade {predicted_grade} {class_name} lesions in {spatial_summary.get('primary_lesion_quadrant', 'retina')} high density {spatial_summary.get('high_lesion_density_pct', 0)}%"
        
        # Retrieve matched guidelines
        retrieved_docs = self.vector_store.retrieve_guidelines(query, top_k=2)
        primary_guideline = retrieved_docs[0] if retrieved_docs else ICDR_KNOWLEDGE_BASE[predicted_grade]

        # Clinical Reasoning Synthesis
        high_pct = spatial_summary.get("high_lesion_density_pct", 0)
        quadrant = spatial_summary.get("primary_lesion_quadrant", "Central Macular Region")
        
        if predicted_grade == 0:
            lesion_synthesis = f"Grad-CAM feature maps confirm normal retinal architecture with minimal focal activation across the {quadrant} ({high_pct}% density)."
        elif predicted_grade == 1:
            lesion_synthesis = f"Grad-CAM highlights localized micro-foci of interest primarily concentrated in the {quadrant} ({high_pct}% spatial coverage), indicative of isolated microaneurysms."
        elif predicted_grade == 2:
            lesion_synthesis = f"Grad-CAM highlights moderate spatial lesion clusters within the {quadrant} ({high_pct}% high-intensity coverage), consistent with microaneurysms and hard exudates."
        elif predicted_grade == 3:
            lesion_synthesis = f"Grad-CAM highlights extensive, multi-quadrant feature activations predominantly localized in the {quadrant} ({high_pct}% high-density area), signaling severe intraretinal hemorrhages and IRMA."
        else:
            lesion_synthesis = f"Grad-CAM demonstrates critical high-intensity feature maps across the {quadrant} ({high_pct}% coverage), revealing visual patterns associated with neovascularization or preretinal hemorrhage."

        report = {
            "summary_header": {
                "diagnosis": class_name,
                "icdr_grade": predicted_grade,
                "confidence_score": f"{confidence * 100:.1f}%",
                "severity_color": DR_SEVERITY_COLORS.get(predicted_grade, "#000000"),
                "urgency_level": primary_guideline["urgency_level"]
            },
            "diagnostic_reasoning": {
                "icdr_criteria_matched": primary_guideline["criteria"],
                "model_confidence_analysis": f"The classifier assigned a {confidence * 100:.1f}% confidence to {class_name}.",
                "xai_gradcam_findings": lesion_synthesis,
                "observed_indicators": primary_guideline["lesion_indicators"]
            },
            "spatial_lesion_analysis": {
                "primary_quadrant": quadrant,
                "high_lesion_density_coverage": f"{high_pct}% of field of view",
                "gradcam_peak_intensity": f"{spatial_summary.get('max_intensity', 0):.2f}"
            },
            "clinical_recommendations": {
                "referral_timeline": primary_guideline["follow_up_recommendation"],
                "patient_counseling_plan": primary_guideline["patient_counseling"]
            },
            "retrieved_guideline_citations": [
                {
                    "title": doc["title"],
                    "criteria": doc["criteria"],
                    "relevance_score": f"{doc.get('similarity_score', 1.0):.2f}"
                } for doc in retrieved_docs
            ]
        }

        return report
