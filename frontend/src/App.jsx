import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import SampleSelector from './components/SampleSelector';
import ImageUploader from './components/ImageUploader';
import PredictionCard from './components/PredictionCard';
import GradCamViewer from './components/GradCamViewer';
import ClinicalReport from './components/ClinicalReport';
import GuidelinesModal from './components/GuidelinesModal';
import { Loader2, AlertCircle, Eye, RefreshCw } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000/api';

export default function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [patientId, setPatientId] = useState('PATIENT-1001');
  const [samples, setSamples] = useState([]);
  const [guidelines, setGuidelines] = useState([]);
  const [selectedGrade, setSelectedGrade] = useState(null);

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const [predictionData, setPredictionData] = useState(null);
  const [isGuidelinesOpen, setIsGuidelinesOpen] = useState(false);

  // Check Backend Health & Fetch Samples on Initial Load
  useEffect(() => {
    fetchHealthAndSamples();
  }, []);

  const fetchHealthAndSamples = async () => {
    try {
      const healthRes = await fetch(`${API_BASE_URL}/health`);
      if (healthRes.ok) {
        setIsConnected(true);
      } else {
        setIsConnected(false);
      }

      const samplesRes = await fetch(`${API_BASE_URL}/samples`);
      if (samplesRes.ok) {
        const data = await samplesRes.json();
        setSamples(data.samples || []);
      }

      const guidelinesRes = await fetch(`${API_BASE_URL}/guidelines`);
      if (guidelinesRes.ok) {
        const gData = await guidelinesRes.json();
        setGuidelines(gData.guidelines || []);
      }
    } catch (err) {
      console.warn("Backend API offline or unreachable:", err);
      setIsConnected(false);
    }
  };

  // Run DR Screening Prediction for Uploaded File
  const handleFileUpload = async (file) => {
    setIsLoading(true);
    setError(null);
    setSelectedGrade(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('patient_id', patientId);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Failed to process fundus scan.");
      }

      const result = await response.json();
      setPredictionData(result);
    } catch (err) {
      setError(err.message || "An unexpected error occurred during model inference.");
    } finally {
      setIsLoading(false);
    }
  };

  // Run DR Screening Prediction for Pre-Loaded Sample
  const handleSelectSample = async (sample) => {
    setIsLoading(true);
    setError(null);
    setSelectedGrade(sample.grade);

    const formData = new FormData();
    formData.append('image_base64', sample.image_base64);
    formData.append('patient_id', `${patientId}-G${sample.grade}`);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Failed to analyze sample fundus image.");
      }

      const result = await response.json();
      setPredictionData(result);
    } catch (err) {
      setError(err.message || "An error occurred analyzing the sample.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '20px 24px' }}>
      <Header
        isConnected={isConnected}
        patientId={patientId}
        setPatientId={setPatientId}
        onOpenGuidelines={() => setIsGuidelinesOpen(true)}
      />

      {/* Main Layout Grid */}
      <div className="dashboard-grid">
        {/* Left Control Column */}
        <div>
          <SampleSelector
            samples={samples}
            selectedGrade={selectedGrade}
            onSelectSample={handleSelectSample}
            isLoading={isLoading}
          />

          <ImageUploader
            onImageUpload={handleFileUpload}
            isLoading={isLoading}
          />

          {!isConnected && (
            <div style={{
              background: 'rgba(245, 158, 11, 0.1)',
              border: '1px solid rgba(245, 158, 11, 0.3)',
              borderRadius: '12px',
              padding: '16px',
              color: '#FBBF24',
              fontSize: '0.85rem'
            }}>
              <div style={{ fontWeight: '700', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <AlertCircle size={16} /> Backend Server Offline
              </div>
              <p>Please ensure the FastAPI backend is running on <code>http://localhost:8000</code>.</p>
              <button
                className="btn-secondary"
                onClick={fetchHealthAndSamples}
                style={{ marginTop: '10px', width: '100%', justifyContent: 'center', fontSize: '0.8rem' }}
              >
                <RefreshCw size={14} /> Retry Connection
              </button>
            </div>
          )}
        </div>

        {/* Right Output Column */}
        <div>
          {isLoading && (
            <div className="glass-card" style={{ padding: '60px 20px', textAlign: 'center' }}>
              <Loader2 size={42} color="var(--accent-cyan)" className="pulse" style={{ animation: 'spin 1s linear infinite', margin: '0 auto 16px' }} />
              <h3 style={{ fontSize: '1.2rem', fontWeight: '700' }}>Analyzing Retinal Scan...</h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '6px' }}>
                Executing CLAHE preprocessing, PyTorch multi-class inference, Grad-CAM XAI feature extraction, and RAG guideline retrieval.
              </p>
            </div>
          )}

          {error && (
            <div className="glass-card" style={{ padding: '24px', borderColor: 'rgba(239, 68, 68, 0.4)', background: 'rgba(239, 68, 68, 0.1)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: '#F87171' }}>
                <AlertCircle size={24} />
                <h3 style={{ fontSize: '1.1rem', fontWeight: '700' }}>Screening Analysis Error</h3>
              </div>
              <p style={{ fontSize: '0.88rem', color: 'var(--text-primary)', marginTop: '8px' }}>{error}</p>
            </div>
          )}

          {!isLoading && !error && predictionData && (
            <>
              <PredictionCard
                prediction={predictionData.prediction}
                spatialSummary={predictionData.spatial_summary}
              />

              <GradCamViewer
                images={predictionData.images}
                spatialSummary={predictionData.spatial_summary}
              />

              <ClinicalReport
                report={predictionData.clinical_report}
                patientId={predictionData.patient_id}
              />
            </>
          )}

          {!isLoading && !error && !predictionData && (
            <div className="glass-card" style={{ padding: '80px 20px', textAlign: 'center' }}>
              <Eye size={48} color="var(--accent-cyan)" style={{ opacity: 0.4, margin: '0 auto 16px' }} />
              <h3 style={{ fontSize: '1.25rem', fontWeight: '700', color: 'var(--text-primary)' }}>
                No Fundus Scan Selected
              </h3>
              <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', marginTop: '6px', maxWidth: '450px', margin: '6px auto 0' }}>
                Select one of the pre-loaded clinical test samples on the left or upload a patient fundus photograph to launch full screening.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* ICDR Guidelines Modal Reference */}
      <GuidelinesModal
        isOpen={isGuidelinesOpen}
        onClose={() => setIsGuidelinesOpen(false)}
        guidelines={guidelines}
      />
    </div>
  );
}
