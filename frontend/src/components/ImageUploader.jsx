import React, { useRef, useState } from 'react';
import { UploadCloud, Image as ImageIcon, AlertTriangle, CheckCircle2 } from 'lucide-react';

export default function ImageUploader({ onImageUpload, isLoading }) {
  const fileInputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);
  const [selectedFileName, setSelectedFileName] = useState(null);

  const validateAndProcessFile = (file) => {
    setErrorMsg(null);
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setErrorMsg("Invalid file type. Please upload a digital fundus photograph (JPEG, PNG, TIFF).");
      return;
    }

    if (file.size > 20 * 1024 * 1024) { // 20MB limit
      setErrorMsg("File size exceeds 20MB limit.");
      return;
    }

    setSelectedFileName(file.name);
    onImageUpload(file);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndProcessFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="glass-card" style={{ padding: '24px', marginBottom: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <ImageIcon size={20} color="var(--accent-cyan)" />
          <h3 style={{ fontSize: '1.05rem', fontWeight: '700' }}>Fundus Scan Input</h3>
        </div>
        <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>APTOS / EyePACS Standard Compliant</span>
      </div>

      <div
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        style={{
          border: dragActive ? '2px dashed var(--accent-cyan)' : '2px dashed rgba(255, 255, 255, 0.15)',
          background: dragActive ? 'rgba(6, 182, 212, 0.08)' : 'rgba(255, 255, 255, 0.02)',
          borderRadius: '14px',
          padding: '32px 20px',
          textAlign: 'center',
          cursor: isLoading ? 'not-allowed' : 'pointer',
          transition: 'all 0.2s ease'
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={(e) => e.target.files?.[0] && validateAndProcessFile(e.target.files[0])}
          style={{ display: 'none' }}
        />

        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px' }}>
          <div style={{
            background: 'rgba(6, 182, 212, 0.1)',
            padding: '16px',
            borderRadius: '50%',
            color: 'var(--accent-cyan)'
          }}>
            <UploadCloud size={32} />
          </div>

          <div>
            <p style={{ fontSize: '0.95rem', fontWeight: '600', color: 'var(--text-primary)' }}>
              {selectedFileName ? `Loaded: ${selectedFileName}` : "Drag & drop patient fundus scan here"}
            </p>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
              or click to browse local DICOM / JPEG files
            </p>
          </div>
        </div>
      </div>

      {errorMsg && (
        <div style={{
          marginTop: '14px',
          padding: '10px 14px',
          borderRadius: '8px',
          background: 'rgba(239, 68, 68, 0.15)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          color: '#F87171',
          fontSize: '0.82rem',
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}>
          <AlertTriangle size={16} />
          {errorMsg}
        </div>
      )}
    </div>
  );
}
