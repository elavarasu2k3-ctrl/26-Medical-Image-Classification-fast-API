
import { useState, useEffect } from "react";
import axios from "axios";
import {
  Activity,
  Brain,
  LayoutDashboard,
  Scan,
  Settings,
  Upload,
  FileImage,
  ShieldCheck,
  ArrowUpRight,
  HeartPulse,
  LoaderCircle,
  AlertCircle,
  CheckCircle2,
} from "lucide-react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

function App() {
  const [activePage, setActivePage] = useState("Dashboard");
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("ResUNet");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const menuItems = [
    { name: "Dashboard", icon: LayoutDashboard },
    { name: "Pneumonia Detection", icon: Activity },
    { name: "Brain Tumor Analysis", icon: Brain },
  ];

  useEffect(() => {
    if (activePage === "Brain Tumor Analysis") {
      loadModels();
    }
  }, [activePage]);

  const loadModels = async () => {
    try {
      const response = await axios.get(
        `${API_BASE}/api/v1/brain-tumor/models`
      );

      setModels(response.data["available models"] || []);
      setSelectedModel(response.data["default model"] || "ResUNet");
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        "Unable to load brain tumor models. Check the backend server."
      );
    }
  };

  const changePage = (page) => {
    setActivePage(page);
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError("");
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];

    if (!file) return;

    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError("");
  };

  const runAnalysis = async () => {
    if (!selectedFile) {
      setError("Please select a medical image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      let response;

      if (activePage === "Pneumonia Detection") {
        response = await axios.post(
          `${API_BASE}/api/v1/pneumonia/predict`,
          formData
        );
      } else if (activePage === "Brain Tumor Analysis") {
        formData.append("model", selectedModel);

        response = await axios.post(
          `${API_BASE}/api/v1/brain-tumor/segment`,
          formData
        );
      }

      setResult(response.data);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        err.message ||
        "Analysis failed. Please check the backend server."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <HeartPulse size={25} />
          </div>
          <div>
            <h2>MedVision</h2>
            <span>AI IMAGING PLATFORM</span>
          </div>
        </div>

        <p className="menu-label">WORKSPACE</p>

        <nav>
          {menuItems.map((item) => {
            const Icon = item.icon;

            return (
              <button
                key={item.name}
                className={`nav-item ${
                  activePage === item.name ? "active" : ""
                }`}
                onClick={() => changePage(item.name)}
              >
                <Icon size={19} />
                <span>{item.name}</span>
              </button>
            );
          })}
        </nav>

        <div className="sidebar-bottom">
          <button className="nav-item">
            <Settings size={19} />
            <span>Settings</span>
          </button>

          <div className="system-status">
            <span className="status-dot"></span>
            <div>
              <strong>System Ready</strong>
              <small>AI research platform</small>
            </div>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <span className="breadcrumb">Workspace / </span>
            <strong>{activePage}</strong>
          </div>

          <div className="profile">
            <ShieldCheck size={19} />
            <span>Medical AI Research</span>
            <div className="avatar">K</div>
          </div>
        </header>

        <section className="page-content">
          <div className="welcome-row">
            <div>
              <p className="eyebrow">MEDICAL IMAGE INTELLIGENCE</p>
              <h1>
                {activePage === "Dashboard"
                  ? "Welcome to MedVision"
                  : activePage}
              </h1>
              <p className="subtitle">
                AI-powered medical image analysis and research platform.
              </p>
            </div>

            <div className="date-chip">
              <Scan size={17} />
              AI Analysis Workspace
            </div>
          </div>

          {activePage === "Dashboard" ? (
            <>
              <div className="stats-grid">
                <div className="stat-card">
                  <div className="stat-icon blue">
                    <Activity />
                  </div>
                  <p>AI Classification</p>
                  <h2>DenseNet121</h2>
                  <span>Pneumonia detection model</span>
                </div>

                <div className="stat-card">
                  <div className="stat-icon purple">
                    <Brain />
                  </div>
                  <p>AI Segmentation</p>
                  <h2>4 Models</h2>
                  <span>Brain tumor segmentation</span>
                </div>

                <div className="stat-card">
                  <div className="stat-icon green">
                    <ShieldCheck />
                  </div>
                  <p>Platform Status</p>
                  <h2>Ready</h2>
                  <span>Research environment</span>
                </div>
              </div>

              <div className="section-heading">
                <div>
                  <h2>Analysis Modules</h2>
                  <p>Select a module to start your analysis.</p>
                </div>
              </div>

              <div className="module-grid">
                <div className="module-card">
                  <div className="module-icon blue">
                    <Activity size={25} />
                  </div>
                  <span className="module-tag">CLASSIFICATION</span>
                  <h3>Pneumonia Detection</h3>
                  <p>
                    Analyze chest X-ray images using the DenseNet121 deep
                    learning model.
                  </p>
                  <button
                    onClick={() => changePage("Pneumonia Detection")}
                  >
                    Open Module <ArrowUpRight size={17} />
                  </button>
                </div>

                <div className="module-card">
                  <div className="module-icon purple">
                    <Brain size={25} />
                  </div>
                  <span className="module-tag">SEGMENTATION</span>
                  <h3>Brain Tumor Analysis</h3>
                  <p>
                    Explore MRI segmentation using UNet, ResUNet,
                    AttentionUNet and UNet++.
                  </p>
                  <button
                    onClick={() => changePage("Brain Tumor Analysis")}
                  >
                    Open Module <ArrowUpRight size={17} />
                  </button>
                </div>
              </div>
            </>
          ) : (
            <div className="analysis-panel">
              <div className="panel-heading">
                <div className="module-icon blue">
                  {activePage === "Pneumonia Detection" ? (
                    <Activity />
                  ) : (
                    <Brain />
                  )}
                </div>

                <div>
                  <h2>{activePage}</h2>
                  <p>
                    {activePage === "Pneumonia Detection"
                      ? "Model: DenseNet121"
                      : "Brain MRI segmentation"}
                  </p>
                </div>
              </div>

              {activePage === "Brain Tumor Analysis" && (
                <div className="model-selector">
                  <label htmlFor="model">Select segmentation model</label>
                  <select
                    id="model"
                    value={selectedModel}
                    onChange={(event) =>
                      setSelectedModel(event.target.value)
                    }
                  >
                    {models.map((model) => (
                      <option key={model} value={model}>
                        {model}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              <div className="upload-area">
                <div className="upload-icon">
                  <Upload size={30} />
                </div>

                <h3>Upload Medical Image</h3>
                <p>Select an image from your computer for AI analysis.</p>

                <label className="upload-button">
                  <FileImage size={18} />
                  Choose Image
                  <input
                    type="file"
                    accept="image/png,image/jpeg,image/jpg"
                    onChange={handleFileChange}
                    hidden
                  />
                </label>

                {selectedFile && (
                  <div className="selected-file">
                    <FileImage size={18} />
                    <span>{selectedFile.name}</span>
                  </div>
                )}

                {preview && (
                  <img
                    src={preview}
                    alt="Selected medical scan preview"
                    className="image-preview"
                  />
                )}

                <small>Supported formats: PNG, JPG, JPEG · Maximum 10 MB</small>

                <button
                  className="analyze-button"
                  onClick={runAnalysis}
                  disabled={!selectedFile || loading}
                >
                  {loading ? (
                    <>
                      <LoaderCircle className="spin" size={18} />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Scan size={18} />
                      Run AI Analysis
                    </>
                  )}
                </button>
              </div>

              {error && (
                <div className="error-message">
                  <AlertCircle size={19} />
                  <span>{error}</span>
                </div>
              )}

              {result && (
                

  activePage === "Pneumonia Detection" ? (
    <div className="friendly-result">
      <div className="friendly-result-header">
        <CheckCircle2 size={26} />
        <div>
          <h2>Analysis Completed</h2>
          <p>Your AI analysis result is ready.</p>
        </div>
      </div>

      <div className={`prediction-banner ${
        result.prediction === "NORMAL" ? "normal" : "pneumonia"
      }`}>
        <span>AI MODEL RESULT</span>
        <h1>
          {result.prediction === "NORMAL"
            ? "Normal Pattern Detected"
            : "Pneumonia Pattern Detected"}
        </h1>
        <p>
          {result.prediction === "NORMAL"
            ? "The AI model classified this X-ray as Normal."
            : "The AI model identified a pattern associated with Pneumonia."}
        </p>
      </div>

      <div className="probability-card">
        <h3>Prediction Probabilities</h3>

        <div className="probability-item">
          <div className="probability-label">
            <strong>Normal</strong>
            <span>
              {((result.probabilities?.NORMAL ?? 0) * 100).toFixed(2)}%
            </span>
          </div>
          <div className="probability-track">
            <div
              className="probability-fill normal-fill"
              style={{
                width: `${(result.probabilities?.NORMAL ?? 0) * 100}%`
              }}
            />
          </div>
        </div>

        <div className="probability-item">
          <div className="probability-label">
            <strong>Pneumonia</strong>
            <span>
              {((result.probabilities?.PNEUMONIA ?? 0) * 100).toFixed(2)}%
            </span>
          </div>
          <div className="probability-track">
            <div
              className="probability-fill pneumonia-fill"
              style={{
                width: `${(result.probabilities?.PNEUMONIA ?? 0) * 100}%`
              }}
            />
          </div>
        </div>
      </div>

      <div className="medical-notice">
        <AlertCircle size={20} />
        <p>
          This is an AI research result, not a medical diagnosis.
          Please consult a qualified healthcare professional for
          medical interpretation.
        </p>
      </div>
    </div>
  ) : (
    <div className="result-placeholder">
      <CheckCircle2 size={23} />
      <div>
        <strong>Analysis completed successfully</strong>
        <p>Selected model: {selectedModel}</p>
        <pre className="result-json">
          {JSON.stringify(result, null, 2)}
        </pre>
      </div>
    </div>
  )
)}

              
            </div>
          )}

          <footer>
            <span>MedVision AI · Medical Image Analysis</span>
            <span>Research prototype — Not for clinical diagnosis</span>
          </footer>
        </section>
      </main>
    </div>
  );
}

export default App;
