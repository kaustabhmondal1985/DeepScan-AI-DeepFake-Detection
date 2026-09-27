# DeepScan — AI Deepfake Video Detection & Forensic Analysis Platform

<p align="center">

**AI-assisted deepfake video detection, analysis and forensic evidence visualization**

</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-RetinaFace-yellow)
![React](https://img.shields.io/badge/React-Frontend-61DAFB)
![License](https://img.shields.io/badge/Status-Active%20Development-blueviolet)

</p>

---

## 📌 Overview

**DeepScan** is an AI-assisted deepfake video detection and forensic analysis platform designed to analyze digital video content and identify signs of potential manipulation.

The system combines computer vision, deep learning, video processing and an interactive web interface to provide users with a structured analysis of uploaded videos.

Instead of returning only a binary `Real` or `Fake` decision, DeepScan is designed to provide additional evidence such as:

- Video metadata
- Detected faces
- Analyzed frames
- Frame-level predictions
- Real/Fake probabilities
- Confidence information
- Model configuration
- Analysis limitations
- Timestamps associated with analyzed frames
- Exportable analysis reports

The current implementation primarily focuses on **visual deepfake detection using RetinaFace and a fine-tuned Xception-based classifier**.

Future versions are planned to extend the system toward temporal, audio and audio-visual analysis.

---

# 🎯 Problem Statement

The increasing availability of generative AI and face manipulation technologies has made it increasingly difficult to distinguish authentic video content from manipulated media.

Deepfake videos can be generated using techniques such as:

- Face swapping
- Face reenactment
- Facial expression manipulation
- Synthetic face generation
- Lip synchronization
- AI-generated talking-head videos

These manipulated videos can potentially be used for:

- Misinformation
- Identity impersonation
- Fraud
- Social engineering
- Reputation damage
- Fabricated evidence
- Media manipulation

Traditional video inspection is often insufficient for detecting subtle manipulation artifacts.

DeepScan aims to provide an AI-assisted system that can automatically analyze video content and identify patterns that may be associated with manipulated media.

---

# 💡 Project Objective

The primary objective of DeepScan is to develop an AI-assisted platform capable of:

1. Accepting video uploads through a web interface.
2. Extracting video metadata.
3. Sampling representative frames.
4. Detecting faces within those frames.
5. Extracting usable face crops.
6. Performing deep-learning-based visual analysis.
7. Generating frame-level predictions.
8. Aggregating frame-level predictions into a video-level assessment.
9. Presenting real/fake probabilities.
10. Providing confidence information.
11. Showing analyzed frames and timestamps.
12. Providing information about the model used.
13. Showing limitations associated with the analysis.
14. Generating an exportable analysis report.

The long-term objective is to extend the system into a multimodal deepfake analysis framework using:

- Visual analysis
- Temporal analysis
- Audio analysis
- Audio-visual consistency
- Explainability
- Evidence localization

---

# 🧠 Core Idea

The fundamental idea behind DeepScan is:

> **Do not rely only on a single frame or a single prediction. Analyze multiple frames, identify faces, perform deep-learning inference, and aggregate the evidence to produce a video-level assessment.**

The current pipeline can be represented as:

```text
                        Uploaded Video
                              │
                              ▼
                    Video Metadata Extraction
                              │
                              ▼
                       Frame Sampling
                              │
                              ▼
                         RetinaFace
                       Face Detection
                              │
                              ▼
                       Face Crop Extraction
                              │
                              ▼
                    Fine-tuned Xception
                              │
                              ▼
                    Frame-level Prediction
                              │
                              ▼
                  Probability Aggregation
                              │
                              ▼
                    Video-level Assessment
                              │
                              ▼
                 ┌────────────────────────┐
                 │ Analysis Result        │
                 │                        │
                 │ Assessment             │
                 │ Confidence             │
                 │ Real Probability       │
                 │ Fake Probability       │
                 │ Frame Evidence         │
                 │ Video Metadata         │
                 └────────────────────────┘
```

---

# 🏗️ System Architecture

The current DeepScan architecture consists of four major layers:

```text
┌────────────────────────────────────────────────────────────┐
│                      FRONTEND                              │
│                                                            │
│ React / Next.js / TypeScript                               │
│                                                            │
│ • Video Upload                                             │
│ • Analysis Progress                                        │
│ • Results Dashboard                                        │
│ • Frame Timeline                                           │
│ • Model Information                                        │
│ • Report Export                                            │
└───────────────────────────┬────────────────────────────────┘
                            │
                            │ HTTP / REST API
                            ▼
┌────────────────────────────────────────────────────────────┐
│                       BACKEND                              │
│                                                            │
│ FastAPI + Python                                           │
│                                                            │
│ • Upload Handling                                          │
│ • Analysis Management                                      │
│ • Status Tracking                                          │
│ • Result Generation                                        │
│ • Video Processing                                         │
└───────────────────────────┬────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│                     ML PIPELINE                            │
│                                                            │
│ Video Processing                                           │
│        ↓                                                   │
│ Frame Extraction                                           │
│        ↓                                                   │
│ RetinaFace                                                 │
│        ↓                                                   │
│ Face Cropping                                               │
│        ↓                                                   │
│ Xception                                                   │
│        ↓                                                   │
│ Classification                                             │
│        ↓                                                   │
│ Probability Aggregation                                    │
└───────────────────────────┬────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│                    ANALYSIS OUTPUT                         │
│                                                            │
│ • Assessment                                               │
│ • Confidence                                               │
│ • Real/Fake Probabilities                                  │
│ • Frame Predictions                                        │
│ • Face Information                                         │
│ • Video Metadata                                           │
│ • Model Metadata                                           │
│ • Limitations                                              │
│ • Report                                                   │
└────────────────────────────────────────────────────────────┘
```

---

# 🔬 Current Machine Learning Pipeline

The current production-oriented development pipeline uses:

```text
Video
  ↓
Frame Sampling
  ↓
RetinaFace
  ↓
Face Detection
  ↓
Face Crop
  ↓
Xception
  ↓
MLP Classification Head
  ↓
Frame Probability
  ↓
Mean Probability Aggregation
  ↓
Video Assessment
```

---

# 👤 Face Detection — RetinaFace

DeepScan currently uses **RetinaFace** for face detection.

RetinaFace identifies faces present in video frames and provides bounding-box information and detection confidence.

The general process is:

```text
Video Frame
     │
     ▼
RetinaFace
     │
     ├───────────────┐
     ▼               ▼
Face Bounding Box   Detection Confidence
     │
     ▼
Face Crop
```

The system uses detected face regions rather than passing the entire video frame directly into the deepfake classifier.

This allows the classifier to focus on the facial region where manipulation artifacts are expected to be more relevant.

---

# 🖼️ Face Crop Processing

After face detection, DeepScan extracts the detected face region.

Conceptually:

```text
Original Frame
┌───────────────────────────────────────┐
│                                       │
│        ┌───────────────────┐          │
│        │                   │          │
│        │       FACE        │          │
│        │                   │          │
│        └───────────────────┘          │
│                                       │
└───────────────────────────────────────┘
                  │
                  ▼
             Face Crop
                  │
                  ▼
             Resize 299×299
                  │
                  ▼
             Xception Model
```

The current preprocessing pipeline uses a standardized input resolution of:

```text
299 × 299
```

---

# 🧠 Xception-based Visual Classifier

DeepScan currently uses an **ImageNet-pretrained Xception backbone**.

The Xception network extracts a high-dimensional representation of the input face.

The current feature representation has:

```text
2048 dimensions
```

The architecture is conceptually:

```text
Face Image
   │
   ▼
299 × 299
   │
   ▼
Xception Backbone
   │
   ▼
2048-dimensional Feature Vector
   │
   ▼
Dropout
   │
   ▼
Linear Layer
2048 → 256
   │
   ▼
ReLU
   │
   ▼
Dropout
   │
   ▼
Linear Layer
256 → 2
   │
   ├───────────────┐
   ▼               ▼
 Real            Fake
```

The classifier outputs two class probabilities:

```text
Real Probability
Fake Probability
```

---

# 🎞️ Multi-frame Analysis

A video is not treated as a single image.

Instead, DeepScan analyzes multiple representative frames.

For example:

```text
Video
 │
 ├── Frame 1 ──► Face ──► Xception ──► Prediction
 │
 ├── Frame 2 ──► Face ──► Xception ──► Prediction
 │
 ├── Frame 3 ──► Face ──► Xception ──► Prediction
 │
 ├── Frame 4 ──► Face ──► Xception ──► Prediction
 │
 ├── ...
 │
 └── Frame N ──► Face ──► Xception ──► Prediction
```

The individual predictions are then aggregated.

---

# 📊 Probability Aggregation

The current model uses mean probability aggregation.

Suppose the model produces:

```text
Frame 1 → Fake = 0.72
Frame 2 → Fake = 0.65
Frame 3 → Fake = 0.81
Frame 4 → Fake = 0.70
```

The video-level probability can be calculated as:

```text
Average Fake Probability

= (0.72 + 0.65 + 0.81 + 0.70) / 4

= 0.72
```

This aggregated probability is then used to generate the final assessment.

---

# 📋 Assessment Categories

DeepScan does not treat the model output as absolute proof.

The interface uses assessment categories such as:

### Likely Authentic

The model output is more consistent with the authentic class under the current model and analysis conditions.

### Potentially Manipulated

The model output contains evidence that is more consistent with manipulation.

### Inconclusive

The system does not have sufficient usable evidence to provide a meaningful assessment.

These labels describe model output and uncertainty.

They should not be interpreted as definitive proof that a video is authentic or manipulated.

---

# 📈 Analysis Output

A typical analysis result can contain:

```json
{
  "analysis_id": "DS-XXXXXXXX",
  "assessment": "Potentially Manipulated",
  "confidence": 0.72,
  "probabilities": {
    "real": 0.28,
    "fake": 0.72
  },
  "frames_analyzed": 8,
  "tracks_analyzed": 1
}
```

Additional information can include:

```text
Video metadata
Frame predictions
Detected faces
Model configuration
Checkpoint information
Processing limitations
```

---

# 📹 Video Metadata

DeepScan extracts metadata such as:

- Duration
- FPS
- Width
- Height
- Resolution
- Total frame count
- Video codec

Example:

```json
{
  "duration_seconds": 13.0,
  "fps": 30.0,
  "width": 1280,
  "height": 720,
  "resolution": "1280x720",
  "total_frames": 390,
  "codec": "h264"
}
```

---

# 🧪 Dataset

The research pipeline uses a subset of the **FaceForensics++ (FF++)** dataset.

The dataset contains authentic and manipulated videos.

For the current experiment, the selected dataset subset contains:

```text
Total videos: 180
```

The current split is:

| Dataset Split | Videos |
|---------------|-------:|
| Training | 108 |
| Validation | 36 |
| Testing | 36 |
| **Total** | **180** |

The split was designed around source-group separation to reduce leakage between training, validation and testing.

---

# 📊 Dataset Class Distribution

The current selected dataset contains:

```text
Real Videos : 30
Fake Videos : 150
Total       : 180
```

The split distribution is approximately:

```text
Training
├── Real: 18
└── Fake: 90

Validation
├── Real: 6
└── Fake: 30

Testing
├── Real: 6
└── Fake: 30
```

Because the dataset is imbalanced, balanced evaluation metrics are particularly important.

---

# 📐 Evaluation Metrics

DeepScan experiments use metrics including:

### Accuracy

Measures the proportion of correctly classified samples.

```text
Accuracy =
Correct Predictions / Total Predictions
```

### Precision

Measures how many predicted fake samples were actually fake.

```text
Precision =
True Positives / (True Positives + False Positives)
```

### Recall

Measures how many actual fake samples were detected.

```text
Recall =
True Positives / (True Positives + False Negatives)
```

### F1 Score

Combines precision and recall.

```text
F1 =
2 × Precision × Recall
----------------------
Precision + Recall
```

### Balanced Accuracy

Balanced accuracy is particularly useful when the dataset has unequal class distributions.

It considers the recall of each class.

```text
Balanced Accuracy =
(Real Recall + Fake Recall) / 2
```

For deepfake detection experiments, balanced accuracy is important because raw accuracy can be misleading when the number of real and fake videos is highly unequal.

---

# 🧪 Current Experimental Status

The project is currently under active model experimentation.

The latest fine-tuned Xception checkpoint used during development was:

```text
Checkpoint:
xception_finetuned_best.pth

Validation Balanced Accuracy:
59.38%

Checkpoint Epoch:
5
```

This should be considered an experimental result rather than a production performance claim.

The model is still being improved and evaluated.

---

# 🔬 Research Direction

The project is being developed toward a multimodal deepfake detection system.

The research question being explored is:

> **Can multimodal analysis improve the robustness and interpretability of deepfake video detection compared with visual-only analysis?**

The planned progression is:

```text
Baseline
   │
   ▼
Xception Visual Detection
   │
   ▼
Xception + Temporal Modeling
   │
   ▼
Xception + Audio Analysis
   │
   ▼
Audio-Visual Consistency
   │
   ▼
Multimodal Fusion
   │
   ▼
Explainable Forensic Analysis
```

---

# ⏱️ Planned Temporal Analysis

A future version can introduce temporal modeling using LSTM or similar sequence architectures.

The idea is to analyze how facial features change across time rather than treating each frame independently.

Conceptually:

```text
Frame 1 ──► Xception ──► Feature 1
Frame 2 ──► Xception ──► Feature 2
Frame 3 ──► Xception ──► Feature 3
Frame 4 ──► Xception ──► Feature 4
                         │
                         ▼
                    Feature Sequence
                         │
                         ▼
                        LSTM
                         │
                         ▼
                    Temporal Output
```

This could help investigate temporal inconsistencies that may not be visible in individual frames.

---

# 🔊 Planned Audio Analysis

Future versions are planned to analyze the audio track of a video.

The planned pipeline is:

```text
Video
  │
  ▼
FFmpeg
  │
  ▼
Audio Extraction
  │
  ▼
Audio Preprocessing
  │
  ▼
Mel Spectrogram / MFCC
  │
  ▼
Audio Model
  │
  ▼
Audio Features
```

Potential audio features include:

- Mel spectrograms
- MFCCs
- Spectral characteristics
- Temporal audio patterns

Libraries planned for this component include:

- FFmpeg
- Librosa
- SoundFile
- PyTorch

Audio analysis is a planned extension and is not currently the primary active detection path.

---

# 👄 Planned Audio-Visual Consistency

A stronger multimodal approach is to investigate whether the audio and visible facial movements are temporally consistent.

For example:

```text
                Video
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
      Audio               Video
        │                   │
        ▼                   ▼
 Audio Features       Face/Mouth Motion
        │                   │
        └─────────┬─────────┘
                  ▼
       Audio-Visual Consistency
                  │
                  ▼
             Consistency Score
```

Potential signals include:

- Speech timing
- Mouth movement
- Lip synchronization
- Facial motion
- Temporal alignment

This component is intended as a research extension.

---

# 🔀 Planned Multimodal Fusion

The long-term architecture is:

```text
                    VIDEO
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
     VISUAL                       AUDIO
        │                           │
        ▼                           ▼
   Face Detection              Audio Extraction
        │                           │
        ▼                           ▼
     Xception                  Audio Model
        │                           │
        ▼                           ▼
 Visual Embedding             Audio Embedding
        │                           │
        └──────────────┬────────────┘
                       │
                       ▼
             Audio-Visual Analysis
                       │
                       ▼
                Fusion Network
                       │
                       ▼
              Final Assessment
```

---

# 🧠 Explainability

One of the future goals of DeepScan is to provide more than a classification score.

Potential explainability techniques include:

- Grad-CAM
- Attention visualization
- Important frame identification
- Face-region highlighting
- Temporal evidence visualization

For example:

```text
Video
  │
  ▼
Important Frames
  │
  ├── 00:02.13
  ├── 00:04.67
  ├── 00:07.20
  └── 00:09.83
        │
        ▼
  Important Facial Regions
        │
        ▼
  Explainability Visualization
```

The objective is to help users understand what evidence contributed to the model output.

---

# 🗂️ Project Structure

The repository is organized into frontend, backend, machine-learning and training components.

```text
DeepScan-AI-DeepFake-Detection/
│
├── backend/
│   │
│   ├── app/
│   │   │
│   │   ├── api/
│   │   │   ├── dependencies.py
│   │   │   └── routes/
│   │   │       ├── analysis.py
│   │   │       ├── health.py
│   │   │       ├── history.py
│   │   │       └── __init__.py
│   │   │
│   │   ├── ml/
│   │   │   │
│   │   │   ├── audio/
│   │   │   │   ├── audio_model.py
│   │   │   │   ├── audio_processor.py
│   │   │   │   ├── audio_analyzer.py
│   │   │   │   └── feature_extractor.py
│   │   │   │
│   │   │   ├── explainability/
│   │   │   │   └── gradcam.py
│   │   │   │
│   │   │   ├── multimodal/
│   │   │   │   ├── av_consistency.py
│   │   │   │   └── fusion.py
│   │   │   │
│   │   │   ├── preprocessing/
│   │   │   │   ├── face_detector.py
│   │   │   │   ├── frame_extractor.py
│   │   │   │   ├── video_processor.py
│   │   │   │   └── face_tracker.py
│   │   │   │
│   │   │   ├── temporal/
│   │   │   │   ├── lstm_model.py
│   │   │   │   ├── temporal_analyzer.py
│   │   │   │   └── deepfake_classifier.py
│   │   │   │
│   │   │   └── visual/
│   │   │       ├── visual_analyzer.py
│   │   │       ├── xception_model.py
│   │   │       └── track_embedding_extractor.py
│   │   │
│   │   ├── models/
│   │   │   ├── analysis.py
│   │   │   └── schemas.py
│   │   │
│   │   ├── services/
│   │   │   ├── analysis_service.py
│   │   │   ├── audio_service.py
│   │   │   ├── report_service.py
│   │   │   └── video_service.py
│   │   │
│   │   └── utils/
│   │       ├── file_utils.py
│   │       ├── logger.py
│   │       └── video_utils.py
│   │
│   ├── models/
│   ├── outputs/
│   ├── uploads/
│   ├── tests/
│   ├── training/
│   ├── .env.example
│   └── README.md
│
├── frontend/
│   └── ...
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# ⚙️ Technology Stack

## Frontend

| Technology | Purpose |
|------------|---------|
| React / Next.js | Web application |
| TypeScript | Type-safe development |
| Tailwind CSS | UI styling |
| Framer Motion | Animations |
| Recharts | Data visualization |
| HTML5 Video | Video playback |

---

## Backend

| Technology | Purpose |
|------------|---------|
| Python | Core backend / ML |
| FastAPI | REST API |
| Uvicorn | ASGI server |
| Python-dotenv | Environment configuration |

---

## Machine Learning

| Technology | Purpose |
|------------|---------|
| PyTorch | Deep learning |
| TorchVision | Vision utilities |
| timm | Pretrained vision architectures |
| Xception | Visual deepfake classification |
| TensorFlow | RetinaFace compatibility |
| RetinaFace | Face detection |

---

## Computer Vision

| Technology | Purpose |
|------------|---------|
| OpenCV | Video and image processing |
| Pillow | Image processing |
| NumPy | Numerical computation |

---

## Audio

| Technology | Purpose |
|------------|---------|
| FFmpeg | Audio/video processing |
| Librosa | Audio feature extraction |
| SoundFile | Audio file handling |

---

## Data & Evaluation

| Technology | Purpose |
|------------|---------|
| Pandas | Dataset processing |
| SciPy | Scientific computing |
| Scikit-learn | Evaluation metrics |

---

## Testing

| Technology | Purpose |
|------------|---------|
| Pytest | Backend testing |
| HTTPX | API testing |

---

# 🛠️ Installation

## Prerequisites

Before installing DeepScan, make sure the following are installed:

- Python 3.12
- Node.js
- npm
- Git
- FFmpeg

For GPU-based model training/inference, a compatible NVIDIA CUDA environment can be used.

---

# 1. Clone the Repository

```bash
git clone https://github.com/kaustabhmondal1985/DeepScan-AI-DeepFake-Detection.git
```

Navigate into the repository:

```bash
cd DeepScan-AI-DeepFake-Detection
```

---

# 2. Backend Virtual Environment

Create a virtual environment:

```powershell
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

---

# 3. Install Python Dependencies

Install the project dependencies:

```powershell
pip install -r requirements.txt
```

---

# 4. FFmpeg

DeepScan uses FFmpeg for video and audio processing.

After installing FFmpeg, verify it:

```powershell
ffmpeg -version
```

The command should display the installed FFmpeg version.

If `ffmpeg` is not recognized, add the FFmpeg `bin` directory to the system PATH.

---

# 5. Environment Configuration

Copy the example environment file:

```powershell
Copy-Item backend\.env.example backend\.env
```

Open:

```text
backend/.env
```

and configure the required variables.

Do not commit `.env` to GitHub.

---

# 🚀 Running the Backend

Navigate to:

```powershell
cd backend
```

Start FastAPI:

```powershell
python -m uvicorn app.main:app --reload
```

The backend will be available at:

```text
http://127.0.0.1:8000
```

---

# 📚 API Documentation

FastAPI automatically generates API documentation.

### Swagger UI

```text
http://127.0.0.1:8000/docs
```

### ReDoc

```text
http://127.0.0.1:8000/redoc
```

---

# ❤️ Health Check

The health endpoint is:

```text
GET /api/health
```

Example:

```text
http://127.0.0.1:8000/api/health
```

Expected response:

```json
{
  "status": "healthy",
  "service": "DeepScan Backend",
  "version": "1.0.0"
}
```

---

# 🔍 Video Analysis API

The primary endpoint is:

```text
POST /api/analysis/analyze
```

The endpoint accepts a video upload and starts the analysis process.

The high-level processing sequence is:

```text
Upload
  ↓
Save Video
  ↓
Extract Metadata
  ↓
Extract Representative Frames
  ↓
Detect Faces
  ↓
Create Face Crops
  ↓
Run Xception
  ↓
Aggregate Predictions
  ↓
Generate Result
```

---

# 📊 Analysis Status

The status endpoint is:

```text
GET /api/analysis/{analysis_id}/status
```

Example:

```text
GET /api/analysis/DS-XXXXXXXX/status
```

This can be used by the frontend to monitor the analysis process.

---

# 📄 Analysis Result

The final result endpoint is:

```text
GET /api/analysis/{analysis_id}/result
```

The result may include:

```text
Analysis ID
Assessment
Confidence
Real Probability
Fake Probability
Frames Analyzed
Faces Detected
Video Metadata
Frame Predictions
Model Information
Limitations
```

---

# 🌐 Frontend Setup

Navigate to the frontend directory:

```powershell
cd frontend
```

Install dependencies:

```powershell
npm install
```

Start the development server:

```powershell
npm run dev
```

The frontend will generally be available at:

```text
http://localhost:3000
```

---

# 🔗 Frontend ↔ Backend

The frontend communicates with the FastAPI backend through REST APIs.

The general flow is:

```text
User
 │
 ▼
Frontend
 │
 │ Upload Video
 ▼
FastAPI Backend
 │
 ▼
ML Pipeline
 │
 ▼
Analysis Result
 │
 ▼
FastAPI
 │
 ▼
Frontend
 │
 ▼
Results Dashboard
```

---

# 🖥️ Frontend Features

The current frontend includes functionality for:

### Video Upload

Users can select a video for analysis.

### Analysis Progress

The interface can display the current processing state.

### Assessment

The final assessment is displayed to the user.

### Confidence

The model confidence is shown alongside the result.

### Real/Fake Probability

The UI presents the model probabilities.

### Video Metadata

Information such as:

- Duration
- FPS
- Resolution
- Total frames
- Codec

can be displayed.

### Frame Timeline

Analyzed frames and their predictions can be displayed along a timeline.

### Face Information

The system can display information related to detected faces.

### Model Information

The frontend can display:

- Model name
- Architecture
- Input size
- Embedding dimension
- Frames analyzed
- Aggregation method
- Checkpoint information

### Report Export

Users can export the analysis into a printable report and save it as PDF through the browser.

---

# 📄 Report Generation

DeepScan provides a report-generation workflow.

The report can contain:

```text
Analysis Summary
       ↓
Assessment
       ↓
Confidence
       ↓
Real/Fake Probabilities
       ↓
Video Metadata
       ↓
Frame Predictions
       ↓
Model Configuration
       ↓
Checkpoint Information
       ↓
Limitations
```

The generated report can be printed or saved as PDF using the browser's print dialog.

---

# 🧪 Training Pipeline

The training workflow is separate from the inference API.

The general research workflow is:

```text
Raw Dataset
     │
     ▼
Dataset Split
     │
     ▼
Frame Sampling
     │
     ▼
RetinaFace
     │
     ▼
Face Crop Extraction
     │
     ▼
Xception Feature / Model Training
     │
     ▼
Validation
     │
     ▼
Checkpoint Selection
     │
     ▼
Test Evaluation
```

---

# 🧠 Training Considerations

The training experiments consider:

- Class imbalance
- Source-group separation
- Validation balanced accuracy
- Video-level evaluation
- Model generalization
- Fine-tuning
- Data augmentation

The model should not be evaluated only using raw accuracy because the dataset contains substantially more fake samples than real samples.

---

# 📦 Model Checkpoints

Large trained model files are intentionally excluded from Git.

Examples:

```text
models/*.pth
models/*.pt
models/*.ckpt
```

This prevents large binary artifacts from unnecessarily increasing repository size.

When a trained checkpoint is required, it should be obtained through the project's documented model-storage or training workflow.

---

# 📁 Generated Data

The following directories contain generated/local data and are excluded from Git:

```text
uploads/
outputs/
training/processed/
```

### `uploads/`

Contains uploaded video files.

### `outputs/`

Contains generated analysis outputs.

### `training/processed/`

Contains processed training data and extracted face crops.

These files should not be committed to the public repository.

---

# 🔐 Security

Never commit sensitive information.

Do not commit:

```text
.env
API keys
Passwords
Tokens
Private datasets
Private videos
Database credentials
Cloud credentials
Authentication secrets
```

Use:

```text
.env.example
```

for documenting required environment variables without exposing actual secrets.

---

# 🧹 Git Ignore

The repository intentionally ignores development artifacts such as:

```text
.env
venv/
uploads/
outputs/
model weights
processed datasets
node_modules/
.next/
```

This keeps the GitHub repository lightweight and safe to share.

---

# 🔬 Research Evaluation Strategy

DeepScan's research evaluation should go beyond a single train/test result.

The planned evaluation strategy includes:

## 1. In-domain Evaluation

Evaluate the model using videos originating from the same dataset distribution while maintaining source-group separation.

---

## 2. Cross-dataset Evaluation

Evaluate the model on a dataset that was not used during training.

This can provide information about generalization.

---

## 3. Unseen-generator Evaluation

Evaluate on manipulation methods or generative systems not observed during training.

This is important because a model may perform well on known manipulation patterns but struggle with new generation methods.

---

## 4. Compression Robustness

Test videos at different compression levels.

```text
Original Video
      │
      ├── High Quality
      ├── Medium Quality
      └── Highly Compressed
```

---

## 5. Resolution Robustness

Evaluate different resolutions:

```text
1080p
720p
480p
360p
```

---

## 6. Frame-rate Robustness

Evaluate videos with different frame rates.

---

# ⚠️ Current Limitations

The current system has several limitations.

### Dataset Limitations

The current model is trained using a selected subset of FaceForensics++.

Therefore, its performance may not generalize to every real-world video source.

---

### Domain Shift

Real-world videos may differ significantly from training data.

Examples include:

- Social media compression
- Mobile-camera recordings
- Different lighting
- Different camera sensors
- Different resolutions
- Different codecs
- Different face poses

---

### Unseen Manipulations

A model trained on known manipulation techniques may not reliably detect completely new manipulation methods.

---

### Face Detection

If a face cannot be detected reliably, the visual classifier cannot perform normal face-based analysis.

Potential causes include:

- Very small faces
- Heavy occlusion
- Extreme angles
- Motion blur
- Poor lighting
- Low video quality

---

### Model Confidence

Model confidence is not equivalent to certainty.

A high confidence score does not guarantee that a video is actually manipulated.

---

### False Positives and False Negatives

The system may produce incorrect predictions.

Therefore, results should be interpreted as AI-assisted evidence rather than definitive conclusions.

---

# ⚖️ Responsible Use

DeepScan is intended for:

- Research
- Educational purposes
- AI experimentation
- Media analysis
- Digital-forensics research
- AI-assisted screening

The system should not be treated as an absolute authority for determining whether a video is genuine or manipulated.

A model prediction alone should not be used as the sole basis for:

- Legal decisions
- Criminal accusations
- Employment decisions
- Financial decisions
- Public accusations
- Identity verification
- High-stakes investigations

Human review and additional evidence should be considered for high-stakes use cases.

---

# 🚧 Development Roadmap

## Phase 1 — Visual Baseline

**Current**

```text
Video
 ↓
Frame Sampling
 ↓
RetinaFace
 ↓
Xception
 ↓
Frame Aggregation
 ↓
Assessment
```

---

## Phase 2 — Improved Visual Model

Planned improvements:

- More training data
- Better class balancing
- Improved augmentation
- More robust video-level aggregation
- Better checkpoint selection
- Cross-dataset testing

---

## Phase 3 — Temporal Modeling

Planned:

```text
Xception
   ↓
Feature Sequence
   ↓
LSTM
   ↓
Temporal Representation
   ↓
Classifier
```

---

## Phase 4 — Audio Analysis

Planned:

```text
Video
 ↓
Audio Extraction
 ↓
Mel / MFCC
 ↓
Audio Model
 ↓
Audio Prediction
```

---

## Phase 5 — Audio-Visual Consistency

Planned:

```text
Audio
  +
Facial Motion
  ↓
Temporal Alignment
  ↓
Consistency Analysis
```

---

## Phase 6 — Multimodal Fusion

Planned:

```text
Visual Features
       +
Temporal Features
       +
Audio Features
       +
Audio-Visual Consistency
       ↓
Multimodal Fusion
       ↓
Final Assessment
```

---

## Phase 7 — Explainability

Planned:

- Grad-CAM
- Important frame detection
- Facial-region highlighting
- Attention visualization
- Temporal evidence
- Explainable analysis reports

---

## Phase 8 — Advanced Forensic Report

Future reports may contain:

```text
Video Information
        ↓
Face Detection Evidence
        ↓
Frame-level Evidence
        ↓
Temporal Evidence
        ↓
Audio Evidence
        ↓
Audio-Visual Evidence
        ↓
Model Explanation
        ↓
Overall Assessment
```

---

# 🔄 Recommended Future Architecture

The long-term DeepScan architecture is envisioned as:

```text
                         ┌─────────────────┐
                         │     VIDEO       │
                         └────────┬────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 │                                 │
                 ▼                                 ▼
        ┌─────────────────┐              ┌─────────────────┐
        │ VISUAL PIPELINE │              │ AUDIO PIPELINE  │
        └────────┬────────┘              └────────┬────────┘
                 │                                │
                 ▼                                ▼
            Frame Sampling                  Audio Extraction
                 │                                │
                 ▼                                ▼
             RetinaFace                    Audio Features
                 │                                │
                 ▼                                ▼
             Face Crops                      Audio Model
                 │                                │
                 ▼                                ▼
             Xception                      Audio Embedding
                 │                                │
                 ▼                                │
         Visual Embedding                         │
                 │                                │
                 └──────────────┬─────────────────┘
                                │
                                ▼
                    Audio-Visual Consistency
                                │
                                ▼
                       Multimodal Fusion
                                │
                                ▼
                       Final Assessment
                                │
                 ┌──────────────┼──────────────┐
                 ▼              ▼              ▼
             Confidence      Evidence     Explanation
                 │              │              │
                 └──────────────┼──────────────┘
                                ▼
                         Forensic Report
```

---

# 👥 Collaboration

DeepScan is designed to support collaborative development.

Recommended Git workflow:

```text
main
 │
 ├── feature/frontend
 │
 ├── feature/audio-analysis
 │
 ├── feature/temporal-model
 │
 ├── feature/multimodal-fusion
 │
 └── feature/explainability
```

Developers should create separate branches for major features.

---

# 🌿 Creating a Feature Branch

```bash
git checkout -b feature-name
```

Example:

```bash
git checkout -b feature/audio-analysis
```

---

# 💾 Commit Changes

```bash
git add .
```

Then:

```bash
git commit -m "Add audio feature extraction"
```

---

# 🚀 Push Branch

```bash
git push -u origin feature/audio-analysis
```

Then create a Pull Request on GitHub.

---

# 🔀 Pull Latest Changes

Before starting new work:

```bash
git checkout main
git pull origin main
```

Then create your feature branch:

```bash
git checkout -b feature-name
```

---

# 🧪 Testing

Backend tests are located in:

```text
backend/tests/
```

Run:

```powershell
pytest
```

For API-specific testing, the FastAPI application can also be tested using HTTPX/TestClient.

---

# 🩺 Debugging

If the backend does not start:

```powershell
python -m uvicorn app.main:app --reload
```

Check:

```text
http://127.0.0.1:8000/docs
```

If FFmpeg is not detected:

```powershell
ffmpeg -version
```

If Python dependencies are missing:

```powershell
pip install -r requirements.txt
```

If the virtual environment is not active:

```powershell
.\venv\Scripts\Activate.ps1
```

---

# 📌 Common Development Commands

## Start Backend

```powershell
cd backend
python -m uvicorn app.main:app --reload
```

## Start Frontend

```powershell
cd frontend
npm run dev
```

## Run Tests

```powershell
pytest
```

## Check Git Status

```powershell
git status
```

## Pull Latest Changes

```powershell
git pull origin main
```

---

# 📦 Repository Contents

The repository contains:

```text
Frontend
Backend
Machine Learning Pipeline
Training Code
API Routes
Preprocessing
Model Integration
Testing
Configuration Examples
Documentation
```

Large and sensitive development artifacts are intentionally excluded.

---

# 📈 Future Improvements

Potential future improvements include:

- Larger training datasets
- More diverse real-world videos
- Cross-dataset benchmarking
- Improved face tracking
- Temporal modeling
- Audio deepfake detection
- Audio-visual synchronization analysis
- Multimodal fusion
- Explainability
- Evidence localization
- Better report generation
- GPU-accelerated inference
- Background job processing
- Cloud storage
- Authentication
- Analysis history
- Scalable deployment
- Automated model evaluation
- Experiment tracking

---

# ☁️ Future Production Architecture

For a production-scale version, the system could evolve into:

```text
                    User
                     │
                     ▼
              Web Application
                     │
                     ▼
                API Gateway
                     │
             ┌───────┴────────┐
             │                │
             ▼                ▼
         FastAPI          Authentication
             │
             ▼
        Job Queue
             │
             ▼
      ┌───────────────┐
      │ ML Workers    │
      │               │
      │ RetinaFace    │
      │ Xception      │
      │ Temporal      │
      │ Audio         │
      │ Fusion        │
      └───────┬───────┘
              │
              ▼
         Result Store
              │
              ▼
        Analysis Report
              │
              ▼
             User
```

Potential production technologies could include:

- Redis
- Celery
- PostgreSQL
- Object storage
- Docker
- Cloud GPU workers
- Authentication
- Monitoring
- Logging

These are future architectural directions and are not necessarily part of the current local implementation.

---

# 🏆 Project Highlights

DeepScan focuses on combining AI detection with interpretable analysis.

Key characteristics include:

### 🔍 Face-focused Detection

Uses RetinaFace to identify facial regions before classification.

### 🧠 Deep Learning

Uses an Xception-based visual model for deepfake classification.

### 🎞️ Multi-frame Analysis

Analyzes multiple video frames instead of relying on a single frame.

### 📊 Probability-based Results

Provides real/fake probability information rather than only a binary label.

### ⏱️ Frame Evidence

Associates predictions with analyzed frames and timestamps.

### 📋 Structured Reports

Provides a structured analysis report containing model and video information.

### 🔬 Research-oriented Architecture

Designed to support future temporal, audio and multimodal research.

---

# 📚 Research Focus

The broader research direction of DeepScan is:

> **Multimodal and explainable deepfake video detection**

The project aims to investigate whether combining multiple evidence sources can improve:

- Detection robustness
- Generalization
- Interpretability
- Evidence localization
- Resistance to unseen manipulation techniques

---

# 📝 Project Status

```text
Status: Active Development
```

### Currently implemented

- Video upload
- Video metadata extraction
- Frame extraction
- RetinaFace face detection
- Face crop extraction
- Xception-based visual classification
- Frame-level prediction
- Video-level probability aggregation
- Analysis status tracking
- Result API
- Frontend result visualization
- Report export
- Basic testing infrastructure

### Under development / planned

- Temporal modeling
- Audio analysis
- Audio-visual consistency
- Multimodal fusion
- Grad-CAM explainability
- Advanced evidence localization
- Cross-dataset evaluation
- Production-scale deployment

---

# ⚠️ Disclaimer

DeepScan is an AI-assisted research and screening system.

Its predictions may be incorrect due to:

- Dataset limitations
- Model limitations
- Distribution shift
- Video quality
- Compression
- Face detection errors
- Unseen manipulation techniques
- Adversarial or synthetic content

The system should therefore be treated as a supporting analytical tool rather than definitive proof of authenticity or manipulation.

For high-stakes decisions, DeepScan output should be combined with human review and additional independent evidence.

---

# 👨‍💻 Contributors

## Kaustabh Mondal

Project development, machine-learning pipeline, backend development, frontend integration, experimentation and system architecture.

Additional contributors are welcome to collaborate through feature branches and pull requests.

---

# ⭐ DeepScan

### Detect. Analyze. Explain.

**An AI-assisted platform for deepfake video analysis and forensic evidence exploration.**

---
