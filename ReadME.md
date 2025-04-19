# Interview Analysis System

A comprehensive system for conducting and analyzing interviews using multiple data points including facial expressions, eye tracking, and speech analysis.

## 🌟 Features

- **Real-time Analysis**
  - Facial Expression Recognition
  - Eye Movement Tracking
  - Speech-to-Text Transcription
  - Voice Analysis
  - Visual Attention Monitoring

- **Data Management**
  - Automatic Session Recording
  - Structured Data Storage
  - Analysis Summary Generation
  - Interview Metrics Tracking

- **API Integration**
  - FastAPI Backend
  - Database Integration
  - RESTful Endpoints
  - User Profile Management

## 🛠️ Technology Stack

- **Core Technologies**
  - Python 3.x
  - OpenCV
  - PyAudio
  - Speech Recognition
  - TensorFlow/PyTorch (for ML models)

- **Backend**
  - FastAPI
  - SQLAlchemy
  - RAG (Retrieval-Augmented Generation)

- **Data Processing**
  - NumPy
  - Pandas
  - JSON Data Storage

## 📋 Prerequisites

```bash
# Install required packages
pip install -r requirements.txt
```

Main dependencies include:
- opencv-python
- pyaudio
- speech_recognition
- numpy
- fastapi
- sqlalchemy
- pyttsx3
- tensorflow/pytorch

## 🚀 Getting Started

1. **Clone the repository**
```bash
git clone [repository-url]
cd [project-directory]
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure the application**
- Update `config.py` with your settings
- Ensure camera and microphone permissions are granted

4. **Run the application**
```bash
python analysis/main.py
```

## 📁 Project Structure

```

## 🎯 Core Components

### 1. Interview Analyzer
- Real-time video capture and processing
- Multi-threaded audio recording
- Synchronized data collection
- Live metric display

### 2. Analysis Modules
- **Facial Analysis**: Emotion detection and tracking
- **Eye Tracking**: Attention and gaze analysis
- **Speech Processing**: Voice-to-text and speech patterns
- **Data Management**: Session recording and metrics storage

### 3. API System
- RESTful endpoints for data access
- User profile management
- Database integration
- Configuration management

## 🔧 Configuration

Key configuration options in `config.py`:
- Database settings
- API endpoints
- Analysis parameters
- Storage locations

## 📊 Data Output

The system generates:
- Real-time metrics
- Session recordings
- Analysis summaries
- JSON data exports

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## ⚠️ Important Notes

- Ensure proper lighting for facial analysis
- Use a good quality microphone for voice analysis
- Check camera permissions before starting
- Regular system calibration recommended

## 🆘 Troubleshooting

Common issues and solutions:
1. Camera access denied: Check system permissions
2. Audio not recording: Verify microphone settings
3. Analysis not running: Check dependency installation
4. Performance issues: Adjust analysis intervals

## 📧 Contact

nikunjgupta2136@gmail.com
