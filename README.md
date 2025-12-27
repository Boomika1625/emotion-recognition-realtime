# Real-Time Emotion Recognition using Deep Learning

This project implements a real-time facial emotion recognition system using
Convolutional Neural Networks (CNN) and OpenCV. The system captures live video
from a webcam, detects human faces, and predicts the emotional state in real time.

---

## Features
- Real-time emotion detection using webcam
- CNN-based facial emotion classification
- Face detection using OpenCV
- Lightweight and CPU-friendly
- Easy to run in VS Code

---

## Emotions Detected
- Angry
- Happy
- Sad
- Surprise
- Fear
- Disgust
- Neutral

---

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-learn

---

## Project Structure
emotion-recognition-realtime/
├── dataset/
│ └── train/
├── models/
├── src/
│ ├── train_model.py
│ └── realtime_emotion.py
├── requirements.txt
└── README.md

yaml
Copy code

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/emotion-recognition-realtime.git
cd emotion-recognition-realtime
2. Create Virtual Environment
bash
Copy code
python -m venv venv
venv\Scripts\activate
3. Install Dependencies
bash
Copy code
python -m pip install -r requirements.txt
Dataset
Place training images inside dataset/train/

Each emotion must have its own folder

Example:

bash
Copy code
dataset/train/happy/
dataset/train/sad/
Note: Dataset and trained models are not included in the repository to keep it lightweight.

How to Run
Train the Model
bash
Copy code
python train.py
Run Real-Time Emotion Recognition
bash
Copy code
python camera.py
Press Q to exit the webcam window.

Applications
Human–Computer Interaction

Emotion-aware systems

AI-based user experience enhancement

Author
Boomika Subramani
Aspiring AI & Machine Learning Engineer