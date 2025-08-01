# SignScribe - Sign Language Detection Using Machine Learning

SignScribe is a real-time sign language detection system that uses computer vision and machine learning to interpret sign language gestures through a webcam feed.

## Features

- Real-time sign language detection using webcam
- Flask-based web interface
- Machine learning model trained on hand landmark data
- Supports detection of letters: A, B, C, H

## Technologies Used

- Python
- Flask
- OpenCV
- MediaPipe
- Keras/TensorFlow
- HTML/CSS

## Project Structure

```
SignScribe/
├── app.py              # Main Flask application
├── function.py         # Helper functions for MediaPipe and data processing
├── trainmodel.py       # Model training script
├── data.py             # Data collection script
├── collectdata.py      # Image collection script
├── static/             # Static HTML, CSS, and image files
├── templates/          # Flask templates
├── MP_Data/            # Training data
├── Image/              # Collected images
└── model.json          # Model architecture
└── new_model2.h5       # Model weights
```

## How It Works

1. The system uses MediaPipe to detect hand landmarks in real-time
2. These landmarks are processed and fed into a trained LSTM neural network
3. The model predicts the sign language gesture being performed
4. Results are displayed on a web interface via Flask

## Setup

1. Install required dependencies:
   ```
   pip install flask opencv-python mediapipe tensorflow keras numpy scikit-learn
   ```

2. Run the application:
   ```
   python app.py
   ```

3. Access the webcam interface at `http://127.0.0.1:5000/`

## Web Interface

The web interface consists of:
- Home page (`static/index.html`)
- Webcam detection page (`templates/webcam.html`)
- About page (`static/about.html`)
- Contact page (`static/contact.html`)

## Model Training

The model is trained using LSTM layers with the following architecture:
- 3 LSTM layers (64, 128, 64 units)
- Dropout layer for regularization
- Dense layers (64, 32 units)
- Output layer with softmax activation

Training data consists of sequences of hand landmarks

