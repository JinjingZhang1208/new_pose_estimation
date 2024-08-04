# Avatar Wave Detection in VR Chat

## Goal

The primary objective of this project is to detect avatars' waving in VR Chat as a conversation starter cue if users do not talk.

## Overview

This project involves:
- **Annotating custom collected datasets** using Roboflow.
- **Utilizing YOLOv8** to detect objects as bounding boxes.
- **Using MediaPipe** to analyze the motions/poses within the detected boxes and extract keypoints.
- **Training an SVM model** to classify waving gestures based on the extracted keypoints.

## Dataset

The dataset for training the YOLO model is hosted on Roboflow and includes:

- **Roboflow Link**: [Roboflow Project](https://app.roboflow.com)
- **API Key**: `"eW43mQoE4D7QBLaccpat"`
  
The dataset details:

- **Image Count**: 270 images
  - **Training Set**: 70% (190 images)
  - **Validation Set**: 20% (54 images)
  - **Test Set**: 10% (26 images)
- **Performance**: The model achieves a precision rate of 91.3%.

The dataset for training the SVM model is uploaded to Google Drive: 
- **Download Link**：[Google Drive Link](https://drive.google.com/drive/folders/1m3Aj9Ez3TlIpfplZsnzc0y3-b6G8__zB?usp=drive_link)

The dataset details:
- **Image Count**: 195 images
- **Video Count**: 16 videos
- **Feature Extracted**: 5480
- **Performance**: The model achieves an accurancy rate of 65%.
  
## Methodology

### YOLOv8 for Object Detection
- **Purpose**: To detect objects within the video frames and draw bounding boxes around them.
- **Dataset**: Annotated using Roboflow.

### MediaPipe for Keypoint Extraction
- **Purpose**: To analyze the motions/poses within the detected bounding boxes and extract keypoints.
- **Process**: Extracts keypoints from the detected bounding boxes to be used as features for the SVM classifier.

### SVM for Gesture Classification
- **Purpose**: To classify waving gestures based on the extracted keypoints.
- **Keypoints Extraction**: Using MediaPipe to extract keypoints from images.
- **Dataset**: Collected and labeled images for wave and non-wave gestures.
- **Preprocessing**: Normalizing and flattening the keypoints.
- **Training**: Training an SVM classifier with the extracted keypoints and corresponding labels.

### Integration
- **Real-Time Detection**: The SVM model classifies the gestures in real-time.
- **Alert System**: If a waving gesture is detected, the system announces it but only once within a minute to avoid repetitive alerts.

## Testing
- **Test Videos Link**: [Test Videos](https://drive.google.com/drive/folders/1Yzg_5M9AbSPTwzvRbJxnrLFyrRXpfhAf?usp=sharing)
- Tested with 15 pre-recorded videos in VR Chat's different worlds with different avatars waving.
