# Avatar Wave Detection in VR Chat

## Goal

The primary objective of this project is to detect avatars' waving in VR Chat as a conversation starter cue if users do not talk.

## Overview

This project involves:

- **Utilizing YOLOv8** to detect objects as bounding boxes.
- **Using MediaPipe** to analyze the motions/poses within the detected boxes.
- **Annotating custom collected datasets** using Roboflow.

## Dataset

The dataset for this project is hosted on Roboflow and includes:

- **Roboflow Link**: [Roboflow Project](https://app.roboflow.com)
- **API Key**: `"eW43mQoE4D7QBLaccpat"`
  
The dataset details:

- **Project**: `vr_pose_avatars`
- **Version**: `3`
- **Download Command**:
  ```python
  rf = Roboflow(api_key="eW43mQoE4D7QBLaccpat")
  project = rf.workspace("annazhang1208").project("vr_pose_avatars")
  version = project.version(3)
  dataset = version.download("yolov5")

- **Image Count**: 270 images
  - **Training Set**: 70% (190 images)
  - **Validation Set**: 20% (54 images)
  - **Test Set**: 10% (26 images)
- **Performance**: The model achieves a precision rate of 91.3%.