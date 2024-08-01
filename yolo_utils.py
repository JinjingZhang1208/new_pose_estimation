import cv2
from ultralytics import YOLO

# mkae sure to edit the model path file. The current is mine.
model_path = 'C:/Users/andyma/Documents/vrchat_pose_detection/train_datasets/yolov8n.pt'
model = YOLO(model_path)

def load_yolo_model():
    return model

def perform_yolo_inference(model, frame):
    yolo_results = model(frame)[0]
    detections = yolo_results.boxes  # Accessing boxes from YOLOv8 results
    return detections, yolo_results.names
