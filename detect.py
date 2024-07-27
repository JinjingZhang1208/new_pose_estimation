import os
import cv2
import time
import shutil
import mediapipe as mp
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from gtts import gTTS
from playsound import playsound
import pygame
import csv
import math
from sound import speak

screenshot_dir = 'captured_images'
csv_file = 'latency_log.csv'

def write_latency_to_csv(latency, csv_file):
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().isoformat(), latency])


def clear_images_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

# Clear captured_images directory at the start
clear_images_directory(screenshot_dir)

# Load YOLO model
model_path = 'C:/Users/andyma/Documents/vrchat_pose_detection/train_datasets/yolov8n.pt'
model = YOLO(model_path)

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

processed_images = set()
last_movement_time = datetime.now()
movement_detected = False

def calculate_angle(a, b, c):
    """Calculate the angle between three points (a, b, c)"""
    # Vector AB
    ab = [b.x - a.x, b.y - a.y]
    # Vector BC
    bc = [c.x - b.x, c.y - b.y]
    
    # Dot product
    dot_product = ab[0] * bc[0] + ab[1] * bc[1]
    
    # Magnitudes
    magnitude_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
    magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)
    
    # Angle in radians
    angle_rad = math.acos(dot_product / (magnitude_ab * magnitude_bc))
    # Convert to degrees
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

def is_waving(landmarks):
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    
    # Calculate angles
    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    # Define thresholds for waving
    LEFT_WAVE_ANGLE_RANGE = (30, 90)  
    RIGHT_WAVE_ANGLE_RANGE = (30, 90) 
    
    # Check if either arm is in the waving angle range
    left_wave = LEFT_WAVE_ANGLE_RANGE[0] <= left_arm_angle <= LEFT_WAVE_ANGLE_RANGE[1]
    right_wave = RIGHT_WAVE_ANGLE_RANGE[0] <= right_arm_angle <= RIGHT_WAVE_ANGLE_RANGE[1]
    
    return left_wave or right_wave

def draw_text_with_background(image, text, position, font, scale, text_color, bg_color, thickness):
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    cv2.rectangle(image, (x, y - text_h - baseline), (x + text_w, y + baseline), bg_color, -1)
    cv2.putText(image, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)

def monitor_images(screenshot_dir):
    global last_movement_time, movement_detected
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        try:
            while True:
                if not os.path.exists(screenshot_dir):
                    clear_images_directory(screenshot_dir)
                
                for image_name in os.listdir(screenshot_dir):
                    image_path = os.path.join(screenshot_dir, image_name)
                    
                    if image_path in processed_images:
                        continue
                    
                    frame = cv2.imread(image_path)
                    
                    if frame is None:
                        continue

                    processed_images.add(image_path)

                    start_time = time.time()

                    # Perform YOLO inference
                    yolo_results = model(frame)[0]
                    detections = yolo_results.boxes  # Accessing boxes from YOLOv8 results
                    
                    for box in detections:
                        # Ensure the coordinates are converted to integers
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())
                        label = yolo_results.names[cls]

                        cropped_frame = frame[y1:y2, x1:x2]

                        # Recolor image to RGB
                        image_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                        image_rgb.flags.writeable = False
                        
                        # Make Mediapipe detection
                        results = pose.process(image_rgb)  
                    
                    # Recolor back to BGR
                    image_rgb.flags.writeable = True
                    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    
                    # Extract landmarks
                    try:
                        landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
                        if landmarks:
                            if is_waving(landmarks):
                                end_time = time.time()
                                draw_text_with_background(image, 'Wave Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 128, 255), 2)
                                print("Wave Detected!")
                                speak("Wave Detected!")
                                latency = end_time - start_time
                                last_movement_time = datetime.now()
                                movement_detected = True
                                write_latency_to_csv(latency, csv_file)
        
                        if results.pose_landmarks:
                            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                      mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                                      mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

                    except Exception as e:
                        print(f"Error processing image: {e}")

                    key = cv2.waitKey(10)
                    if key == ord('q') or key == 27:
                        raise KeyboardInterrupt        
                time.sleep(1)
        
        except KeyboardInterrupt:
            print("Exiting program.")
        
        finally:
            # Clean up
            cv2.destroyAllWindows()
            clear_images_directory(screenshot_dir)
