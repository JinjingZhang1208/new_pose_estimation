import os
import cv2
import time
import shutil
import mediapipe as mp
import numpy as np
from datetime import datetime, timedelta
from ultralytics import YOLO
import csv
from sound import speak
from mediapipe_utils import extract_keypoints
from yolo_utils import load_yolo_model, perform_yolo_inference
from svm_model import train_svm_model, load_model, is_waving_with_svm, save_model

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Paths to image folders
wave_folder = 'wave_frames'
not_wave_folder = 'not_wave_images'

# Train the SVM model (or load a pre-trained model)
model_path = 'svm_model.pkl'
scaler_path = 'scaler.pkl'

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    clf, scaler = train_svm_model(wave_folder, not_wave_folder, extract_keypoints)
    save_model(clf, scaler, model_path, scaler_path)
else:
    clf, scaler = load_model(model_path, scaler_path)

# Directory and file paths
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
model = load_yolo_model()

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils

processed_images = set()
last_movement_time = datetime.now()
movement_detected = False
last_wave_detection_time = datetime.min

def draw_text_with_background(image, text, position, font, scale, text_color, bg_color, thickness):
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    cv2.rectangle(image, (x, y - text_h - baseline), (x + text_w, y + baseline), bg_color, -1)
    cv2.putText(image, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)

def monitor_images(screenshot_dir):
    global last_movement_time, movement_detected, last_wave_detection_time
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
                    detections, yolo_results_names = perform_yolo_inference(model, frame)
                    
                    for box in detections:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())
                        label = yolo_results_names[cls]

                        cropped_frame = frame[y1:y2, x1:x2]

                        # Recolor image to RGB
                        image_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                        image_rgb.flags.writeable = False
                        
                        # Make Mediapipe detection
                        results = pose.process(image_rgb)  
                    
                    image_rgb.flags.writeable = True
                    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    
                    try:
                        landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
                        if landmarks:
                            if is_waving_with_svm(landmarks, clf, scaler):
                                current_time = datetime.now()
                                if current_time - last_wave_detection_time > timedelta(minutes=1):
                                    end_time = time.time()
                                    draw_text_with_background(image, 'Wave Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 128, 255), 2)
                                    print("Wave Detected!")
                                    speak("Hello, my friend!")
                                    latency = end_time - start_time
                                    last_wave_detection_time = current_time  # Ensure the sound only appears once within one minute
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
            cv2.destroyAllWindows()
