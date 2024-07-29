import os
import cv2
import time
import shutil
import mediapipe as mp
import numpy as np
from datetime import datetime, timedelta
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO
import csv
import math
from sound import speak

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def extract_keypoints(image):
    """Extract keypoints from an image using MediaPipe Pose."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        return [landmark for landmark in results.pose_landmarks.landmark]
    return []

def preprocess_keypoints(keypoints):
    """Normalize and flatten keypoints."""
    keypoints_array = np.array([(kp.x, kp.y, kp.z) for kp in keypoints])
    keypoints_array = (keypoints_array - np.min(keypoints_array, axis=0)) / (np.ptp(keypoints_array, axis=0) + 1e-6)
    return keypoints_array.flatten()

def process_folder(folder_path, label):
    """Process all images in a folder, extract keypoints, and assign labels."""
    keypoints_data = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            keypoints = extract_keypoints(image)
            if keypoints:
                features = preprocess_keypoints(keypoints)
                keypoints_data.append(features)
                labels.append(label)
    return keypoints_data, labels

wave_folder = 'wave_frames'
not_wave_folder = 'not_wave_images'

# Process both folders
wave_data, wave_labels = process_folder(wave_folder, 'wave')
not_wave_data, not_wave_labels = process_folder(not_wave_folder, 'not_wave')

# Combine data and labels
keypoints_data = wave_data + not_wave_data
labels = wave_labels + not_wave_labels

# Convert to numpy arrays
X = np.array(keypoints_data)
y = np.array(labels)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the SVM model
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Evaluate the model
accuracy = clf.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

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
mp_drawing = mp.solutions.drawing_utils

processed_images = set()
last_movement_time = datetime.now()
movement_detected = False
last_wave_detection_time = datetime.min

def is_waving_with_svm(keypoints):
    """Predict if the keypoints correspond to a waving gesture using SVM."""
    features = preprocess_keypoints(keypoints)
    features = scaler.transform([features])
    prediction = clf.predict(features)
    return prediction[0] == 'wave'

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
                    yolo_results = model(frame)[0]
                    detections = yolo_results.boxes  # Accessing boxes from YOLOv8 results
                    
                    for box in detections:
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
                    
                    image_rgb.flags.writeable = True
                    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    
                    try:
                        landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
                        if landmarks:
                            if is_waving_with_svm(landmarks):
                                current_time = datetime.now()
                                if current_time - last_wave_detection_time > timedelta(minutes=1):
                                    end_time = time.time()
                                    draw_text_with_background(image, 'Wave Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 128, 255), 2)
                                    print("Wave Detected!")
                                    speak("Wave Detected!")
                                    latency = end_time - start_time
                                    last_wave_detection_time = current_time  # MAKE SURE THAT THE SOUND ONLY APPEARS ONCE WITHIN ONE MINUTE
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
