import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

def predict_gesture(image_path):
    """Predict if an image contains a 'wave' gesture or 'not wave'."""
    image = cv2.imread(image_path)
    keypoints = extract_keypoints(image)
    if keypoints:
        features = preprocess_keypoints(keypoints)
        features = scaler.transform([features])
        prediction = clf.predict(features)
        return prediction[0]
    return None

#Test to see if this model works， implement this in detect.py
if __name__ == "__main__":
    test_image_path = 'test.png'
    gesture = predict_gesture(test_image_path)
    print(f"Predicted gesture: {gesture}")
