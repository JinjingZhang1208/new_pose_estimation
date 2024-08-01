import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mediapipe_utils import extract_keypoints, preprocess_angles, extract_angles
import cv2

def process_folder_with_angles(folder_path, label, frame_step=5):
    """Process all images in a folder, extract angles, and assign labels."""
    angles_data = []
    labels = []
    frames = []
    for i, filename in enumerate(sorted(os.listdir(folder_path))):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            if i % frame_step != 0:
                continue
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            keypoints = extract_keypoints(image)
            if keypoints:
                angles = extract_angles(keypoints)
                if angles:
                    frames.append(angles)
    
    if len(frames) > 0:
        angles_data.append(preprocess_angles(frames))
        labels.append(label)
    
    return angles_data, labels

def train_svm(wave_folder, not_wave_folder):
    wave_data, wave_labels = process_folder_with_angles(wave_folder, 'wave')
    not_wave_data, not_wave_labels = process_folder_with_angles(not_wave_folder, 'not_wave')

    keypoints_data = wave_data + not_wave_data
    labels = wave_labels + not_wave_labels

    X = np.array(keypoints_data)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")

    return clf, scaler

def is_waving_with_svm(clf, scaler, keypoints):
    angles = extract_angles(keypoints)
    if angles:
        features = preprocess_angles([angles])
        features = scaler.transform([features])
        prediction = clf.predict(features)
        return prediction[0] == 'wave'
    return False
