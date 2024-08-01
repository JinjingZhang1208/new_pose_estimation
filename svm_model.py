import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import cv2

def preprocess_keypoints(keypoints):
    """Normalize and flatten keypoints."""
    keypoints_array = np.array([(kp.x, kp.y, kp.z) for kp in keypoints])
    keypoints_array = (keypoints_array - np.min(keypoints_array, axis=0)) / (np.ptp(keypoints_array, axis=0) + 1e-6)
    return keypoints_array.flatten()

def process_folder(folder_path, label, extract_keypoints):
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

def train_svm_model(wave_folder, not_wave_folder, extract_keypoints):
    # Process both folders
    wave_data, wave_labels = process_folder(wave_folder, 'wave', extract_keypoints)
    not_wave_data, not_wave_labels = process_folder(not_wave_folder, 'not_wave', extract_keypoints)

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

    return clf, scaler

def save_model(model, scaler, model_path, scaler_path):
    """Save the trained SVM model and scaler to files."""
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

def load_model(model_path, scaler_path):
    """Load the trained SVM model and scaler from files."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def is_waving_with_svm(keypoints, model, scaler):
    """Predict if the keypoints correspond to a waving gesture using SVM."""
    features = preprocess_keypoints(keypoints)
    features = scaler.transform([features])
    prediction = model.predict(features)
    return prediction[0] == 'wave'
