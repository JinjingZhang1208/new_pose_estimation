import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def extract_keypoints(image):
    """Extract keypoints from an image using MediaPipe Pose."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        return [landmark for landmark in results.pose_landmarks.landmark]
    return []

def extract_angles(keypoints):
    """Extract angles from keypoints."""
    angles = []
    if len(keypoints) >= 3: 
        shoulder = (keypoints[5].x, keypoints[5].y)
        elbow = (keypoints[6].x, keypoints[6].y)
        wrist = (keypoints[7].x, keypoints[7].y)
        angle = compute_angle(shoulder, elbow, wrist)
        angles.append(angle)
    return angles

def compute_angle(p1, p2, p3):
    """Compute the angle between three points."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)
