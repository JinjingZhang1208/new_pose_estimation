from roboflow import Roboflow

# Initialize Roboflow with your API key
rf = Roboflow(api_key="eW43mQoE4D7QBLaccpat")

# Access the project and version
project = rf.workspace("annazhang1208").project("vr_pose_avatars")
version = project.version(2)

# Download the dataset in YOLO format
dataset = version.download("yolov5")
