import threading
import capture
import detect
import os

save_dir = 'captured_images'

if __name__ == "__main__":
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Start capturing images in a separate thread
    capture_images_thread = threading.Thread(target=capture.capture_images, args=(save_dir,))
    capture_images_thread.start()

    # Run the detection function
    detect.monitor_images(save_dir)