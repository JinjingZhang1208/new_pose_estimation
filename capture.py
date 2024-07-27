import os
import time
import pyautogui
import keyboard

def capture_images(save_dir, interval=0.033):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Press 'q' to stop capturing.")
    
    try:
        i = 0
        while True:
            if keyboard.is_pressed('q'):
                print("Image capture stopped.")
                break

            screenshot = pyautogui.screenshot()
            screenshot_path = os.path.join(save_dir, f'image_{i + 1:03}.png')
            screenshot.save(screenshot_path)
            print(f'Captured {screenshot_path}')
            time.sleep(interval)
            i += 1
            
    except KeyboardInterrupt:
        print("Image capture stopped by user.")