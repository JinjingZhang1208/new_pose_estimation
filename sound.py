from gtts import gTTS
import pygame
import os

def speak(text):
    tts = gTTS(text=text, lang='en')
    temp_file = "temp.mp3"
    tts.save(temp_file)
    
    if os.path.exists(temp_file):
        pygame.mixer.init()
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.quit() 
        os.remove(temp_file)
    else:
        print(f"Error: The file {temp_file} was not created.")