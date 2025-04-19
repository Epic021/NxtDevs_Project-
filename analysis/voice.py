import speech_recognition as sr
import pyttsx3
from datetime import datetime
import json
import time

# Initialize the recognizer and text-to-speech engine
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# List to store all transcripts
transcripts = []

# Adjust recognition parameters for better accuracy
recognizer.energy_threshold = 300  # Minimum audio energy to consider for recording
recognizer.dynamic_energy_threshold = True  # Automatically adjust for ambient noise
recognizer.dynamic_energy_adjustment_damping = 0.15
recognizer.dynamic_energy_ratio = 1.5
recognizer.pause_threshold = 0.8  # Seconds of non-speaking audio before a phrase is considered complete
recognizer.operation_timeout = None  # None = no timeout for API operations
recognizer.phrase_threshold = 0.3  # Minimum seconds of speaking audio before we consider the speaking audio a phrase
recognizer.non_speaking_duration = 0.5  # Seconds of non-speaking audio to keep on both sides of the recording

def initialize_microphone():
    """Initialize microphone with improved noise handling"""
    try:
        # Use default microphone
        mic = sr.Microphone()
        print("\nInitializing default microphone and adjusting for ambient noise...")
        with mic as source:
            # Longer ambient noise adjustment for better calibration
            recognizer.adjust_for_ambient_noise(source, duration=2.0)
            
        print("Microphone initialized successfully!")
        return mic
    except Exception as e:
        print(f"Error initializing microphone: {e}")
        return None

def save_transcripts():
    """Save transcripts to a JSON file with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"transcripts_{timestamp}.json"
    
    data = {
        "date": datetime.now().isoformat(),
        "transcripts": transcripts
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Transcripts saved to {filename}")
    except Exception as e:
        print(f"Error saving transcripts: {e}")

def get_best_transcription(audio, retries=3):
    """Attempt to get the best transcription with multiple tries and language options"""
    errors = 0
    best_transcription = None
    languages = ['en-US', 'en-GB', 'en-IN']  # Add more English variants if needed
    
    for language in languages:
        try:
            text = recognizer.recognize_google(audio, language=language)
            if text and len(text) > 0:
                return text.lower()
        except sr.UnknownValueError:
            errors += 1
            continue
        except sr.RequestError:
            continue
    
    if errors >= len(languages):
        raise sr.UnknownValueError("Could not understand audio in any language variant")
    
    return best_transcription

# Initialize default microphone
microphone = initialize_microphone()
if not microphone:
    print("Failed to initialize microphone. Please check your microphone settings.")
    exit(1)

print("\nStarting transcription session.")
print("Commands:")
print("- Press Enter to start recording")
print("- Type 'save' to save transcripts")
print("- Type 'exit' to quit")
print("- Type 'clear' to clear screen")
print("\nSpeaking tips:")
print("- Speak clearly and at a moderate pace")
print("- Keep a consistent distance from the microphone")
print("- Avoid background noise when possible")
print("- Wait for the 'Listening...' prompt before speaking")

while True:
    user_input = input("\nCommand (Enter/save/exit/clear): ").strip().lower()
    
    if user_input == 'exit':
        if transcripts:
            save = input("Would you like to save the transcripts before exiting? (y/n): ").strip().lower()
            if save == 'y':
                save_transcripts()
        print("Exiting program...")
        break
    
    elif user_input == 'save':
        save_transcripts()
        continue
    
    elif user_input == 'clear':
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        continue

    try:
        with microphone as mic:
            print("\nListening... (Speak now)")
            # Added parameters for better recognition
            audio = recognizer.listen(
                mic,
                timeout=10,
                phrase_time_limit=30,
                snowboy_configuration=None
            )
            
            print("Processing speech...")
            
            # Try to get the best transcription
            text = get_best_transcription(audio)
            
            # Create a transcript entry with timestamp and text
            transcript_entry = {
                "timestamp": datetime.now().isoformat(),
                "text": text
            }
            
            # Add to transcripts list
            transcripts.append(transcript_entry)
            
            # Print the recognized text with clear formatting
            print("\n" + "="*50)
            print(f"Recognized text: {text}")
            print(f"Total transcripts: {len(transcripts)}")
            print("="*50)

    except sr.WaitTimeoutError:
        print("\nNo speech detected within timeout period. Please try again.")
        continue
    except sr.UnknownValueError:
        print("\nSorry, could not understand the audio. Tips:")
        print("- Speak more clearly and slowly")
        print("- Reduce background noise")
        print("- Move closer to the microphone")
        continue
    except sr.RequestError as e:
        print(f"\nError with the speech recognition service: {e}")
        print("Trying again...")
        time.sleep(2)  # Wait before retrying
        continue
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        continue
