import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import librosa
import numpy as np

# --- Configuration ---
# Set the duration of the recording in seconds.
DURATION = 10  
# Set the sample rate (Hz). 44100 Hz is standard CD quality.
SAMPLE_RATE = 44100
# Define the output filename for the audio file.
AUDIO_FILENAME = "recorded_voice.wav"

def record_audio(filename, duration, samplerate):
    """
    Records audio from the default input device for a specified duration.
    
    Args:
        filename (str): The name of the file to save the audio to.
        duration (int): The recording duration in seconds.
        samplerate (int): The sample rate of the audio.
    """
    print(f"🎙 Recording for {duration} seconds... Please speak clearly.")
    try:
        # Record audio data from the microphone.
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        # Wait until the recording is finished.
        sd.wait()
        # Save the recorded data to a WAV file.
        sf.write(filename, audio_data, samplerate)
        print("✅ Recording complete. Audio saved to " + filename)
    except Exception as e:
        print(f"❌ An error occurred during recording: {e}")
        return False
    return True

def speech_to_text(filename):
    """
    Transcribes speech from an audio file to text using Google's Speech Recognition API.
    
    Args:
        filename (str): The path to the audio file.
        
    Returns:
        str: The transcribed text, or None if transcription fails.
    """
    recognizer = sr.Recognizer()
    try:
        # Load the audio file.
        with sr.AudioFile(filename) as source:
            # Adjust for ambient noise and read the audio data.
            print("⏳ Transcribing audio...")
            audio_data = recognizer.record(source)
        
        # Use Google's Speech Recognition API to transcribe the audio.
        # This requires an internet connection.
        text = recognizer.recognize_google(audio_data)
        print("✅ Transcription complete.")
        return text
    except sr.UnknownValueError:
        print("❌ Could not understand the audio. Please try again.")
        return None
    except sr.RequestError as e:
        print(f"❌ Could not request results from Google Speech Recognition service; {e}")
        return None
    except Exception as e:
        print(f"❌ An error occurred during transcription: {e}")
        return None

def analyze_sentiment(text):
    """
    Analyzes the sentiment of a given text using the VADER model.
    
    Args:
        text (str): The text to analyze.
        
    Returns:
        dict: A dictionary containing sentiment scores.
    """
    print("🧠 Analyzing sentiment...")
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    print("✅ Sentiment analysis complete.")
    return sentiment_scores

def calculate_pitch(filename, samplerate):
    """
    Calculates the average pitch of an audio file and classifies it as high or low.
    
    Args:
        filename (str): The path to the audio file.
        samplerate (int): The sample rate of the audio.
        
    Returns:
        tuple: A tuple containing the average pitch (float) and a pitch classification (str).
    """
    print("🎶 Calculating pitch...")
    try:
        # Load the audio file and get the pitch using librosa.
        y, sr = librosa.load(filename, sr=samplerate)
        pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
        
        # Filter out pitches that are not significant (low magnitude).
        # This helps to remove background noise and silence.
        pitch_values = pitches[magnitudes > 0.1]
        
        # If there are no pitch values found, return a default.
        if len(pitch_values) == 0:
            return 0.0, "Undetermined"
        
        # Calculate the average pitch.
        mean_pitch = np.mean(pitch_values)
        
        # Define a simple threshold to classify pitch.
        # These values can be adjusted based on typical human voice ranges.
        # Male average: 85-180 Hz, Female average: 165-255 Hz.
        PITCH_THRESHOLD = 150.0  # Adjust as needed.
        
        if mean_pitch > PITCH_THRESHOLD:
            pitch_class = "High"
        else:
            pitch_class = "Low"
            
        print("✅ Pitch calculation complete.")
        return mean_pitch, pitch_class

    except Exception as e:
        print(f"❌ An error occurred during pitch calculation: {e}")
        return 0.0, "Undetermined"

def main():
    """Main function to run the entire analysis workflow."""
    print("--- Voice Sentimental & Pitch Analysis Tool ---")
    
    # Step 1: Record the voice.
    if not record_audio(AUDIO_FILENAME, DURATION, SAMPLE_RATE):
        return
        
    # Step 2: Convert speech to text.
    text = speech_to_text(AUDIO_FILENAME)
    
    if text:
        print(f"\n🗣 Transcribed Text: \"{text}\"\n")
        
        # Step 3: Analyze sentiment.
        sentiment_scores = analyze_sentiment(text)
        print("--- Sentiment Analysis Results ---")
        for key, value in sentiment_scores.items():
            print(f"   {key.capitalize()}: {value:.2f}")
            
        # Step 4: Calculate and analyze pitch.
        pitch_value, pitch_class = calculate_pitch(AUDIO_FILENAME, SAMPLE_RATE)
        print("\n--- Pitch Analysis Results ---")
        print(f"   Calculated Pitch: {pitch_value:.2f} Hz")
        print(f"   Pitch Classification: {pitch_class}")
        
    else:
        print("\n🚫 Cannot proceed with sentiment and pitch analysis without a successful transcription.")

if __name__ == "_main_":
    main() 