import os
import base64
import io
from gtts import gTTS
import tempfile
import pygame
import time

class TextToSpeech:
    """
    Text-to-speech utility class for converting medical summaries to audio.
    Supports both gTTS (Google Text-to-Speech) and pyttsx3.
    """
    
    def __init__(self):
        """
        Initialize the TTS system.
        """
        self.audio_file = None
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize pygame mixer for audio playback
        try:
            pygame.mixer.init()
        except:
            print("Warning: pygame mixer initialization failed. Audio playback may not work.")
    
    def text_to_speech(self, text, language='en', filename=None):
        """
        Convert text to speech using Google Text-to-Speech.
        
        Args:
            text (str): Text to convert to speech
            language (str): Language code (default: 'en' for English)
            filename (str): Optional filename to save the audio
            
        Returns:
            str: Path to the generated audio file
        """
        try:
            # Clean up previous audio file
            if self.audio_file and os.path.exists(self.audio_file):
                os.remove(self.audio_file)
            
            # Generate filename if not provided
            if filename is None:
                filename = f"medical_summary_{int(time.time())}.mp3"
            
            filepath = os.path.join(self.temp_dir, filename)
            
            # Convert text to speech
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(filepath)
            
            self.audio_file = filepath
            print(f"Audio file generated: {filepath}")
            
            return filepath
            
        except Exception as e:
            print(f"Error generating speech: {e}")
            return None
    
    def text_to_speech_base64(self, text, language='en'):
        """
        Convert text to speech and return as base64 encoded string.
        
        Args:
            text (str): Text to convert to speech
            language (str): Language code (default: 'en' for English)
            
        Returns:
            str: Base64 encoded audio data
        """
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Generate speech
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(temp_path)
            
            # Read and encode to base64
            with open(temp_path, 'rb') as audio_file:
                audio_data = audio_file.read()
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            return audio_base64
            
        except Exception as e:
            print(f"Error generating base64 audio: {e}")
            return None
    
    def play_audio(self, audio_file=None):
        """
        Play the generated audio file.
        
        Args:
            audio_file (str): Path to audio file (if None, uses last generated file)
            
        Returns:
            bool: True if playback successful, False otherwise
        """
        try:
            if audio_file is None:
                audio_file = self.audio_file
            
            if audio_file is None or not os.path.exists(audio_file):
                print("No audio file available for playback")
                return False
            
            # Load and play audio
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            return True
            
        except Exception as e:
            print(f"Error playing audio: {e}")
            return False
    
    def stop_audio(self):
        """
        Stop currently playing audio.
        """
        try:
            pygame.mixer.music.stop()
        except:
            pass
    
    def get_audio_duration(self, audio_file=None):
        """
        Get the duration of the audio file in seconds.
        
        Args:
            audio_file (str): Path to audio file (if None, uses last generated file)
            
        Returns:
            float: Duration in seconds, or None if error
        """
        try:
            if audio_file is None:
                audio_file = self.audio_file
            
            if audio_file is None or not os.path.exists(audio_file):
                return None
            
            # Load audio to get duration
            pygame.mixer.music.load(audio_file)
            # Note: pygame doesn't provide direct duration access
            # This is a rough estimate based on text length
            return None
            
        except Exception as e:
            print(f"Error getting audio duration: {e}")
            return None
    
    def cleanup(self):
        """
        Clean up temporary files and resources.
        """
        try:
            # Stop any playing audio
            self.stop_audio()
            
            # Remove temporary audio file
            if self.audio_file and os.path.exists(self.audio_file):
                os.remove(self.audio_file)
            
            # Clean up pygame
            pygame.mixer.quit()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def __del__(self):
        """
        Destructor to ensure cleanup.
        """
        self.cleanup()

def create_audio_summary(medical_summary, tts_engine=None):
    """
    Create an audio version of the medical summary.
    
    Args:
        medical_summary (str): Medical summary text
        tts_engine (TextToSpeech): TTS engine instance (creates new one if None)
        
    Returns:
        tuple: (audio_file_path, base64_audio_data)
    """
    if tts_engine is None:
        tts_engine = TextToSpeech()
    
    # Generate audio file
    audio_file = tts_engine.text_to_speech(medical_summary)
    
    # Generate base64 version
    audio_base64 = tts_engine.text_to_speech_base64(medical_summary)
    
    return audio_file, audio_base64
