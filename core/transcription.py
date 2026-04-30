"""Audio transcription using OpenAI Whisper."""


import whisper #main speech-to-text model
import torch #backend (CPU/GPU + memory handling)
from pathlib import Path #safe file handling
from typing import Dict #type hints (clean code)
from loguru import logger #logs instead of print
import gc #garbage collection (memory cleanup)

class TranscriptionEngine:
    """Handles audio-to-text transcription using Whisper."""
    
    def __init__(self, model_size: str = "base", device: str = None):
        """
        Initialize Whisper model.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (cpu or cuda)
        """
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load Whisper model into memory."""
        logger.info(f"Loading Whisper {self.model_size} model on {self.device}...")
        
        try:
            self.model = whisper.load_model(self.model_size, device=self.device)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    def _clean_text(self, text: str) -> str:
        """Clean transcript text."""
        return " ".join(text.split()).strip()


    def transcribe(self, audio_path: str, language: str = "en") -> Dict[str, any]:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            language: Language code (default: en)
            
        Returns:
            Dict with transcript and metadata
        """
        if isinstance(audio_path, dict):  # mock / pre-transcribed input
            logger.info("Using pre-transcribed input (mock mode)")
            return audio_path

        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        logger.info(f"Transcribing: {audio_path}")
        
        try:
            # Transcribe (Output of result--> Text+timestamp)

            result = self.model.transcribe(
                audio_path,
                language=language,
                fp16=False,  # Use FP32 for CPU
                verbose=False
            )
            
            transcript_data = {
                "text": self._clean_text(result["text"]),
                "language": result["language"],
                "segments": result["segments"],
                "audio_file": str(audio_path)
            }
            
            logger.info(f"Transcription complete: {len(transcript_data['text'])} characters")
            
            return transcript_data #output goes to next agent
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

        # Scaaling logic(mini version)
    def batch_transcribe(self, audio_files: list, language: str = "en") -> list:
        """
        Transcribe multiple audio files.
        
        Args:
            audio_files: List of audio file paths
            language: Language code
            
        Returns:
            List of transcript dictionaries
        """
        results = []
        
        for audio_file in audio_files:
            try:
                result = self.transcribe(audio_file, language)
                results.append(result)
                
                # Memory cleanup after each file
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Failed to transcribe {audio_file}: {e}")
                results.append({"error": str(e), "audio_file": audio_file})
                
        return results
    
    def cleanup(self):
        """Free model from memory."""
        if self.model is not None:
            del self.model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Whisper model cleaned from memory")
