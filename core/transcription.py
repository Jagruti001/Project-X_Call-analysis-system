"""audio transcription using OpenAI Whisper."""

import whisper
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger
import gc
import time
from functools import wraps


# ===========================
# RETRY DECORATOR
# ===========================

def retry_on_failure(max_attempts: int = 3, delay: float = 2.0):
    """Retry decorator for handling transient failures."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            f"Attempt {attempt}/{max_attempts} failed: {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay * attempt)
                    else:
                        logger.error(f"All {max_attempts} attempts failed: {e}")
            raise last_exception
        return wrapper
    return decorator


# ===========================
# TRANSCRIPTION ENGINE
# ===========================

class TranscriptionEngine:
    """
    Features:
    - Multiple model sizes (tiny to large)
    - GPU/CPU support with automatic detection
    - Memory management and cleanup
    - Batch processing support
    - Comprehensive error handling
    - Performance monitoring
    """
    
    def __init__(
        self, 
        model_size: str = "base", 
        device: Optional[str] = None,
        download_root: Optional[str] = None
    ):
        """
        Initialize Whisper transcription engine.
        
        Args:
            model_size: Whisper model size
                - tiny: Fastest, least accurate (~1GB RAM)
                - base: Fast, good accuracy (~1GB RAM) [RECOMMENDED]
                - small: Slower, better accuracy (~2GB RAM)
                - medium: Slow, high accuracy (~5GB RAM)
                - large: Slowest, best accuracy (~10GB RAM)
            device: Device to run on ('cpu', 'cuda', or None for auto-detect)
            download_root: Directory to store model files
        """
        self.model_size = model_size
        self.device = device or self._detect_device()
        self.download_root = download_root
        self.model = None
        
        logger.info(
            f"Initializing TranscriptionEngine: "
            f"model={model_size}, device={self.device}"
        )
        
        self._load_model()
    
    # ===========================
    # MODEL MANAGEMENT
    # ===========================
    
    def _detect_device(self) -> str:
        """Automatically detect best available device."""
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {gpu_name}")
        else:
            device = "cpu"
            logger.info("No GPU detected - using CPU")
        
        return device
    
    def _load_model(self):
        """Load Whisper model into memory with error handling."""
        logger.info(f"Loading Whisper '{self.model_size}' model on {self.device}...")
        
        try:
            start_time = time.time()
            
            self.model = whisper.load_model(
                self.model_size, 
                device=self.device,
                download_root=self.download_root
            )
            
            load_time = time.time() - start_time
            logger.info(f"✅ Whisper model loaded successfully in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
            raise RuntimeError(f"Could not initialize Whisper model: {e}")
    
    # ===========================
    # TRANSCRIPTION
    # ===========================
    
    @retry_on_failure(max_attempts=2, delay=2.0)
    def transcribe(
        self, 
        audio_path: str, 
        language: str = "en",
        task: str = "transcribe",
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file (WAV, MP3, M4A, etc.)
            language: Language code (default: 'en' for English)
            task: 'transcribe' or 'translate' (translate to English)
            verbose: Show detailed progress
            
        Returns:
            Dictionary containing:
                - text: Full transcript
                - language: Detected/specified language
                - segments: List of timestamped segments
                - audio_file: Path to audio file
                - duration_seconds: Audio duration
                - transcription_time: Processing time
                
        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If transcription fails
        """
        # Handle mock/pre-transcribed input (for testing)
        if isinstance(audio_path, dict):
            logger.info("Using pre-transcribed input (test mode)")
            return audio_path
        
        # Validate file exists
        audio_path_obj = Path(audio_path)
        if not audio_path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"🎤 Transcribing: {audio_path_obj.name}")
        
        try:
            start_time = time.time()
            
            # Transcribe audio
            result = self.model.transcribe(
                str(audio_path),
                language=language,
                task=task,
                fp16=False,  # Use FP32 for CPU compatibility
                verbose=verbose,
                condition_on_previous_text=True,  # Better accuracy
                temperature=0.0  # Deterministic output
            )
            
            transcription_time = time.time() - start_time
            
            # Build structured output
            transcript_data = {
                "text": self._clean_text(result["text"]),
                "language": result.get("language", language),
                "segments": result.get("segments", []),
                "audio_file": str(audio_path),
                "duration_seconds": self._calculate_duration(result["segments"]),
                "transcription_time_seconds": round(transcription_time, 2),
                "model": self.model_size,
                "device": self.device,
                "word_count": len(result["text"].split()),
                "character_count": len(result["text"])
            }
            
            logger.info(
                f"✅ Transcription complete: "
                f"{transcript_data['word_count']} words, "
                f"{transcript_data['duration_seconds']:.1f}s audio, "
                f"processed in {transcription_time:.1f}s"
            )
            
            return transcript_data
            
        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}", exc_info=True)
            raise RuntimeError(f"Transcription error: {e}")
    
    # ===========================
    # BATCH PROCESSING
    # ===========================
    
    def batch_transcribe(
        self, 
        audio_files: List[str], 
        language: str = "en",
        cleanup_memory: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Transcribe multiple audio files in batch.
        
        Args:
            audio_files: List of audio file paths
            language: Language code
            cleanup_memory: Run garbage collection after each file
            
        Returns:
            List of transcript dictionaries (same as transcribe())
        """
        logger.info(f"📚 Batch transcribing {len(audio_files)} files...")
        
        results = []
        successful = 0
        failed = 0
        
        for idx, audio_file in enumerate(audio_files, 1):
            logger.info(f"Processing {idx}/{len(audio_files)}: {Path(audio_file).name}")
            
            try:
                result = self.transcribe(audio_file, language=language)
                results.append(result)
                successful += 1
                
                # Memory cleanup after each file
                if cleanup_memory:
                    self._cleanup_memory()
                    
            except Exception as e:
                logger.error(f"Failed to transcribe {audio_file}: {e}")
                results.append({
                    "error": str(e),
                    "audio_file": audio_file,
                    "status": "failed"
                })
                failed += 1
        
        logger.info(
            f"✅ Batch complete: {successful} successful, {failed} failed"
        )
        
        return results
    
    # ===========================
    # UTILITIES
    # ===========================
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize transcript text.
        
        Args:
            text: Raw transcript text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        cleaned = " ".join(text.split())
        
        # Strip leading/trailing whitespace
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _calculate_duration(self, segments: List[Dict]) -> float:
        """Calculate total audio duration from segments."""
        if not segments:
            return 0.0
        
        try:
            last_segment = segments[-1]
            return round(last_segment.get("end", 0.0), 2)
        except Exception:
            return 0.0
    
    def _cleanup_memory(self):
        """Clean up memory after transcription."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ===========================
    # ADVANCED FEATURES
    # ===========================
    
    def transcribe_with_timestamps(
        self, 
        audio_path: str, 
        language: str = "en"
    ) -> List[Dict[str, Any]]:
        """
        Transcribe with detailed timestamp information.
        
        Args:
            audio_path: Path to audio file
            language: Language code
            
        Returns:
            List of segments with timestamps
        """
        result = self.transcribe(audio_path, language=language)
        
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "text": seg.get("text", "").strip(),
                "start": round(seg.get("start", 0.0), 2),
                "end": round(seg.get("end", 0.0), 2),
                "duration": round(seg.get("end", 0.0) - seg.get("start", 0.0), 2)
            })
        
        return segments
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get transcription engine information.
        
        Returns:
            Engine configuration and status
        """
        return {
            "model_size": self.model_size,
            "device": self.device,
            "model_loaded": self.model is not None,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
    
    # ===========================
    # CLEANUP
    # ===========================
    
    def cleanup(self):
        """Free model from memory and clean up resources."""
        if self.model is not None:
            logger.info("Cleaning up Whisper model from memory...")
            del self.model
            self.model = None
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("✅ Whisper model cleaned from memory")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass
    
    # ===========================
    # CONTEXT MANAGER
    # ===========================
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatic cleanup."""
        self.cleanup()
        return False


# ===========================
# CONVENIENCE FUNCTIONS
# ===========================

def quick_transcribe(audio_path: str, model_size: str = "base") -> str:
    """
    Quick transcription convenience function.
    
    Args:
        audio_path: Path to audio file
        model_size: Whisper model size
        
    Returns:
        Transcript text
    
    Example:
        >>> text = quick_transcribe("call.mp3")
        >>> print(text)
    """
    with TranscriptionEngine(model_size=model_size) as engine:
        result = engine.transcribe(audio_path)
        return result["text"]


def transcribe_folder(
    folder_path: str, 
    model_size: str = "base",
    file_extensions: List[str] = None
) -> Dict[str, str]:
    """
    Transcribe all audio files in a folder.
    
    Args:
        folder_path: Path to folder containing audio files
        model_size: Whisper model size
        file_extensions: List of extensions to process (default: common audio formats)
        
    Returns:
        Dictionary mapping filenames to transcripts
    
    Example:
        >>> results = transcribe_folder("./audio_files/")
        >>> for filename, transcript in results.items():
        >>>     print(f"{filename}: {transcript[:100]}...")
    """
    if file_extensions is None:
        file_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
    
    folder = Path(folder_path)
    audio_files = []
    
    for ext in file_extensions:
        audio_files.extend(folder.glob(f"*{ext}"))
    
    results = {}
    
    with TranscriptionEngine(model_size=model_size) as engine:
        for audio_file in audio_files:
            try:
                result = engine.transcribe(str(audio_file))
                results[audio_file.name] = result["text"]
            except Exception as e:
                logger.error(f"Failed to transcribe {audio_file.name}: {e}")
                results[audio_file.name] = f"ERROR: {e}"
    
    return results