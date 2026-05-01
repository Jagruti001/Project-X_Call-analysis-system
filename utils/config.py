"""Configuration management for the call analysis system."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger
import yaml


class Config:
    """
    Configuration manager for the call analysis system.
    
    Supports:
    - Environment variables
    - YAML config files
    - Default values
    - Type-safe access
    """
    
    DEFAULT_CONFIG = {
        # Whisper Configuration
        "whisper": {
            "model_size": "base",  # tiny, base, small, medium, large
            "device": None  # None for auto-detect, 'cpu', or 'cuda'
        },
        
        # LLM Configuration
        "llm": {
            "base_url": "http://localhost:11434",
            "model": "qwen2.5:3b",
            "temperature": 0.3,
            "max_retries": 3
        },
        
        # Storage Configuration
        "storage": {
            "transcripts_dir": "data/transcripts",
            "analysis_dir": "data/analysis"
        },
        
        # ChromaDB Configuration
        "chromadb": {
            "persist_directory": "data/chromadb",
            "collection_name": "call_issues"
        },
        
        # Embeddings Configuration
        "embeddings": {
            "model": "sentence-transformers/all-MiniLM-L6-v2"
        },
        
        # Clustering Configuration
        "clustering": {
            "min_cluster_size": 3,
            "min_samples": 2,
            "metric": "cosine"
        },
        
        # System Configuration
        "system": {
            "log_level": "INFO",
            "batch_size": 10
        }
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Optional path to YAML config file
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Load from file if provided
        if config_file and Path(config_file).exists():
            self._load_from_file(config_file)
        
        # Override with environment variables
        self._load_from_env()
        
        # Setup logging
        self._setup_logging()
    
    def _load_from_file(self, config_file: str):
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
            
            if file_config:
                self._deep_update(self.config, file_config)
                logger.info(f"Configuration loaded from {config_file}")
                
        except Exception as e:
            logger.warning(f"Could not load config file {config_file}: {e}")
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # LLM settings
        if os.getenv("OLLAMA_BASE_URL"):
            self.config["llm"]["base_url"] = os.getenv("OLLAMA_BASE_URL")
        
        if os.getenv("LLM_MODEL"):
            self.config["llm"]["model"] = os.getenv("LLM_MODEL")
        
        # Whisper settings
        if os.getenv("WHISPER_MODEL"):
            self.config["whisper"]["model_size"] = os.getenv("WHISPER_MODEL")
        
        if os.getenv("WHISPER_DEVICE"):
            self.config["whisper"]["device"] = os.getenv("WHISPER_DEVICE")
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Recursively update nested dictionary."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _setup_logging(self):
        """Setup logging based on configuration."""
        log_level = self.config["system"].get("log_level", "INFO")
        logger.remove()  # Remove default handler
        logger.add(
            lambda msg: print(msg, end=""),
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'llm.model')
            default: Default value if key not found
            
        Returns:
            Configuration value
        
        Example:
            >>> config = Config()
            >>> model = config.get('llm.model')
            >>> print(model)  # 'qwen2.5:3b'
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict:
        """Get full configuration as dictionary."""
        return self.config.copy()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Config({self.config})"


def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        "data/audio",
        "data/transcripts",
        "data/analysis",
        "data/chromadb"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ All directories ensured")


def cleanup_audio(audio_path: str, keep_file: bool = False):
    """
    Clean up temporary audio file.
    
    Args:
        audio_path: Path to audio file
        keep_file: If True, don't delete the file
    """
    if keep_file:
        return
    
    try:
        audio_file = Path(audio_path)
        if audio_file.exists() and audio_file.parent.name == "audio":
            audio_file.unlink()
            logger.debug(f"Cleaned up audio file: {audio_path}")
    except Exception as e:
        logger.warning(f"Could not clean up {audio_path}: {e}")