"""Utility functions for configuration and logging."""

import yaml
import os
from pathlib import Path
from loguru import logger
from typing import Dict, Any

class Config:
    """Configuration manager."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._setup_logging()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logger.add(
            "logs/call_analysis_{time}.log",
            rotation="500 MB",
            retention="10 days",
            level="INFO"
        )
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
                
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access."""
        return self.get(key)


def ensure_directories():
    """Ensure all required directories exist."""
    dirs = [
        "data/audio",
        "data/transcripts", 
        "data/analysis",
        "data/chromadb",
        "logs"
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        
    logger.info("All directories created/verified")


def cleanup_audio(file_path: str):
    """Delete audio file to free memory."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up audio file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup {file_path}: {e}")
