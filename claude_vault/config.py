import json
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field

APP_NAME = "claude-vault"
CONFIG_DIR = Path.home() / f".{APP_NAME}"
CONFIG_FILE = CONFIG_DIR / "config.json"


class OllamaConfig(BaseModel):
    model: str = "llama3.2:3b"
    url: str = "http://localhost:11434/api/generate"
    timeout: int = 15
    temperature: float = 0.3


class Config(BaseModel):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)

    # Custom fallback keywords if user wants to override defaults
    # Mapping of "tag" -> ["keyword1", "keyword2"]
    custom_keywords: Optional[Dict[str, list[str]]] = None


def get_config_path() -> Path:
    return CONFIG_FILE


def load_config() -> Config:
    """Load configuration from file or return defaults"""
    if not CONFIG_FILE.exists():
        return Config()

    try:
        with open(CONFIG_FILE) as f:
            data = json.load(f)
            return Config(**data)
    except Exception:
        # If config is corrupt, return defaults
        return Config()
