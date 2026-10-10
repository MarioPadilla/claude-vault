import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

APP_NAME = "claude-vault"
CONFIG_DIR = Path.home() / f".{APP_NAME}"
CONFIG_FILE = CONFIG_DIR / "config.json"


class OllamaConfig(BaseModel):
    model: str = "llama3.2:3b"
    url: str = "http://localhost:11434/api/generate"
    timeout: int = 15
    temperature: float = 0.3


class WatchConfig(BaseModel):
    enabled: bool = True
    debounce_seconds: float = 2.0
    throttle_seconds: float = 10.0
    auto_start: bool = False
    watch_paths: List[Dict[str, str]] = Field(default_factory=list)
    max_queue_size: int = 100
    log_level: str = "INFO"


class EmbeddingConfig(BaseModel):
    model: str = "nomic-embed-text"
    enabled: bool = True
    auto_generate: bool = False
    chunk_size: int = 2000
    chunk_overlap: int = 200
    url: str = "http://localhost:11434/api/embed"


class PIIConfig(BaseModel):
    """Configuration for PII and sensitive-content detection"""

    # Whether to run detection at all (also enabled per-command via --detect-pii)
    enabled: bool = False
    # Automatically redact detected PII before writing markdown
    redact: bool = False
    # Skip syncing conversations whose risk level meets or exceeds risk_threshold
    skip_sensitive: bool = False
    # Use Ollama LLM classification in addition to regex (requires Ollama running)
    use_llm: bool = True
    # Minimum risk level that triggers the "sensitive" tag: "low" | "medium" | "high"
    risk_threshold: str = "medium"


class Config(BaseModel):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    watch: WatchConfig = Field(default_factory=WatchConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    pii: PIIConfig = Field(default_factory=PIIConfig)

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
        # utf-8-sig accepts a BOM (Windows editors add one) and plain UTF-8; the
        # locale default would fail on non-ASCII content under cp1252
        with open(CONFIG_FILE, encoding="utf-8-sig") as f:
            data = json.load(f)
            return Config(**data)
    except Exception as e:
        # Defaults mean privacy settings such as pii.redact are off, so say so
        warnings.warn(
            f"Could not read {CONFIG_FILE} ({type(e).__name__}: {e}). "
            "Using default settings, which means PII handling is OFF.",
            stacklevel=2,
        )
        return Config()
