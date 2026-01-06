"""Configuration management for Claude Vault"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class TaggingConfig(BaseModel):
    """Tagging-related configuration"""

    model: str = Field(default="llama3.2:3b", description="Ollama model for tagging")
    mode: str = Field(
        default="quick", description="Tag analysis mode: 'quick' or 'full'"
    )
    use_hierarchy: bool = Field(
        default=False, description="Use primary/secondary/tertiary tag hierarchy"
    )


class SummarizationConfig(BaseModel):
    """Summarization-related configuration"""

    enabled: bool = Field(
        default=False, description="Whether to generate summary report after sync"
    )
    include_ollama_summary: bool = Field(
        default=False, description="Whether to summarize each conversation via Ollama"
    )
    model: str = Field(
        default="llama3.2:3b", description="Ollama model for summarization"
    )
    prompt: str = Field(
        default="Summarize this conversation in 2-3 sentences, focusing on the main topics discussed and any key outcomes or decisions made.",
        description="Prompt template for summarization",
    )


class AppConfig(BaseModel):
    """Main application configuration"""

    naming_pattern: str = Field(default="{date}-{title}")
    folder_structure: str = Field(default="flat")
    template: str = Field(default="default")
    version: str = Field(default="0.4.0")
    tagging: TaggingConfig = Field(default_factory=TaggingConfig)
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)


class ConfigManager:
    """Manages reading and writing configuration"""

    def __init__(self, vault_path: Path):
        self.vault_path = vault_path
        self.config_dir = vault_path / ".claude-vault"
        self.config_file = self.config_dir / "config.json"
        self._config: Optional[AppConfig] = None

    def ensure_initialized(self) -> bool:
        """Check if vault is initialized"""
        return self.config_dir.exists() and self.config_file.exists()

    def load(self) -> AppConfig:
        """Load configuration from file"""
        if self._config is not None:
            return self._config

        if not self.config_file.exists():
            # Return defaults if no config exists
            self._config = AppConfig()
            return self._config

        try:
            with open(self.config_file, encoding="utf-8") as f:
                data = json.load(f)

            # Handle migration from old config format
            if "tagging" not in data:
                data["tagging"] = TaggingConfig().model_dump()
            if "summarization" not in data:
                data["summarization"] = SummarizationConfig().model_dump()

            self._config = AppConfig(**data)
        except Exception as e:
            print(f"Warning: Could not load config: {e}. Using defaults.")
            self._config = AppConfig()

        return self._config

    def save(self, config: Optional[AppConfig] = None) -> None:
        """Save configuration to file"""
        if config is not None:
            self._config = config

        if self._config is None:
            self._config = AppConfig()

        self.config_dir.mkdir(parents=True, exist_ok=True)

        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(self._config.model_dump(), f, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by dot-notation key"""
        config = self.load()
        parts = key.split(".")

        value = config.model_dump()
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value by dot-notation key and save"""
        config = self.load()
        config_dict = config.model_dump()

        parts = key.split(".")
        target = config_dict

        # Navigate to parent
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]

        # Set value
        target[parts[-1]] = value

        # Rebuild config and save
        self._config = AppConfig(**config_dict)
        self.save()

    def update_tagging(
        self,
        model: Optional[str] = None,
        mode: Optional[str] = None,
        use_hierarchy: Optional[bool] = None,
    ) -> None:
        """Update tagging configuration"""
        config = self.load()

        if model is not None:
            config.tagging.model = model
        if mode is not None:
            config.tagging.mode = mode
        if use_hierarchy is not None:
            config.tagging.use_hierarchy = use_hierarchy

        self.save(config)

    def get_tagging_config(self) -> TaggingConfig:
        """Get tagging configuration"""
        return self.load().tagging

    def update_summarization(
        self,
        enabled: Optional[bool] = None,
        include_ollama_summary: Optional[bool] = None,
        model: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> None:
        """Update summarization configuration"""
        config = self.load()

        if enabled is not None:
            config.summarization.enabled = enabled
        if include_ollama_summary is not None:
            config.summarization.include_ollama_summary = include_ollama_summary
        if model is not None:
            config.summarization.model = model
        if prompt is not None:
            config.summarization.prompt = prompt

        self.save(config)

    def get_summarization_config(self) -> SummarizationConfig:
        """Get summarization configuration"""
        return self.load().summarization


def get_default_config() -> AppConfig:
    """Get default configuration"""
    return AppConfig()
