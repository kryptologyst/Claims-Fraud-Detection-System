"""Configuration management utilities."""

from typing import Any, Dict
import logging
from pathlib import Path
import yaml
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to the main configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML files."""
        try:
            config = OmegaConf.load(self.config_path)
            return OmegaConf.to_container(config, resolve=True)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
            
    def update(self, key: str, value: Any) -> None:
        """Update configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: New value
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
        
    def save(self, output_path: str = None) -> None:
        """Save configuration to file.
        
        Args:
            output_path: Output file path (defaults to original path)
        """
        output_path = output_path or self.config_path
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)
        
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style assignment."""
        self.update(key, value)
