import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded successfully from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

def save_config(config, save_path):
    """Save configuration to YAML file."""
    try:
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Configuration saved successfully to {save_path}")
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        raise
