import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

@dataclass
class AirSimConfig:
    """AirSim configuration parameters."""
    sim_mode: str = "ComputerVision"
    clock_speed: float = 1.0
    view_mode: str = "NoDisplay"
    camera_defaults: Dict = None
    
@dataclass
class EnvironmentConfig:
    """Environment configuration parameters."""
    longitude: float = 118.43
    latitude: float = 44.54
    altitude: float = 1190
    default_temperature: float = 15.2
    default_humidity: float = 50.0
    default_wind_speed: float = 2.7
    
@dataclass
class CameraConfig:
    """Camera configuration parameters."""
    width: int = 320
    height: int = 240
    fov_degrees: float = 90.0
    capture_interval: float = 0.1

class ConfigManager:
    """
    Manages configuration settings for AirSim-IR system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path) if config_path else Path("config/default_config.yaml")
        self.logger = logging.getLogger(__name__)
        
        # Set default configurations
        self.airsim_config = AirSimConfig()
        self.env_config = EnvironmentConfig()
        self.camera_config = CameraConfig()
        
        # Load configuration if file exists
        if self.config_path.exists():
            self.load_config()
        else:
            self.logger.warning(f"No configuration file found at {self.config_path}")
            self.save_config()  # Save default configuration
            
    def load_config(self) -> None:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Update configurations
            if 'airsim' in config:
                self.airsim_config = AirSimConfig(**config['airsim'])
            if 'environment' in config:
                self.env_config = EnvironmentConfig(**config['environment'])
            if 'camera' in config:
                self.camera_config = CameraConfig(**config['camera'])
                
            self.logger.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise
            
    def save_config(self) -> None:
        """Save current configuration to file."""
        config = {
            'airsim': {
                'sim_mode': self.airsim_config.sim_mode,
                'clock_speed': self.airsim_config.clock_speed,
                'view_mode': self.airsim_config.view_mode,
                'camera_defaults': self.airsim_config.camera_defaults
            },
            'environment': {
                'longitude': self.env_config.longitude,
                'latitude': self.env_config.latitude,
                'altitude': self.env_config.altitude,
                'default_temperature': self.env_config.default_temperature,
                'default_humidity': self.env_config.default_humidity,
                'default_wind_speed': self.env_config.default_wind_speed
            },
            'camera': {
                'width': self.camera_config.width,
                'height': self.camera_config.height,
                'fov_degrees': self.camera_config.fov_degrees,
                'capture_interval': self.camera_config.capture_interval
            }
        }
        
        # Create directory if it doesn't exist
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            self.logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            raise
            
    def get_airsim_settings(self) -> Dict[str, Any]:
        """
        Get AirSim settings in the format expected by AirSim.
        
        Returns:
            Dictionary of AirSim settings
        """
        return {
            "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",
            "SettingsVersion": 1.2,
            "SimMode": self.airsim_config.sim_mode,
            "ClockSpeed": self.airsim_config.clock_speed,
            "ViewMode": self.airsim_config.view_mode,
            "CameraDefaults": self.airsim_config.camera_defaults or {
                "CaptureSettings": [
                    {
                        "ImageType": 8,  # Infrared
                        "Width": self.camera_config.width,
                        "Height": self.camera_config.height,
                        "FOV_Degrees": self.camera_config.fov_degrees,
                        "AutoExposureSpeed": 100,
                        "AutoExposureBias": 0,
                        "AutoExposureMaxBrightness": 0.64,
                        "AutoExposureMinBrightness": 0.03,
                        "MotionBlurAmount": 0
                    }
                ]
            }
        }
        
    def update_camera_config(self, **kwargs) -> None:
        """Update camera configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.camera_config, key):
                setattr(self.camera_config, key, value)
            else:
                self.logger.warning(f"Unknown camera config parameter: {key}")
                
    def update_environment_config(self, **kwargs) -> None:
        """Update environment configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.env_config, key):
                setattr(self.env_config, key, value)
            else:
                self.logger.warning(f"Unknown environment config parameter: {key}")
                
    def validate_config(self) -> bool:
        """
        Validate current configuration settings.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate camera config
            assert 0 < self.camera_config.width <= 4096
            assert 0 < self.camera_config.height <= 4096
            assert 0 < self.camera_config.fov_degrees <= 180
            assert self.camera_config.capture_interval > 0
            
            # Validate environment config
            assert -90 <= self.env_config.latitude <= 90
            assert -180 <= self.env_config.longitude <= 180
            assert self.env_config.altitude >= 0
            assert -100 <= self.env_config.default_temperature <= 100
            assert 0 <= self.env_config.default_humidity <= 100
            assert self.env_config.default_wind_speed >= 0
            
            return True
            
        except AssertionError as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
