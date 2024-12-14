import airsim
import numpy as np
from pathlib import Path
import time
from datetime import datetime

from environment.african_environment import AfricanEnvironment
from thermal.thermal_model import ThermalModel
from capture.ir_capture import IRImageCapture

def main():
    """Main function to run the AirSim-IR simulation."""
    
    # Initialize components
    env = AfricanEnvironment()
    thermal = ThermalModel()
    capture = IRImageCapture()
    
    # Setup output directory
    output_dir = Path("output/synthetic_ir_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure environment parameters
    env_configs = [
        # Morning
        {
            'time': '07:00',
            'temperature': 15.2,
            'humidity': 50.0,
            'wind_speed': 2.7,
            'cloud_cover': 0.3
        },
        # Noon
        {
            'time': '12:00',
            'temperature': 25.5,
            'humidity': 40.0,
            'wind_speed': 3.5,
            'cloud_cover': 0.1
        },
        # Evening
        {
            'time': '17:00',
            'temperature': 22.0,
            'humidity': 55.0,
            'wind_speed': 2.0,
            'cloud_cover': 0.4
        }
    ]
    
    # Define capture targets
    targets = [
        {
            'name': 'elephant',
            'position': (10, 0, 0),
            'temperature': 37.0
        },
        {
            'name': 'zebra',
            'position': (-5, 8, 0),
            'temperature': 38.5
        },
        {
            'name': 'vehicle',
            'position': (0, -12, 0),
            'temperature': 45.0
        }
    ]
    
    # Generate dataset
    for config in env_configs:
        # Set environment conditions
        env.set_time_of_day(config['time'])
        env.set_weather(
            temperature=config['temperature'],
            humidity=config['humidity'],
            wind_speed=config['wind_speed'],
            cloud_cover=config['cloud_cover']
        )
        
        # Get thermal parameters
        thermal_params = env.get_thermal_parameters()
        
        # Create segmentation map
        object_temps = {
            target['name']: target['temperature'] 
            for target in targets
        }
        capture.create_segmentation_map(object_temps)
        
        # Capture images for each target
        for target in targets:
            print(f"Capturing {target['name']} at {config['time']}")
            
            # Calculate target temperature considering environment
            target_temp = thermal.calculate_temperature_distribution(
                ambient_temp=config['temperature'],
                material=target['name'],
                solar_radiation=1000 * (1 - config['cloud_cover']),
                wind_speed=config['wind_speed']
            )
            
            # Setup output subdirectory
            target_dir = output_dir / f"{target['name']}_{config['time']}"
            target_dir.mkdir(exist_ok=True)
            
            # Capture multi-angle images
            metadata = capture.capture_multi_angle(
                target_position=target['position'],
                output_dir=str(target_dir),
                num_angles=20,
                radius_range=(5, 15),
                height_range=(2, 10)
            )
            
            # Add thermal parameters to metadata
            for meta in metadata:
                meta['thermal_params'] = thermal_params
                meta['target_temperature'] = target_temp
                
            print(f"Captured {len(metadata)} images")
            time.sleep(1)  # Allow system to stabilize

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        airsim.MultirotorClient().reset()
