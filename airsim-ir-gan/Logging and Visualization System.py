import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Union
import numpy as np
import cv2
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px

class LoggingManager:
    """Manages logging for the AirSim-IR system."""
    
    def __init__(self, log_dir: Union[str, Path], level: int = logging.INFO):
        """
        Initialize logging manager.
        
        Args:
            log_dir: Directory for log files
            level: Logging level
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"airsim_ir_{timestamp}.log"
        
        # Configure logging
        self.logger = logging.getLogger("AirSimIR")
        self.logger.setLevel(level)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
    def log_environment_state(self, env_state: Dict) -> None:
        """Log environment state."""
        self.logger.info("Environment State:")
        for key, value in env_state.items():
            self.logger.info(f"  {key}: {value}")
            
    def log_capture_event(self, metadata: Dict) -> None:
        """Log image capture event."""
        self.logger.info(f"Image Captured:")
        self.logger.info(f"  Position: {metadata.get('position')}")
        self.logger.info(f"  Rotation: {metadata.get('rotation')}")
        self.logger.info(f"  Target: {metadata.get('target_type')}")
        
    def log_error(self, error: Exception, context: str = "") -> None:
        """Log error with context."""
        self.logger.error(f"Error in {context}: {str(error)}")
        
class DataVisualizer:
    """Handles visualization of synthetic IR dataset."""
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize data visualizer.
        
        Args:
            output_dir: Directory for saving visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_temperature_distribution(self, 
                                   dataset_stats: pd.DataFrame,
                                   save: bool = True) -> None:
        """Plot temperature distributions by target type."""
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=dataset_stats, x='target_type', y='temperature')
        plt.xticks(rotation=45)
        plt.title('Temperature Distribution by Target Type')
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'temperature_distribution.png')
            plt.close()
            
    def plot_capture_positions(self,
                             metadata_list: List[Dict],
                             save: bool = True) -> None:
        """Create 3D scatter plot of capture positions."""
        positions = np.array([meta['position'] for meta in metadata_list])
        
        fig = go.Figure(data=[go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=positions[:, 2],
                colorscale='Viridis',
            )
        )])
        
        fig.update_layout(
            title='Camera Capture Positions',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)'
            )
        )
        
        if save:
            fig.write_html(str(self.output_dir / 'capture_positions.html'))
            
    def create_image_grid(self,
                         images: List[np.ndarray],
                         grid_size: Tuple[int, int],
                         save_path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """
        Create a grid of images for visualization.
        
        Args:
            images: List of images to display
            grid_size: (rows, cols) for grid
            save_path: Path to save the grid image
            
        Returns:
            Grid image as numpy array
        """
        rows, cols = grid_size
        cell_size = images[0].shape[:2]
        grid = np.zeros((cell_size[0] * rows, cell_size[1] * cols, 3), dtype=np.uint8)
        
        for idx, img in enumerate(images[:rows*cols]):
            i, j = idx // cols, idx % cols
            if len(img.shape) == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            grid[i*cell_size[0]:(i+1)*cell_size[0],
                 j*cell_size[1]:(j+1)*cell_size[1]] = img
                
        if save_path:
            cv2.imwrite(str(save_path), grid)
            
        return grid
    
    def plot_thermal_profile(self,
                           ir_image: np.ndarray,
                           line_position: int,
                           save: bool = True) -> None:
        """Plot thermal profile along a horizontal line."""
        profile = ir_image[line_position, :]
        
        plt.figure(figsize=(10, 4))
        plt.plot(profile, '-b', label='Thermal Profile')
        plt.title(f'Thermal Profile at Line {line_position}')
        plt.xlabel('Pixel Position')
        plt.ylabel('Intensity')
        plt.grid(True)
        plt.legend()
        
        if save:
            plt.savefig(self.output_dir / f'thermal_profile_line_{line_position}.png')
            plt.close()
            
    def create_temporal_visualization(self,
                                   image_sequences: List[np.ndarray],
                                   timestamps: List[float],
                                   save: bool = True) -> None:
        """Create visualization of temporal changes in IR images."""
        fig = plt.figure(figsize=(15, 5))
        
        num_frames = len(image_sequences)
        for i, (img, ts) in enumerate(zip(image_sequences, timestamps)):
            ax = fig.add_subplot(1, num_frames, i+1)
            ax.imshow(img, cmap='inferno')
            ax.set_title(f'T={ts:.1f}s')
            ax.axis('off')
            
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'temporal_changes.png')
            plt.close()
            
    def plot_dataset_statistics(self, stats_df: pd.DataFrame) -> None:
        """Create comprehensive statistical visualizations."""
        # Create subplots
        fig = plt.figure(figsize=(20, 10))
        
        # Temperature vs Time of Day
        plt.subplot(2, 2, 1)
        sns.scatterplot(data=stats_df, x='time_of_day', y='temperature', 
                       hue='target_type', alpha=0.6)
        plt.xticks(rotation=45)
        plt.title('Temperature vs Time of Day')
        
        # IR Intensity Distribution
        plt.subplot(2, 2, 2)
        sns.histplot(data=stats_df, x='ir_mean', hue='target_type', 
                    multiple="stack", bins=30)
        plt.title('IR Intensity Distribution')
        
        # Capture Angle Distribution
        plt.subplot(2, 2, 3)
        sns.histplot(data=stats_df, x='capture_angle', bins=36)
        plt.title('Capture Angle Distribution')
        
        # Target Type Distribution
        plt.subplot(2, 2, 4)
        sns.countplot(data=stats_df, x='target_type')
        plt.xticks(rotation=45)
        plt.title('Target Type Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dataset_statistics.png')
        plt.close()
        
    def create_segmentation_overlay(self,
                                  ir_image: np.ndarray,
                                  seg_image: np.ndarray,
                                  alpha: float = 0.5,
                                  save: bool = True) -> np.ndarray:
        """Create visualization of segmentation overlay on IR image."""
        # Normalize IR image
        ir_norm = cv2.normalize(ir_image, None, 0, 255, cv2.NORM_MINMAX)
        ir_color = cv2.applyColorMap(ir_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)
        
        # Create color-coded segmentation
        seg_color = np.zeros_like(ir_color)
        for seg_id in np.unique(seg_image):
            if seg_id == 0:  # Background
                continue
            mask = seg_image == seg_id
            seg_color[mask] = plt.cm.tab20(seg_id % 20)[:3][::-1] * 255
            
        # Create overlay
        overlay = cv2.addWeighted(ir_color, 1-alpha, seg_color, alpha, 0)
        
        if save:
            cv2.imwrite(str(self.output_dir / 'segmentation_overlay.png'), overlay)
            
        return overlay

def main():
    """Example usage of logging and visualization."""
    # Initialize logging
    logger = LoggingManager("logs")
    logger.logger.info("Starting visualization example")
    
    # Initialize visualizer
    visualizer = DataVisualizer("visualizations")
    
    try:
        # Example: Create some sample data
        stats_df = pd.DataFrame({
            'target_type': ['elephant', 'zebra'] * 50,
            'temperature': np.random.normal(35, 5, 100),
            'time_of_day': np.random.choice(['morning', 'noon', 'evening'], 100),
            'ir_mean': np.random.normal(128, 30, 100),
            'capture_angle': np.random.uniform(0, 360, 100)
        })
        
        # Generate visualizations
        visualizer.plot_dataset_statistics(stats_df)
        visualizer.plot_temperature_distribution(stats_df)
        
        logger.logger.info("Visualizations generated successfully")
        
    except Exception as e:
        logger.log_error(e, "visualization generation")

if __name__ == "__main__":
    main()
