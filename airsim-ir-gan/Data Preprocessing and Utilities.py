import numpy as np
import cv2
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import h5py
import pandas as pd
from tqdm import tqdm

@dataclass
class ImagePair:
    """Container for IR image pairs and metadata."""
    ir_image: np.ndarray
    segmentation: np.ndarray
    scene_image: np.ndarray
    metadata: Dict
    timestamp: float
    target_type: str

class DataPreprocessor:
    """
    Handles preprocessing and organization of synthetic IR dataset.
    """
    
    def __init__(self,
                data_dir: Union[str, Path],
                output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize data preprocessor.
        
        Args:
            data_dir: Directory containing raw synthetic data
            output_dir: Directory to save processed data
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Image normalization parameters
        self.norm_params = {
            'ir_mean': None,
            'ir_std': None,
            'seg_mean': None,
            'seg_std': None
        }
        
    def load_dataset(self) -> List[ImagePair]:
        """
        Load and organize the synthetic dataset.
        
        Returns:
            List of ImagePair objects
        """
        image_pairs = []
        
        # Iterate through all subdirectories
        for target_dir in tqdm(list(self.data_dir.glob("*_*"))):
            if not target_dir.is_dir():
                continue
                
            target_type = target_dir.name.split('_')[0]
            
            # Load all images and metadata for this target
            for metadata_file in target_dir.glob("metadata_*.json"):
                idx = metadata_file.stem.split('_')[1]
                
                # Load metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    
                # Load corresponding images
                ir_path = target_dir / f"ir_{idx}.png"
                seg_path = target_dir / f"segmentation_{idx}.png"
                scene_path = target_dir / f"scene_{idx}.png"
                
                if not all(p.exists() for p in [ir_path, seg_path, scene_path]):
                    continue
                    
                ir_img = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
                seg_img = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
                scene_img = cv2.imread(str(scene_path))
                
                image_pairs.append(ImagePair(
                    ir_image=ir_img,
                    segmentation=seg_img,
                    scene_image=scene_img,
                    metadata=metadata,
                    timestamp=metadata['timestamp'],
                    target_type=target_type
                ))
                
        return image_pairs
    
    def calculate_normalization_params(self, image_pairs: List[ImagePair]) -> None:
        """Calculate dataset statistics for normalization."""
        ir_values = np.concatenate([pair.ir_image.flatten() for pair in image_pairs])
        seg_values = np.concatenate([pair.segmentation.flatten() for pair in image_pairs])
        
        self.norm_params['ir_mean'] = np.mean(ir_values)
        self.norm_params['ir_std'] = np.std(ir_values)
        self.norm_params['seg_mean'] = np.mean(seg_values)
        self.norm_params['seg_std'] = np.std(seg_values)
        
        # Save normalization parameters
        with open(self.output_dir / 'normalization_params.json', 'w') as f:
            json.dump(self.norm_params, f, indent=2)
    
    def preprocess_images(self, 
                         image_pairs: List[ImagePair],
                         normalize: bool = True,
                         augment: bool = True) -> List[ImagePair]:
        """
        Preprocess images with normalization and augmentation.
        
        Args:
            image_pairs: List of ImagePair objects
            normalize: Whether to normalize images
            augment: Whether to apply data augmentation
            
        Returns:
            Processed image pairs
        """
        processed_pairs = []
        
        for pair in tqdm(image_pairs):
            # Normalize if requested
            if normalize:
                ir_norm = (pair.ir_image - self.norm_params['ir_mean']) / self.norm_params['ir_std']
                seg_norm = (pair.segmentation - self.norm_params['seg_mean']) / self.norm_params['seg_std']
            else:
                ir_norm = pair.ir_image
                seg_norm = pair.segmentation
            
            processed_pairs.append(ImagePair(
                ir_image=ir_norm,
                segmentation=seg_norm,
                scene_image=pair.scene_image,
                metadata=pair.metadata,
                timestamp=pair.timestamp,
                target_type=pair.target_type
            ))
            
            # Apply augmentations if requested
            if augment:
                # Horizontal flip
                processed_pairs.append(ImagePair(
                    ir_image=np.fliplr(ir_norm),
                    segmentation=np.fliplr(seg_norm),
                    scene_image=cv2.flip(pair.scene_image, 1),
                    metadata={**pair.metadata, 'augmentation': 'horizontal_flip'},
                    timestamp=pair.timestamp,
                    target_type=pair.target_type
                ))
                
                # Rotation augmentations
                for angle in [90, 180, 270]:
                    matrix = cv2.getRotationMatrix2D(
                        (ir_norm.shape[1]/2, ir_norm.shape[0]/2), angle, 1.0)
                    
                    processed_pairs.append(ImagePair(
                        ir_image=cv2.warpAffine(ir_norm, matrix, ir_norm.shape[::-1]),
                        segmentation=cv2.warpAffine(seg_norm, matrix, seg_norm.shape[::-1]),
                        scene_image=cv2.warpAffine(pair.scene_image, matrix, 
                                                 pair.scene_image.shape[:2][::-1]),
                        metadata={**pair.metadata, 'augmentation': f'rotation_{angle}'},
                        timestamp=pair.timestamp,
                        target_type=pair.target_type
                    ))
        
        return processed_pairs
    
    def save_processed_dataset(self,
                             image_pairs: List[ImagePair],
                             format: str = 'hdf5') -> None:
        """
        Save processed dataset in specified format.
        
        Args:
            image_pairs: List of processed ImagePair objects
            format: Output format ('hdf5' or 'files')
        """
        if format == 'hdf5':
            # Save as single HDF5 file
            with h5py.File(self.output_dir / 'synthetic_ir_dataset.h5', 'w') as f:
                # Create datasets
                ir_dset = f.create_dataset('ir_images', 
                                         (len(image_pairs),) + image_pairs[0].ir_image.shape,
                                         dtype=np.float32)
                seg_dset = f.create_dataset('segmentation',
                                          (len(image_pairs),) + image_pairs[0].segmentation.shape,
                                          dtype=np.float32)
                scene_dset = f.create_dataset('scene_images',
                                            (len(image_pairs),) + image_pairs[0].scene_image.shape,
                                            dtype=np.uint8)
                
                # Store images
                for i, pair in enumerate(tqdm(image_pairs)):
                    ir_dset[i] = pair.ir_image
                    seg_dset[i] = pair.segmentation
                    scene_dset[i] = pair.scene_image
                
                # Store metadata
                metadata_group = f.create_group('metadata')
                for i, pair in enumerate(image_pairs):
                    metadata_group.create_dataset(
                        f'metadata_{i}',
                        data=json.dumps(pair.metadata)
                    )
                
                # Store normalization parameters
                f.create_dataset('norm_params',
                               data=json.dumps(self.norm_params))
                
        else:  # Save as individual files
            for i, pair in enumerate(tqdm(image_pairs)):
                base_path = self.output_dir / f"sample_{i:05d}"
                
                # Save images
                np.save(base_path / "ir.npy", pair.ir_image)
                np.save(base_path / "segmentation.npy", pair.segmentation)
                cv2.imwrite(str(base_path / "scene.png"), pair.scene_image)
                
                # Save metadata
                with open(base_path / "metadata.json", 'w') as f:
                    json.dump(pair.metadata, f, indent=2)
    
    def generate_dataset_statistics(self, image_pairs: List[ImagePair]) -> pd.DataFrame:
        """
        Generate statistical analysis of the dataset.
        
        Returns:
            DataFrame containing dataset statistics
        """
        stats = []
        
        for pair in image_pairs:
            stats.append({
                'target_type': pair.target_type,
                'timestamp': pair.timestamp,
                'ir_mean': np.mean(pair.ir_image),
                'ir_std': np.std(pair.ir_image),
                'ir_min': np.min(pair.ir_image),
                'ir_max': np.max(pair.ir_image),
                'seg_unique_values': len(np.unique(pair.segmentation)),
                'temperature': pair.metadata.get('target_temperature'),
                'capture_angle': pair.metadata.get('rotation')[1],  # yaw angle
                'time_of_day': pair.metadata.get('time_of_day', 'unknown')
            })
        
        df = pd.DataFrame(stats)
        
        # Save statistics
        df.to_csv(self.output_dir / 'dataset_statistics.csv', index=False)
        
        return df

def main():
    """Example usage of the DataPreprocessor."""
    processor = DataPreprocessor('raw_data', 'processed_data')
    
    # Load dataset
    print("Loading dataset...")
    image_pairs = processor.load_dataset()
    
    # Calculate normalization parameters
    print("Calculating normalization parameters...")
    processor.calculate_normalization_params(image_pairs)
    
    # Preprocess images
    print("Preprocessing images...")
    processed_pairs = processor.preprocess_images(image_pairs)
    
    # Save processed dataset
    print("Saving processed dataset...")
    processor.save_processed_dataset(processed_pairs)
    
    # Generate statistics
    print("Generating dataset statistics...")
    stats = processor.generate_dataset_statistics(processed_pairs)
    print("\nDataset Statistics:")
    print(stats.describe())

if __name__ == "__main__":
    main()
