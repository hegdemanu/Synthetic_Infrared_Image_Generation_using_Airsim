import numpy as np
import cv2
import h5py
import json
import pickle
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
import pandas as pd
from tqdm import tqdm
import shutil
import zipfile
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from dataclasses import dataclass, asdict

@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    num_workers: int = multiprocessing.cpu_count()
    batch_size: int = 32
    export_format: str = 'hdf5'
    compression: bool = True
    include_metadata: bool = True

class DataExporter:
    """Handles exporting synthetic IR data in various formats."""
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize data exporter.
        
        Args:
            output_dir: Base directory for exported data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_to_coco(self, 
                      image_pairs: List[Dict],
                      dataset_name: str) -> None:
        """
        Export dataset in COCO format.
        
        Args:
            image_pairs: List of image pairs with metadata
            dataset_name: Name of the dataset
        """
        coco_dir = self.output_dir / 'coco' / dataset_name
        coco_dir.mkdir(parents=True, exist_ok=True)
        
        # Create COCO structure
        coco_dict = {
            "info": {
                "description": "Synthetic IR Dataset",
                "url": "",
                "version": "1.0",
                "year": 2024,
                "contributor": "AirSim-IR",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Add categories
        category_map = {}
        for i, cat in enumerate(set(pair['target_type'] for pair in image_pairs)):
            category_map[cat] = i + 1
            coco_dict["categories"].append({
                "id": i + 1,
                "name": cat,
                "supercategory": "object"
            })
        
        # Process images and annotations
        annotation_id = 1
        for img_id, pair in enumerate(tqdm(image_pairs, desc="Exporting to COCO")):
            # Save IR image
            ir_path = coco_dir / f"ir_{img_id:06d}.png"
            cv2.imwrite(str(ir_path), pair['ir_image'])
            
            # Add image info
            coco_dict["images"].append({
                "id": img_id,
                "file_name": ir_path.name,
                "width": pair['ir_image'].shape[1],
                "height": pair['ir_image'].shape[0],
                "date_captured": pair['metadata']['timestamp']
            })
            
            # Add annotation
            if 'segmentation' in pair:
                for obj in pair['metadata'].get('objects', []):
                    coco_dict["annotations"].append({
                        "id": annotation_id,
                        "image_id": img_id,
                        "category_id": category_map[obj['category']],
                        "segmentation": obj['segmentation'],
                        "area": obj['area'],
                        "bbox": obj['bbox'],
                        "iscrowd": 0
                    })
                    annotation_id += 1
        
        # Save COCO JSON
        with open(coco_dir / "annotations.json", 'w') as f:
            json.dump(coco_dict, f)
            
    def export_to_tfrecord(self, 
                          image_pairs: List[Dict],
                          dataset_name: str) -> None:
        """
        Export dataset in TFRecord format.
        
        Args:
            image_pairs: List of image pairs with metadata
            dataset_name: Name of the dataset
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow is required for TFRecord export")
            
        tfrecord_dir = self.output_dir / 'tfrecord' / dataset_name
        tfrecord_dir.mkdir(parents=True, exist_ok=True)
        
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        
        def _float_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))
        
        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
        
        # Write TFRecords in chunks
        samples_per_file = 1000
        for chunk_idx in range(0, len(image_pairs), samples_per_file):
            chunk = image_pairs[chunk_idx:chunk_idx + samples_per_file]
            output_path = tfrecord_dir / f"data_{chunk_idx//samples_per_file:03d}.tfrecord"
            
            with tf.io.TFRecordWriter(str(output_path)) as writer:
                for pair in tqdm(chunk, desc=f"Writing TFRecord chunk {chunk_idx//samples_per_file}"):
                    feature = {
                        'ir_image': _bytes_feature(pair['ir_image'].tobytes()),
                        'segmentation': _bytes_feature(pair['segmentation'].tobytes()),
                        'height': _int64_feature([pair['ir_image'].shape[0]]),
                        'width': _int64_feature([pair['ir_image'].shape[1]]),
                        'target_type': _bytes_feature(pair['target_type'].encode()),
                        'temperature': _float_feature([pair['metadata']['temperature']])
                    }
                    
                    example = tf.train.Example(
                        features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
                    
    def export_to_yolo(self,
                      image_pairs: List[Dict],
                      dataset_name: str) -> None:
        """
        Export dataset in YOLO format.
        
        Args:
            image_pairs: List of image pairs with metadata
            dataset_name: Name of the dataset
        """
        yolo_dir = self.output_dir / 'yolo' / dataset_name
        yolo_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directories
        (yolo_dir / 'images').mkdir(exist_ok=True)
        (yolo_dir / 'labels').mkdir(exist_ok=True)
        
        # Create class mapping
        classes = sorted(set(pair['target_type'] for pair in image_pairs))
        class_map = {cls: idx for idx, cls in enumerate(classes)}
        
        # Save class names
        with open(yolo_dir / 'classes.txt', 'w') as f:
            f.write('\n'.join(classes))
            
        # Process images and labels
        for idx, pair in enumerate(tqdm(image_pairs, desc="Exporting to YOLO")):
            # Save image
            img_path = yolo_dir / 'images' / f"{idx:06d}.png"
            cv2.imwrite(str(img_path), pair['ir_image'])
            
            # Create YOLO label file
            label_path = yolo_dir / 'labels' / f"{idx:06d}.txt"
            with open(label_path, 'w') as f:
                for obj in pair['metadata'].get('objects', []):
                    # Convert bbox to YOLO format (x_center, y_center, width, height)
                    x, y, w, h = obj['bbox']
                    x_center = (x + w/2) / pair['ir_image'].shape[1]
                    y_center = (y + h/2) / pair['ir_image'].shape[0]
                    w = w / pair['ir_image'].shape[1]
                    h = h / pair['ir_image'].shape[0]
                    
                    # Write label
                    class_id = class_map[obj['category']]
                    f.write(f"{class_id} {x_center} {y_center} {w} {h}\n")

class BatchProcessor:
    """Handles batch processing of synthetic IR data."""
    
    def __init__(self, config: BatchConfig):
        """
        Initialize batch processor.
        
        Args:
            config: Batch processing configuration
        """
        self.config = config
        
    def process_batch(self,
                     batch: List[Dict],
                     processing_fn: callable) -> List[Dict]:
        """
        Process a batch of images.
        
        Args:
            batch: List of image pairs
            processing_fn: Function to apply to each image
            
        Returns:
            Processed batch
        """
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = [executor.submit(processing_fn, item) for item in batch]
            results = []
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing batch item: {e}")
                    
        return results
    
    def process_dataset(self,
                       dataset: List[Dict],
                       processing_fn: callable) -> List[Dict]:
        """
        Process entire dataset in batches.
        
        Args:
            dataset: List of image pairs
            processing_fn: Function to apply to each image
            
        Returns:
            Processed dataset
        """
        processed_dataset = []
        
        for i in range(0, len(dataset), self.config.batch_size):
            batch = dataset[i:i + self.config.batch_size]
            processed_batch = self.process_batch(batch, processing_fn)
            processed_dataset.extend(processed_batch)
            
        return processed_dataset
    
    def export_processed_data(self,
                            processed_data: List[Dict],
                            output_path: Union[str, Path]) -> None:
        """
        Export processed data in specified format.
        
        Args:
            processed_data: Processed dataset
            output_path: Path to save exported data
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.config.export_format == 'hdf5':
            with h5py.File(output_path / 'processed_data.h5', 'w') as f:
                for i, item in enumerate(processed_data):
                    group = f.create_group(f'sample_{i}')
                    for key, value in item.items():
                        if isinstance(value, np.ndarray):
                            group.create_dataset(key, data=value, 
                                              compression='gzip' if self.config.compression else None)
                        elif isinstance(value, dict) and self.config.include_metadata:
                            group.attrs[key] = json.dumps(value)
                            
        elif self.config.export_format == 'pickle':
            with open(output_path / 'processed_data.pkl', 'wb') as f:
                pickle.dump(processed_data, f)
                
        elif self.config.export_format == 'numpy':
            for i, item in enumerate(processed_data):
                np.save(output_path / f'sample_{i}.npy', item)

def example_preprocessing_fn(image_pair: Dict) -> Dict:
    """Example preprocessing function for batch processing."""
    # Apply some preprocessing
    processed = image_pair.copy()
    
    # Normalize IR image
    processed['ir_image'] = cv2.normalize(
        image_pair['ir_image'], None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply smoothing
    processed['ir_image'] = cv2.GaussianBlur(processed['ir_image'], (3, 3), 0)
    
    return processed

def main():
    """Example usage of export and batch processing."""
    # Create sample data
    sample_data = [
        {
            'ir_image': np.random.randint(0, 255, (64, 64), dtype=np.uint8),
            'segmentation': np.random.randint(0, 5, (64, 64), dtype=np.uint8),
            'target_type': 'elephant',
            'metadata': {
                'temperature': 35.0,
                'timestamp': '2024-01-01 12:00:00',
                'objects': [
                    {
                        'category': 'elephant',
                        'bbox': [10, 10, 40, 40],
                        'area': 1600,
                        'segmentation': [[10, 10, 50, 10, 50, 50, 10, 50]]
                    }
                ]
            }
        }
        for _ in range(10)
    ]
    
    # Initialize components
    config = BatchConfig(
        num_workers=4,
        batch_size=2,
        export_format='hdf5',
        compression=True
    )
    
    batch_processor = BatchProcessor(config)
    exporter = DataExporter("exported_data")
    
    # Process and export data
    processed_data = batch_processor.process_dataset(
        sample_data, example_preprocessing_fn)
    
    # Export in different formats
    exporter.export_to_coco(processed_data, "sample_dataset")
    exporter.export_to_yolo(processed_data, "sample_dataset")
    
    batch_processor.export_processed_data(
        processed_data, "processed_output")

if __name__ == "__main__":
    main()
