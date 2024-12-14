import numpy as np
import cv2
import json
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
import xml.etree.ElementTree as ET
from datetime import datetime
import hashlib
from tqdm import tqdm
import logging
import shutil

class DataFormatConverter:
    """Handles conversion between different dataset formats."""
    
    def __init__(self, working_dir: Union[str, Path]):
        """
        Initialize converter.
        
        Args:
            working_dir: Working directory for temporary files
        """
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def coco_to_pascal_voc(self, 
                          coco_path: Union[str, Path],
                          output_path: Union[str, Path]) -> None:
        """
        Convert COCO format to Pascal VOC format.
        
        Args:
            coco_path: Path to COCO dataset
            output_path: Output path for Pascal VOC dataset
        """
        coco_path = Path(coco_path)
        output_path = Path(output_path)
        
        # Create VOC directory structure
        voc_dir = output_path / "VOC2024"
        for subdir in ['Annotations', 'ImageSets', 'JPEGImages']:
            (voc_dir / subdir).mkdir(parents=True, exist_ok=True)
            
        # Load COCO annotations
        with open(coco_path / "annotations.json", 'r') as f:
            coco_data = json.load(f)
            
        # Create category mapping
        cat_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Process each image
        for img in tqdm(coco_data['images'], desc="Converting to Pascal VOC"):
            # Create XML annotation
            root = ET.Element('annotation')
            ET.SubElement(root, 'folder').text = 'VOC2024'
            ET.SubElement(root, 'filename').text = img['file_name']
            
            size = ET.SubElement(root, 'size')
            ET.SubElement(size, 'width').text = str(img['width'])
            ET.SubElement(size, 'height').text = str(img['height'])
            ET.SubElement(size, 'depth').text = '3'
            
            # Add objects
            img_anns = [ann for ann in coco_data['annotations'] 
                       if ann['image_id'] == img['id']]
            
            for ann in img_anns:
                obj = ET.SubElement(root, 'object')
                ET.SubElement(obj, 'name').text = cat_map[ann['category_id']]
                bbox = ann['bbox']
                bndbox = ET.SubElement(obj, 'bndbox')
                ET.SubElement(bndbox, 'xmin').text = str(int(bbox[0]))
                ET.SubElement(bndbox, 'ymin').text = str(int(bbox[1]))
                ET.SubElement(bndbox, 'xmax').text = str(int(bbox[0] + bbox[2]))
                ET.SubElement(bndbox, 'ymax').text = str(int(bbox[1] + bbox[3]))
                
            # Save XML
            tree = ET.ElementTree(root)
            xml_path = voc_dir / 'Annotations' / f"{img['file_name'].split('.')[0]}.xml"
            tree.write(xml_path)
            
            # Copy image
            shutil.copy(
                coco_path / img['file_name'],
                voc_dir / 'JPEGImages' / img['file_name']
            )
            
    def yolo_to_coco(self,
                     yolo_path: Union[str, Path],
                     output_path: Union[str, Path]) -> None:
        """
        Convert YOLO format to COCO format.
        
        Args:
            yolo_path: Path to YOLO dataset
            output_path: Output path for COCO dataset
        """
        yolo_path = Path(yolo_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Read class names
        with open(yolo_path / 'classes.txt', 'r') as f:
            classes = f.read().splitlines()
            
        # Create COCO structure
        coco_dict = {
            "info": {
                "description": "Converted from YOLO format",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "images": [],
            "annotations": [],
            "categories": [
                {"id": i+1, "name": name, "supercategory": "object"}
                for i, name in enumerate(classes)
            ]
        }
        
        annotation_id = 1
        
        # Process images and labels
        for img_path in tqdm(list(yolo_path.glob('images/*.jpg')), desc="Converting to COCO"):
            img = cv2.imread(str(img_path))
            height, width = img.shape[:2]
            
            # Add image info
            image_id = len(coco_dict["images"])
            coco_dict["images"].append({
                "id": image_id,
                "file_name": img_path.name,
                "width": width,
                "height": height
            })
            
            # Process label file
            label_path = yolo_path / 'labels' / f"{img_path.stem}.txt"
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        class_id, x_center, y_center, w, h = map(float, line.strip().split())
                        
                        # Convert YOLO to COCO format
                        x = (x_center - w/2) * width
                        y = (y_center - h/2) * height
                        w = w * width
                        h = h * height
                        
                        coco_dict["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": int(class_id) + 1,
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": 0
                        })
                        annotation_id += 1
                        
        # Save COCO annotations
        with open(output_path / "annotations.json", 'w') as f:
            json.dump(coco_dict, f, indent=2)
            
        # Copy images
        (output_path / "images").mkdir(exist_ok=True)
        for img_path in yolo_path.glob('images/*'):
            shutil.copy(img_path, output_path / "images" / img_path.name)

class DatasetValidator:
    """Validates dataset integrity and format correctness."""
    
    def __init__(self):
        """Initialize validator."""
        self.logger = logging.getLogger(__name__)
        
    def validate_coco_format(self, coco_path: Union[str, Path]) -> bool:
        """
        Validate COCO dataset format.
        
        Args:
            coco_path: Path to COCO dataset
            
        Returns:
            True if valid, False otherwise
        """
        try:
            coco_path = Path(coco_path)
            
            # Check annotations file
            ann_path = coco_path / "annotations.json"
            if not ann_path.exists():
                self.logger.error("Annotations file not found")
                return False
                
            with open(ann_path, 'r') as f:
                coco_data = json.load(f)
                
            # Validate required fields
            required_fields = ['images', 'annotations', 'categories']
            if not all(field in coco_data for field in required_fields):
                self.logger.error("Missing required fields in annotations")
                return False
                
            # Validate image files
            for img in tqdm(coco_data['images'], desc="Validating images"):
                img_path = coco_path / img['file_name']
                if not img_path.exists():
                    self.logger.error(f"Image not found: {img['file_name']}")
                    return False
                    
                # Validate image dimensions
                actual_img = cv2.imread(str(img_path))
                if actual_img.shape[:2] != (img['height'], img['width']):
                    self.logger.error(f"Image dimensions mismatch: {img['file_name']}")
                    return False
                    
            # Validate annotations
            img_ids = {img['id'] for img in coco_data['images']}
            cat_ids = {cat['id'] for cat in coco_data['categories']}
            
            for ann in coco_data['annotations']:
                if ann['image_id'] not in img_ids:
                    self.logger.error(f"Invalid image_id in annotation: {ann['id']}")
                    return False
                if ann['category_id'] not in cat_ids:
                    self.logger.error(f"Invalid category_id in annotation: {ann['id']}")
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return False
            
    def validate_yolo_format(self, yolo_path: Union[str, Path]) -> bool:
        """
        Validate YOLO dataset format.
        
        Args:
            yolo_path: Path to YOLO dataset
            
        Returns:
            True if valid, False otherwise
        """
        try:
            yolo_path = Path(yolo_path)
            
            # Check directory structure
            required_dirs = ['images', 'labels']
            if not all((yolo_path / d).exists() for d in required_dirs):
                self.logger.error("Missing required directories")
                return False
                
            # Check classes file
            if not (yolo_path / 'classes.txt').exists():
                self.logger.error("classes.txt not found")
                return False
                
            # Read class names
            with open(yolo_path / 'classes.txt', 'r') as f:
                classes = f.read().splitlines()
            num_classes = len(classes)
            
            # Validate label files
            for img_path in tqdm(list(yolo_path.glob('images/*')), desc="Validating YOLO format"):
                label_path = yolo_path / 'labels' / f"{img_path.stem}.txt"
                
                # Check label file exists
                if not label_path.exists():
                    self.logger.warning(f"Missing label file for {img_path.name}")
                    continue
                    
                # Validate label format
                img = cv2.imread(str(img_path))
                if img is None:
                    self.logger.error(f"Invalid image: {img_path.name}")
                    return False
                    
                with open(label_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            parts = line.strip().split()
                            if len(parts) != 5:
                                self.logger.error(
                                    f"Invalid format in {label_path.name} line {line_num}")
                                return False
                                
                            class_id = int(parts[0])
                            if class_id >= num_classes:
                                self.logger.error(
                                    f"Invalid class ID in {label_path.name} line {line_num}")
                                return False
                                
                            # Validate bbox coordinates
                            x_center, y_center, w, h = map(float, parts[1:])
                            if not all(0 <= v <= 1 for v in [x_center, y_center, w, h]):
                                self.logger.error(
                                    f"Invalid bbox coordinates in {label_path.name} line {line_num}")
                                return False
                                
                        except ValueError:
                            self.logger.error(
                                f"Invalid number format in {label_path.name} line {line_num}")
                            return False
                            
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return False
            
    def compute_dataset_checksum(self, dataset_path: Union[str, Path]) -> str:
        """
        Compute checksum for dataset files.
        
        Args:
            dataset_path: Path to dataset
            
        Returns:
            Dataset checksum
        """
        dataset_path = Path(dataset_path)
        checksums = []
        
        for file_path in tqdm(sorted(dataset_path.rglob('*')), 
                            desc="Computing checksum"):
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                checksums.append(f"{file_path.name}:{file_hash}")
                
        return hashlib.sha256('\n'.join(checksums).encode()).hexdigest()

def main():
    """Example usage of conversion and validation utilities."""
    # Initialize components
    converter = DataFormatConverter("working_dir")
    validator = DatasetValidator()
    
    # Example conversion
    converter.yolo_to_coco("yolo_dataset", "coco_output")
    converter.coco_to_pascal_voc("coco_dataset", "voc_output")
    
    # Validate converted datasets
    if validator.validate_coco_format("coco_output"):
        print("COCO format validation successful")
    
    if validator.validate_yolo_format("yolo_dataset"):
        print("YOLO format validation successful")
    
    # Compute dataset checksum
    checksum = validator.compute_dataset_checksum("coco_output")
    print(f"Dataset checksum: {checksum}")

if __name__ == "__main__":
    main()
