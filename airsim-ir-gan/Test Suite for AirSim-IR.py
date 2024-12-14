import unittest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import shutil
import json

from environment.african_environment import AfricanEnvironment
from thermal.thermal_model import ThermalModel
from capture.ir_capture import IRImageCapture
from preprocessing.data_preprocessor import DataPreprocessor, ImagePair

class TestAirSimIR(unittest.TestCase):
    """Test suite for AirSim-IR synthetic data generation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.output_dir = Path(cls.temp_dir) / "test_output"
        cls.output_dir.mkdir()
        
        # Initialize components
        cls.environment = AfricanEnvironment()
        cls.thermal = ThermalModel()
        cls.capture = IRImageCapture()
        cls.preprocessor = DataPreprocessor(cls.output_dir)
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.temp_dir)
        
    def test_environment_setup(self):
        """Test environment configuration."""
        # Test time of day setting
        self.environment.set_time_of_day("12:00")
        self.assertEqual(self.environment.current_time.hour, 12)
        
        # Test weather setting
        self.environment.set_weather(temperature=25.0, humidity=60.0)
        self.assertEqual(self.environment.weather_state['temperature'], 25.0)
        self.assertEqual(self.environment.weather_state['humidity'], 60.0)
        
    def test_thermal_calculations(self):
        """Test thermal modeling calculations."""
        # Test radiation calculation
        radiance, integrated = self.thermal.calculate_radiation(
            temperature=25.0,
            emissivity=0.95
        )
        self.assertTrue(isinstance(radiance, np.ndarray))
        self.assertTrue(isinstance(integrated, float))
        self.assertTrue(np.all(radiance >= 0))
        
        # Test temperature distribution
        temp = self.thermal.calculate_temperature_distribution(
            ambient_temp=25.0,
            material='elephant',
            solar_radiation=800
        )
        self.assertTrue(isinstance(temp, float))
        self.assertTrue(temp > 25.0)  # Should be warmer due to solar radiation
        
    def test_image_capture(self):
        """Test image capture functionality."""
        # Test single capture
        self.capture.set_camera_pose((0, 0, -10), (0, 0, 0))
        images = self.capture.capture_images()
        
        self.assertIn('ir', images)
        self.assertIn('segmentation', images)
        self.assertIn('scene', images)
        
        # Verify image properties
        self.assertEqual(images['ir'].dtype, np.uint8)
        self.assertEqual(len(images['ir'].shape), 3)
        
    def test_multi_angle_capture(self):
        """Test multi-angle capture system."""
        metadata = self.capture.capture_multi_angle(
            target_position=(0, 0, 0),
            output_dir=str(self.output_dir),
            num_angles=5
        )
        
        self.assertEqual(len(metadata), 5)
        self.assertTrue(all('position' in meta for meta in metadata))
        self.assertTrue(all('rotation' in meta for meta in metadata))
        
    def test_preprocessing(self):
        """Test data preprocessing pipeline."""
        # Create some test data
        test_pairs = [
            ImagePair(
                ir_image=np.random.randint(0, 255, (64, 64), dtype=np.uint8),
                segmentation=np.random.randint(0, 10, (64, 64), dtype=np.uint8),
      scene_image=np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
                metadata={'test': 'metadata1'},
                timestamp=1234567890.0,
                target_type='elephant'
            ),
            ImagePair(
                ir_image=np.random.randint(0, 255, (64, 64), dtype=np.uint8),
                segmentation=np.random.randint(0, 10, (64, 64), dtype=np.uint8),
                scene_image=np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
                metadata={'test': 'metadata2'},
                timestamp=1234567891.0,
                target_type='zebra'
            )
        ]
        
        # Test normalization parameter calculation
        self.preprocessor.calculate_normalization_params(test_pairs)
        self.assertIsNotNone(self.preprocessor.norm_params['ir_mean'])
        self.assertIsNotNone(self.preprocessor.norm_params['ir_std'])
        
        # Test preprocessing with normalization
        processed_pairs = self.preprocessor.preprocess_images(
            test_pairs,
            normalize=True,
            augment=False
        )
        self.assertEqual(len(processed_pairs), len(test_pairs))
        
        # Test preprocessing with augmentation
        augmented_pairs = self.preprocessor.preprocess_images(
            test_pairs,
            normalize=True,
            augment=True
        )
        self.assertEqual(len(augmented_pairs), len(test_pairs) * 5)  # Original + 4 augmentations
        
    def test_dataset_statistics(self):
        """Test dataset statistics generation."""
        # Create test data
        test_pairs = [
            ImagePair(
                ir_image=np.random.randint(0, 255, (64, 64), dtype=np.uint8),
                segmentation=np.random.randint(0, 10, (64, 64), dtype=np.uint8),
                scene_image=np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
                metadata={
                    'target_temperature': 35.0,
                    'time_of_day': '12:00',
                    'rotation': [0, 45, 0]
                },
                timestamp=1234567890.0,
                target_type='elephant'
            )
        ]
        
        # Generate statistics
        stats = self.preprocessor.generate_dataset_statistics(test_pairs)
        
        # Verify statistics
        self.assertEqual(len(stats), len(test_pairs))
        self.assertTrue('target_type' in stats.columns)
        self.assertTrue('ir_mean' in stats.columns)
        self.assertTrue('temperature' in stats.columns)
        
    def test_data_saving(self):
        """Test dataset saving functionality."""
        # Create test data
        test_pairs = [
            ImagePair(
                ir_image=np.random.randint(0, 255, (64, 64), dtype=np.uint8),
                segmentation=np.random.randint(0, 10, (64, 64), dtype=np.uint8),
                scene_image=np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
                metadata={'test': 'metadata'},
                timestamp=1234567890.0,
                target_type='elephant'
            )
        ]
        
        # Test HDF5 saving
        self.preprocessor.save_processed_dataset(test_pairs, format='hdf5')
        self.assertTrue((self.preprocessor.output_dir / 'synthetic_ir_dataset.h5').exists())
        
        # Test individual file saving
        self.preprocessor.save_processed_dataset(test_pairs, format='files')
        self.assertTrue((self.preprocessor.output_dir / 'sample_00000').exists())
        
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        
        # Test invalid camera pose
        with self.assertRaises(Exception):
            self.capture.set_camera_pose((float('inf'), 0, 0), (0, 0, 0))
            
        # Test invalid time format
        with self.assertRaises(ValueError):
            self.environment.set_time_of_day("25:00")
            
        # Test invalid temperature values
        with self.assertRaises(ValueError):
            self.thermal.calculate_temperature_distribution(
                ambient_temp=float('inf'),
                material='elephant'
            )
            
    def test_segmentation_map(self):
        """Test segmentation map generation."""
        # Test object temperature mapping
        object_temps = {
            'elephant': 35.0,
            'zebra': 38.0,
            'truck': 45.0
        }
        
        self.capture.create_segmentation_map(object_temps)
        
        # Capture a segmentation image
        images = self.capture.capture_images()
        seg_img = images['segmentation']
        
        # Verify segmentation image properties
        self.assertEqual(seg_img.dtype, np.uint8)
        self.assertEqual(len(seg_img.shape), 3)
        
    def test_thermal_model_materials(self):
        """Test thermal model material properties."""
        # Test all material properties
        for material in self.thermal.materials.keys():
            props = self.thermal.materials[material]
            self.assertTrue(0 <= props.emissivity <= 1)
            self.assertTrue(props.specific_heat > 0)
            self.assertTrue(props.thermal_conductivity > 0)
            self.assertTrue(props.density > 0)

if __name__ == '__main__':
    unittest.main(verbosity=2)
