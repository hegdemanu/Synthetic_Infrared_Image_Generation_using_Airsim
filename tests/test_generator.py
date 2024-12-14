import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import unittest
from models.generator import Generator
import matplotlib.pyplot as plt
import numpy as np

class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.channels = 3
        self.height = 256
        self.width = 256
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = Generator().to(self.device)

    def test_model_architecture(self):
        total_params = sum(p.numel() for p in self.generator.parameters())
        print(f"\nTotal parameters: {total_params:,}")
        
        # Test model properties
        self.assertTrue(hasattr(self.generator, 'initial'), "Generator missing initial layer")
        self.assertTrue(hasattr(self.generator, 'resnet_blocks'), "Generator missing resnet blocks")
        self.assertTrue(hasattr(self.generator, 'final'), "Generator missing final layer")

    def test_forward_pass(self):
        # Create random input tensor
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = self.generator(x)
        
        # Test output shape
        expected_shape = (self.batch_size, self.channels, self.height, self.width)
        self.assertEqual(output.shape, expected_shape, 
                        f"Expected output shape {expected_shape}, got {output.shape}")
        
        # Test output range (should be between -1 and 1 due to tanh)
        self.assertTrue(torch.all(output >= -1) and torch.all(output <= 1),
                       "Output values should be in range [-1, 1]")

    def test_initialization(self):
        # Test different initialization methods
        init_types = ['normal', 'xavier', 'kaiming']
        for init_type in init_types:
            self.generator.init_weights(init_type=init_type)
            
            # Check if weights are properly initialized
            for name, param in self.generator.named_parameters():
                if 'weight' in name:
                    self.assertFalse(torch.all(param == 0), 
                                   f"Weights not properly initialized for {init_type}")

    def visualize_output(self):
        # Create a sample input
        x = torch.randn(1, self.channels, self.height, self.width).to(self.device)
        
        # Generate output
        with torch.no_grad():
            output = self.generator(x)
        
        # Convert tensors to numpy arrays for visualization
        input_img = x[0].cpu().numpy().transpose(1, 2, 0)
        output_img = output[0].cpu().numpy().transpose(1, 2, 0)
        
        # Normalize to [0, 1] for visualization
        input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())
        output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min())
        
        # Create visualization
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(input_img)
        plt.title('Input Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(output_img)
        plt.title('Generated Output')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('generator_test_output.png')
        plt.close()

def main():
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGenerator)
    
    # Run tests
    print("\nRunning Generator tests...")
    unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Visualize sample output
    print("\nGenerating visualization...")
    test = TestGenerator()
    test.setUp()
    test.visualize_output()
    print("Visualization saved as 'generator_test_output.png'")

if __name__ == '__main__':
    main()
