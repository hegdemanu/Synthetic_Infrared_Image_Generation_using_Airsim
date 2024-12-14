import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import unittest
from models.discriminator import Discriminator
import numpy as np

class TestDiscriminator(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.channels = 3
        self.height = 256
        self.width = 256
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.discriminator = Discriminator().to(self.device)

    def test_model_architecture(self):
        total_params = sum(p.numel() for p in self.discriminator.parameters())
        print(f"\nTotal parameters: {total_params:,}")
        
        # Test model properties
        self.assertTrue(hasattr(self.discriminator, 'initial'), "Missing initial layer")
        self.assertTrue(hasattr(self.discriminator, 'dilated1'), "Missing dilated1 layer")
        self.assertTrue(hasattr(self.discriminator, 'classifier'), "Missing classifier layer")

    def test_forward_pass(self):
        # Create random input tensor
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = self.discriminator(x)
        
        # Test output shape (PatchGAN output)
        self.assertEqual(len(output.shape), 4, "Output should be a 4D tensor")
        self.assertEqual(output.shape[1], 1, "Output should have 1 channel")
        
        # Test output range (should be between 0 and 1 due to sigmoid)
        self.assertTrue(torch.all(output >= 0) and torch.all(output <= 1),
                       "Output values should be in range [0, 1]")

    def test_receptive_field(self):
        # Test if dilated convolutions are working as expected
        x = torch.randn(1, self.channels, 32, 32).to(self.device)
        
        # Get intermediate outputs
        with torch.no_grad():
            initial = self.discriminator.initial(x)
            dilated1 = self.discriminator.dilated1(initial)
            dilated2 = self.discriminator.dilated2(dilated1)
            dilated3 = self.discriminator.dilated3(dilated2)
        
        # Check if feature maps have expected shapes
        self.assertNotEqual(dilated1.shape, dilated2.shape, 
                          "Dilated convolutions should produce different spatial dimensions")

    def test_initialization(self):
        # Test different initialization methods
        init_types = ['normal', 'xavier', 'kaiming']
        for init_type in init_types:
            self.discriminator.init_weights(init_type=init_type)
            
            # Check if weights are properly initialized
            for name, param in self.discriminator.named_parameters():
                if 'weight' in name:
                    self.assertFalse(torch.all(param == 0), 
                                   f"Weights not properly initialized for {init_type}")

def main():
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDiscriminator)
    
    # Run tests
    print("\nRunning Discriminator tests...")
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    main()
