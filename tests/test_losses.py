 import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import unittest
from losses.sir_losses import SIRLoss
from models.discriminator import Discriminator

class TestSIRLoss(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.channels = 3
        self.height = 256
        self.width = 256
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sir_loss = SIRLoss()
        self.discriminator = Discriminator().to(self.device)

    def test_adversarial_loss(self):
        real = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        fake = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        
        # Test generator loss
        g_loss = self.sir_loss.adversarial_loss(self.discriminator, real, fake, is_generator=True)
        self.assertIsInstance(g_loss, torch.Tensor)
        
        # Test discriminator loss
        d_loss = self.sir_loss.adversarial_loss(self.discriminator, real, fake, is_generator=False)
        self.assertIsInstance(d_loss, torch.Tensor)

    def test_cycle_consistency_loss(self):
        original = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        reconstructed = original.clone() + 0.1 * torch.randn_like(original)
        
        loss = self.sir_loss.cycle_consistency_loss(original, reconstructed)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.item() > 0)  # Loss should be positive

    def test_infrared_loss(self):
        img1 = torch.rand(self.batch_size, self.channels, self.height, self.width).to(self.device)
        img2 = torch.rand(self.batch_size, self.channels, self.height, self.width).to(self.device)
        
        loss = self.sir_loss.infrared_loss(img1, img2)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.item() >= 0)  # Loss should be non-negative

    def test_structure_loss(self):
        img1 = torch.rand(self.batch_size, self.channels, self.height, self.width).to(self.device)
        img2 = torch.rand(self.batch_size, self.channels, self.height, self.width).to(self.device)
        
        loss = self.sir_loss.structure_loss(img1, img2)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(0 <= loss.item() <= 1)  # Loss should be between 0 and 1

def main():
    unittest.main(verbosity=2)

if __name__ == '__main__':
    main()
