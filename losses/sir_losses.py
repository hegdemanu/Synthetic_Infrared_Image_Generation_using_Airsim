import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SIRLoss:
    """Implementation of all loss components for SIR-GAN."""
    def __init__(self, lambda_cycle=5.0, lambda_refine=2.0):
        self.lambda_cycle = lambda_cycle  # Weight for cycle consistency loss
        self.lambda_refine = lambda_refine  # Weight for refinement loss
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def adversarial_loss(self, discriminator, real_images, fake_images, is_generator=True):
        """
        Calculate the adversarial loss for either generator or discriminator.
        For generator: Minimizes log(1 - D(G(x)))
        For discriminator: Maximizes log(D(x)) + log(1 - D(G(x)))
        """
        if is_generator:
            # Generator tries to make discriminator output 1 for fake images
            d_fake = discriminator(fake_images)
            loss = self.mse_loss(d_fake, torch.ones_like(d_fake))
            return loss
        else:
            # Discriminator tries to output 1 for real and 0 for fake
            d_real = discriminator(real_images)
            d_fake = discriminator(fake_images.detach())  # Detach to avoid training generator
            real_loss = self.mse_loss(d_real, torch.ones_like(d_real))
            fake_loss = self.mse_loss(d_fake, torch.zeros_like(d_fake))
            return (real_loss + fake_loss) * 0.5

    def cycle_consistency_loss(self, original_images, reconstructed_images):
        """
        Calculate cycle consistency loss to ensure bidirectional mapping consistency.
        L1 distance between original and reconstructed images.
        """
        return self.l1_loss(reconstructed_images, original_images)

    def infrared_loss(self, generated_images, target_images, r=0.5):
        """
        Calculate infrared loss based on radiation intensity distribution.
        Φ = Φmin + (g/255 - r) × (Φmax - Φmin)/(1 - r)
        """
        # Convert images to radiation values using the provided formula
        def to_radiation(images):
            # Normalize images to [0, 1]
            normalized = (images + 1) / 2  # Assuming images are in [-1, 1]
            
            # Constants for radiation calculation (can be adjusted based on specific IR camera)
            phi_min, phi_max = 0.0, 1.0
            
            # Calculate radiation values
            radiation = phi_min + (normalized - r) * (phi_max - phi_min) / (1 - r)
            return radiation

        generated_radiation = to_radiation(generated_images)
        target_radiation = to_radiation(target_images)
        
        return self.l1_loss(generated_radiation, target_radiation)

    def structure_loss(self, generated_images, target_images):
        """
        Calculate structure loss using gradient correlation.
        Preserves structural information while allowing thermal variations.
        """
        def gradient_correlation(x, y):
            # Calculate gradients
            sobelx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
            sobely = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                dtype=torch.float32, device=x.device).view(1, 1, 3, 3)

            # Expand sobel filters for all channels
            sobelx = sobelx.repeat(x.size(1), 1, 1, 1)
            sobely = sobely.repeat(x.size(1), 1, 1, 1)

            # Calculate gradients for both images
            x_gradx = F.conv2d(x, sobelx, padding=1, groups=x.size(1))
            x_grady = F.conv2d(x, sobely, padding=1, groups=x.size(1))
            y_gradx = F.conv2d(y, sobelx, padding=1, groups=y.size(1))
            y_grady = F.conv2d(y, sobely, padding=1, groups=y.size(1))

            # Calculate normalized cross correlation
            def normalized_cross_correlation(a, b):
                a_mean = torch.mean(a, dim=[2, 3], keepdim=True)
                b_mean = torch.mean(b, dim=[2, 3], keepdim=True)
                a_centered = a - a_mean
                b_centered = b - b_mean
                
                numerator = torch.sum(a_centered * b_centered, dim=[2, 3])
                denominator = torch.sqrt(torch.sum(a_centered**2, dim=[2, 3]) * 
                                      torch.sum(b_centered**2, dim=[2, 3]))
                
                return torch.mean(numerator / (denominator + 1e-6))

            gc_x = normalized_cross_correlation(x_gradx, y_gradx)
            gc_y = normalized_cross_correlation(x_grady, y_grady)
            
            return (gc_x + gc_y) / 2

        return 1 - gradient_correlation(generated_images, target_images)

    def refinement_loss(self, generated_images, target_images):
        """
        Combine infrared and structure losses into the refinement loss.
        """
        ir_loss = self.infrared_loss(generated_images, target_images)
        struct_loss = self.structure_loss(generated_images, target_images)
        return ir_loss + struct_loss

    def compute_total_loss(self, real_s, real_r, fake_s, fake_r, cycle_s, cycle_r,
                          disc_s, disc_r, is_generator=True):
        """
        Compute total loss combining all components:
        - Adversarial loss
        - Cycle consistency loss
        - Refinement loss (infrared + structure)
        """
        # Adversarial losses
        g_s2r_loss = self.adversarial_loss(disc_r, real_r, fake_r, is_generator)
        g_r2s_loss = self.adversarial_loss(disc_s, real_s, fake_s, is_generator)
        g_adv_loss = g_s2r_loss + g_r2s_loss

        if is_generator:
            # Cycle consistency losses
            cycle_s_loss = self.cycle_consistency_loss(real_s, cycle_s)
            cycle_r_loss = self.cycle_consistency_loss(real_r, cycle_r)
            cycle_loss = cycle_s_loss + cycle_r_loss

            # Refinement losses
            refine_s2r = self.refinement_loss(fake_r, real_r)
            refine_r2s = self.refinement_loss(fake_s, real_s)
            refine_loss = refine_s2r + refine_r2s

            # Combine all losses with weights
            total_loss = (g_adv_loss + 
                         self.lambda_cycle * cycle_loss + 
                         self.lambda_refine * refine_loss)
        else:
            total_loss = g_adv_loss

        return total_loss
