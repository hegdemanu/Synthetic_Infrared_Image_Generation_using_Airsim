# losses/

import torch
import torch.nn as nn
import torch.nn.functional as F

class SIRLoss:
    """Combined loss for SIR-GAN."""
    def __init__(self, lambda_adv=1, lambda_cycle=5, lambda_ir=1, lambda_struct=1):
        self.lambda_adv = lambda_adv
        self.lambda_cycle = lambda_cycle
        self.lambda_ir = lambda_ir
        self.lambda_struct = lambda_struct
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def adversarial_loss(self, pred, target_is_real):
        """Adversarial loss."""
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.mse_loss(pred, target)

    def cycle_consistency_loss(self, real, cycled):
        """Cycle consistency loss."""
        return self.l1_loss(cycled, real)

    def infrared_loss(self, pred, target, phi_min=0, phi_max=1, r=0.5):
        """Infrared radiation loss."""
        # Convert to radiation values using the provided formula
        pred_rad = phi_min + (pred/255.0 - r) * (phi_max - phi_min) / (1 - r)
        target_rad = phi_min + (target/255.0 - r) * (phi_max - phi_min) / (1 - r)
        return self.l1_loss(pred_rad, target_rad)

    def structure_loss(self, pred, target):
        """Structure loss using gradient correlation."""
        # Compute gradients
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        # Compute normalized cross correlation
        def normalized_cross_correlation(x, y):
            mean_x = torch.mean(x, dim=[2, 3], keepdim=True)
            mean_y = torch.mean(y, dim=[2, 3], keepdim=True)
            x_centered = x - mean_x
            y_centered = y - mean_y
            
            numerator = torch.sum(x_centered * y_centered, dim=[2, 3])
            denominator = torch.sqrt(torch.sum(x_centered**2, dim=[2, 3]) * 
                                   torch.sum(y_centered**2, dim=[2, 3]))
            return numerator / (denominator + 1e-8)

        nc_x = normalized_cross_correlation(pred_dx, target_dx)
        nc_y = normalized_cross_correlation(pred_dy, target_dy)
        
        return 1 - 0.5 * (torch.mean(nc_x) + torch.mean(nc_y))

    def __call__(self, real_A, real_B, fake_A, fake_B, cycled_A, cycled_B, 
                 disc_A_fake, disc_A_real, disc_B_fake, disc_B_real):
        """Calculate total loss."""
        # Adversarial loss
        g_loss_A = self.adversarial_loss(disc_A_fake, True)
        g_loss_B = self.adversarial_loss(disc_B_fake, True)
        g_loss = self.lambda_adv * (g_loss_A + g_loss_B)

        # Cycle consistency loss
        cycle_loss = self.lambda_cycle * (
            self.cycle_consistency_loss(real_A, cycled_A) +
            self.cycle_consistency_loss(real_B, cycled_B)
        )

        # Infrared loss
        ir_loss = self.lambda_ir * (
            self.infrared_loss(fake_A, real_A) +
            self.infrared_loss(fake_B, real_B)
        )

        # Structure loss
        struct_loss = self.lambda_struct * (
            self.structure_loss(fake_A, real_A) +
            self.structure_loss(fake_B, real_B)
        )

        # Total loss
        total_loss = g_loss + cycle_loss + ir_loss + struct_loss

        return {
            'total_loss': total_loss,
            'adv_loss': g_loss,
            'cycle_loss': cycle_loss,
            'ir_loss': ir_loss,
            'struct_loss': struct_loss
        }
