import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import confusion_matrix

class ValidationMetrics:
    """Metrics for evaluating IR image refinement quality."""
    def __init__(self, device):
        self.device = device
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def compute_psnr(self, img1, img2):
        """Compute Peak Signal-to-Noise Ratio."""
        mse = self.mse_loss(img1, img2)
        return 10 * torch.log10(1 / mse)
    
    def compute_ssim(self, img1, img2):
        """Compute Structural Similarity Index."""
        img1_np = img1.cpu().numpy().transpose(0, 2, 3, 1)
        img2_np = img2.cpu().numpy().transpose(0, 2, 3, 1)
        
        ssim_values = []
        for i in range(img1_np.shape[0]):
            ssim_val = ssim(img1_np[i], img2_np[i], multichannel=True)
            ssim_values.append(ssim_val)
        
        return np.mean(ssim_values)
    
    def compute_thermal_consistency(self, img1, img2):
        """Compute thermal distribution consistency between images."""
        def get_thermal_hist(img):
            img_gray = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]
            hist = torch.histc(img_gray, bins=256, min=-1, max=1)
            return hist / hist.sum()
        
        hist1 = get_thermal_hist(img1)
        hist2 = get_thermal_hist(img2)
        return -torch.sum(hist1 * torch.log(hist2 + 1e-10))
    
    def compute_all_metrics(self, real_images, generated_images):
        """Compute all validation metrics."""
        return {
            'psnr': self.compute_psnr(real_images, generated_images).item(),
            'ssim': self.compute_ssim(real_images, generated_images),
            'thermal_consistency': self.compute_thermal_consistency(real_images, generated_images).item()
        }
