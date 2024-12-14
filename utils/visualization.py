import torch
import torchvision
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image

def denormalize(tensor):
    """Convert normalized image tensor back to displayable range."""
    return (tensor + 1) / 2

def save_sample_images(G_S2R, G_R2S, real_s, real_r, save_path):
    """Save a grid of sample images showing the translation results."""
    os.makedirs(save_path, exist_ok=True)
    
    with torch.no_grad():
        # Generate fake images
        fake_r = G_S2R(real_s)
        fake_s = G_R2S(real_r)
        
        # Generate reconstructed images
        cycle_s = G_R2S(fake_r)
        cycle_r = G_S2R(fake_s)
        
        # Create image grid
        img_tensor = torch.cat([
            denormalize(real_s),
            denormalize(fake_r),
            denormalize(cycle_s),
            denormalize(real_r),
            denormalize(fake_s),
            denormalize(cycle_r)
        ], dim=0)
        
        # Save grid image
        save_image(img_tensor, 
                  os.path.join(save_path, 'grid.png'),
                  nrow=3,
                  normalize=False)
        
def plot_training_progress(losses, save_path):
    """Plot and save training loss curves."""
    plt.figure(figsize=(10, 5))
    
    # Plot generator losses
    plt.subplot(1, 2, 1)
    plt.plot(losses['g_loss'], label='Generator Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Generator Training Progress')
    plt.legend()
    
    # Plot discriminator losses
    plt.subplot(1, 2, 2)
    plt.plot(losses['d_loss'], label='Discriminator Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Discriminator Training Progress')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_progress.png'))
    plt.close()
