import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import os
from datetime import datetime

from models.generator import Generator
from models.discriminator import Discriminator
from losses.sir_losses import SIRLoss
from utils.dataset import IRDataset
from utils.visualization import save_sample_images
from utils.checkpoint import save_checkpoint, load_checkpoint

class SIRGANTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_logging()
        self.initialize_models()
        self.setup_optimizers()
        self.setup_dataloaders()
        self.setup_losses()

    def setup_logging(self):
        """Initialize logging configuration."""
        self.log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_models(self):
        """Initialize generator and discriminator models."""
        # Initialize generators
        self.G_S2R = Generator().to(self.device)
        self.G_R2S = Generator().to(self.device)
        
        # Initialize discriminators
        self.D_S = Discriminator().to(self.device)
        self.D_R = Discriminator().to(self.device)
        
        # Initialize weights
        for model in [self.G_S2R, self.G_R2S, self.D_S, self.D_R]:
            model.init_weights(init_type='normal', gain=0.02)

    def setup_optimizers(self):
        """Initialize optimizers with Nesterov-accelerated Adam."""
        beta1 = self.config.get('beta1', 0.975)
        beta2 = self.config.get('beta2', 0.999)
        lr = self.config.get('learning_rate', 0.0002)

        # Generator optimizers
        self.optimizer_G = optim.Adam(
            list(self.G_S2R.parameters()) + list(self.G_R2S.parameters()),
            lr=lr, betas=(beta1, beta2), nesterov=True
        )

        # Discriminator optimizers
        self.optimizer_D = optim.Adam(
            list(self.D_S.parameters()) + list(self.D_R.parameters()),
            lr=lr, betas=(beta1, beta2), nesterov=True
        )

    def setup_dataloaders(self):
        """Initialize data loaders for real and simulated IR images."""
        # Create datasets
        self.sim_dataset = IRDataset(
            self.config['sim_data_path'],
            transform=self.config.get('transforms', None)
        )
        self.real_dataset = IRDataset(
            self.config['real_data_path'],
            transform=self.config.get('transforms', None)
        )

        # Create data loaders
        self.sim_loader = DataLoader(
            self.sim_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4)
        )
        self.real_loader = DataLoader(
            self.real_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4)
        )

    def setup_losses(self):
        """Initialize loss functions."""
        self.sir_loss = SIRLoss(
            lambda_cycle=self.config.get('lambda_cycle', 5.0),
            lambda_refine=self.config.get('lambda_refine', 2.0)
        )

    def train_step(self, real_s, real_r):
        """Execute a single training step."""
        # Forward cycle: S -> R -> S
        fake_r = self.G_S2R(real_s)
        cycle_s = self.G_R2S(fake_r)
        
        # Backward cycle: R -> S -> R
        fake_s = self.G_R2S(real_r)
        cycle_r = self.G_S2R(fake_s)

        # Update Generators
        self.optimizer_G.zero_grad()
        g_loss = self.sir_loss.compute_total_loss(
            real_s, real_r, fake_s, fake_r, cycle_s, cycle_r,
            self.D_S, self.D_R, is_generator=True
        )
        g_loss.backward()
        self.optimizer_G.step()

        # Update Discriminators
        self.optimizer_D.zero_grad()
        d_loss = self.sir_loss.compute_total_loss(
            real_s, real_r, fake_s, fake_r, cycle_s, cycle_r,
            self.D_S, self.D_R, is_generator=False
        )
        d_loss.backward()
        self.optimizer_D.step()

        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item()
        }

    def train(self):
        """Execute the main training loop."""
        num_epochs = self.config['num_epochs']
        
        for epoch in range(num_epochs):
            self.logger.info(f'Starting epoch {epoch+1}/{num_epochs}')
            
            # Create iterators for both dataloaders
            sim_iter = iter(self.sim_loader)
            real_iter = iter(self.real_loader)
            
            # Progress bar for tracking
            pbar = tqdm(total=min(len(self.sim_loader), len(self.real_loader)))
            
            epoch_losses = []
            
            while True:
                try:
                    real_s = next(sim_iter).to(self.device)
                    real_r = next(real_iter).to(self.device)
                except StopIteration:
                    break

                # Execute training step
                losses = self.train_step(real_s, real_r)
                epoch_losses.append(losses)
                
                pbar.update(1)
                pbar.set_postfix(losses)

            pbar.close()
            
            # Calculate and log average losses
            avg_losses = {
                k: sum(d[k] for d in epoch_losses) / len(epoch_losses)
                for k in epoch_losses[0].keys()
            }
            self.logger.info(f'Epoch {epoch+1} average losses: {avg_losses}')
            
            # Save sample images
            if (epoch + 1) % self.config['sample_interval'] == 0:
                save_sample_images(
                    self.G_S2R, self.G_R2S,
                    real_s, real_r,
                    os.path.join(self.log_dir, f'samples_epoch_{epoch+1}')
                )
            
            # Save checkpoint
            if (epoch + 1) % self.config['checkpoint_interval'] == 0:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'G_S2R_state_dict': self.G_S2R.state_dict(),
                        'G_R2S_state_dict': self.G_R2S.state_dict(),
                        'D_S_state_dict': self.D_S.state_dict(),
                        'D_R_state_dict': self.D_R.state_dict(),
                        'optimizer_G_state_dict': self.optimizer_G.state_dict(),
                        'optimizer_D_state_dict': self.optimizer_D.state_dict(),
                    },
                    os.path.join(self.log_dir, f'checkpoint_epoch_{epoch+1}.pth')
                )

def main():
    # Load configuration
    config = {
        'sim_data_path': 'data/simulated',
        'real_data_path': 'data/real',
        'batch_size': 4,
        'num_epochs': 200,
        'learning_rate': 0.0002,
        'beta1': 0.975,
        'beta2': 0.999,
        'lambda_cycle': 5.0,
        'lambda_refine': 2.0,
        'sample_interval': 5,
        'checkpoint_interval': 10,
        'num_workers': 4
    }

    # Initialize and start training
    trainer = SIRGANTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
