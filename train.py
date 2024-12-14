import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import os
import numpy as np
from datetime import datetime

from models.generator import Generator
from models.discriminator import Discriminator
from losses.sir_losses import SIRLoss
from utils.dataset import IRDataset
from utils.visualization import save_sample_images, plot_training_progress
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.metrics import ValidationMetrics

class EarlyStopping:
    """Early stopping handler to prevent overfitting."""
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
            return False

        if current_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = current_loss
            self.counter = 0
            
        return self.should_stop

class SIRGANTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_logging()
        self.initialize_models()
        self.setup_optimizers()
        self.setup_dataloaders()
        self.setup_losses()
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 7),
            min_delta=config.get('min_delta', 0.001)
        )
        self.metrics = ValidationMetrics(self.device)

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
        self.G_S2R = Generator().to(self.device)
        self.G_R2S = Generator().to(self.device)
        self.D_S = Discriminator().to(self.device)
        self.D_R = Discriminator().to(self.device)
        
        for model in [self.G_S2R, self.G_R2S, self.D_S, self.D_R]:
            model.init_weights(init_type='normal', gain=0.02)

    def setup_optimizers(self):
        """Initialize optimizers with Nesterov-accelerated Adam."""
        beta1 = self.config.get('beta1', 0.975)
        beta2 = self.config.get('beta2', 0.999)
        lr = self.config.get('learning_rate', 0.0002)

        self.optimizer_G = optim.Adam(
            list(self.G_S2R.parameters()) + list(self.G_R2S.parameters()),
            lr=lr, betas=(beta1, beta2), nesterov=True
        )

        self.optimizer_D = optim.Adam(
            list(self.D_S.parameters()) + list(self.D_R.parameters()),
            lr=lr, betas=(beta1, beta2), nesterov=True
        )

    def setup_dataloaders(self):
        """Initialize data loaders for real and simulated IR images."""
        self.sim_dataset = IRDataset(
            self.config['sim_data_path'],
            transform=self.config.get('transforms', None)
        )
        self.real_dataset = IRDataset(
            self.config['real_data_path'],
            transform=self.config.get('transforms', None)
        )

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

    def log_model_statistics(self):
        """Log statistics about model parameters and gradients."""
        for name, model in [('G_S2R', self.G_S2R), ('G_R2S', self.G_R2S),
                          ('D_S', self.D_S), ('D_R', self.D_R)]:
            total_norm = 0.0
            param_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
                    param_norm += p.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            param_norm = param_norm ** 0.5
            
            self.logger.info(f'{name} - Gradient Norm: {total_norm:.4f}, Parameter Norm: {param_norm:.4f}')

    def train_step(self, real_s, real_r):
        """Execute a single training step with gradient clipping."""
        # Forward cycle: S -> R -> S
        fake_r = self.G_S2R(real_s)
        cycle_s = self.G_R2S(fake_r)
        
        # Backward cycle: R -> S -> R
        fake_s = self.G_R2S(real_r)
        cycle_r = self.G_S2R(fake_s)

        # Update Generators with gradient clipping
        self.optimizer_G.zero_grad()
        g_loss = self.sir_loss.compute_total_loss(
            real_s, real_r, fake_s, fake_r, cycle_s, cycle_r,
            self.D_S, self.D_R, is_generator=True
        )
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.G_S2R.parameters()) + list(self.G_R2S.parameters()),
            max_norm=self.config.get('grad_clip_value', 5.0)
        )
        self.optimizer_G.step()

        # Update Discriminators with gradient clipping
        self.optimizer_D.zero_grad()
        d_loss = self.sir_loss.compute_total_loss(
            real_s, real_r, fake_s, fake_r, cycle_s, cycle_r,
            self.D_S, self.D_R, is_generator=False
        )
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.D_S.parameters()) + list(self.D_R.parameters()),
            max_norm=self.config.get('grad_clip_value', 5.0)
        )
        self.optimizer_D.step()

        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item()
        }

    def validate(self):
        """Execute validation loop and compute metrics."""
        self.G_S2R.eval()
        self.G_R2S.eval()
        
        validation_metrics = {'psnr': [], 'ssim': [], 'thermal_consistency': []}
        
        with torch.no_grad():
            for real_s, real_r in zip(self.sim_loader, self.real_loader):
                real_s = real_s.to(self.device)
                real_r = real_r.to(self.device)
                
                # Generate translations
                fake_r = self.G_S2R(real_s)
                fake_s = self.G_R2S(real_r)
                
                # Compute metrics for both directions
                s2r_metrics = self.metrics.compute_all_metrics(real_r, fake_r)
                r2s_metrics = self.metrics.compute_all_metrics(real_s, fake_s)
                
                # Aggregate metrics
                for key in validation_metrics:
                    validation_metrics[key].append((s2r_metrics[key] + r2s_metrics[key]) / 2)
        
        # Calculate average metrics
        avg_metrics = {key: np.mean(values) for key, values in validation_metrics.items()}
        
        self.logger.info(f"Validation Metrics: {avg_metrics}")
        return avg_metrics

    def resume_from_checkpoint(self, checkpoint_path):
        """Resume training from a checkpoint."""
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            
            self.G_S2R.load_state_dict(checkpoint['G_S2R_state_dict'])
            self.G_R2S.load_state_dict(checkpoint['G_R2S_state_dict'])
            self.D_S.load_state_dict(checkpoint['D_S_state_dict'])
            self.D_R.load_state_dict(checkpoint['D_R_state_dict'])
            
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            
            return checkpoint['epoch']
        
        return 0

    def clear_cache(self):
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train(self):
        """Execute the main training loop with validation and early stopping."""
        num_epochs = self.config['num_epochs']
        start_epoch = 0

        # Resume from checkpoint if specified
        if self.config.get('resume_checkpoint'):
            start_epoch = self.resume_from_checkpoint(self.config['resume_checkpoint'])

        for epoch in range(start_epoch, num_epochs):
            self.logger.info(f'Starting epoch {epoch+1}/{num_epochs}')
            
            # Training
            self.G_S2R.train()
            self.G_R2S.train()
            
            sim_iter = iter(self.sim_loader)
            real_iter = iter(self.real_loader)
            pbar = tqdm(total=min(len(self.sim_loader), len(self.real_loader)))
            epoch_losses = []
            
            while True:
                try:
                    real_s = next(sim_iter).to(self.device)
                    real_r = next(real_iter).to(self.device)
                except StopIteration:
                    break

                losses = self.train_step(real_s, real_r)
                epoch_losses.append(losses)
                
                pbar.update(1)
                pbar.set_postfix(losses)

            pbar.close()
            
            # Log training statistics
            avg_losses = {
                k: sum(d[k] for d in epoch_losses) / len(epoch_losses)
                for k in epoch_losses[0].keys()
            }
            self.logger.info(f'Epoch {epoch+1} average losses: {avg_losses}')
            self.log_model_statistics()

            # Validation
            if (epoch + 1) % self.config.get('validation_interval', 5) == 0:
                metrics = self.validate()
                
                # Early stopping check
                if self.early_stopping(metrics['psnr']):
                    self.logger.info('Early stopping triggered')
                    break

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

            # Clear cache periodically
            if (epoch + 1) % 10 == 0:
                self.clear_cache()

def main():
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
        'validation_interval': 5,
        'num_workers': 4,
        'patience': 7,
        'min_delta': 0.001,
        'grad_clip_value': 5.0,
        'resume_checkpoint': None,
    }

    trainer = SIRGANTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
