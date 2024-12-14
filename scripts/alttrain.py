# 

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb  # For logging
import os

from models import Generator, Discriminator
from losses import SIRLoss
from utils import IRDataset, save_checkpoint, load_checkpoint, WarmupCosineScheduler

class SIRGANTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.gen_S2R = Generator().to(self.device)
        self.gen_R2S = Generator().to(self.device)
        self.disc_S = Discriminator().to(self.device)
        self.disc_R = Discriminator().to(self.device)
        
        # Initialize optimizers
        self.optimizer_G = optim.NAdam(
            list(self.gen_S2R.parameters()) + list(self.gen_R2S.parameters()),
            lr=config['learning_rate'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        self.optimizer_D = optim.NAdam(
            list(self.disc_S.parameters()) + list(self.disc_R.parameters()),
            lr=config['learning_rate'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Initialize learning rate scheduler
        self.scheduler_G = WarmupCosineScheduler(
            self.optimizer_G,
            warmup_epochs=config['warmup_epochs'],
            max_epochs=config['num_epochs']
        )
        self.scheduler_D = WarmupCosineScheduler(
            self.optimizer_D,
            warmup_epochs=config['warmup_epochs'],
            max_epochs=config['num_epochs']
        )
        
        # Initialize loss
        self.criterion = SIRLoss(
            lambda_adv=config['lambda_adv'],
            lambda_cycle=config['lambda_cycle'],
            lambda_ir=config['lambda_ir'],
            lambda_struct=config['lambda_struct']
        )
        
        # Initialize dataloaders
        self.train_dataloader = DataLoader(
            IRDataset(config['sim_dir'], config['real_dir'], phase='train'),
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers']
        )
        self.val_dataloader = DataLoader(
            IRDataset(config['sim_dir'], config['real_dir'], phase='val'),
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers']
        )

    def train_step(self, real_batch, sim_batch):
        # Generate fake images
        fake_real = self.gen_S2R(sim_batch)
        fake_sim = self.gen_R2S(real_batch)
        
        # Cycle consistency
        cycled_sim = self.gen_R2S(fake_real)
        cycled_real = self.gen_S2R(fake_sim)
        
        # Discriminator predictions
        disc_real_pred = self.disc_R(real_batch)
        disc_fake_real_pred = self.disc_R(fake_real.detach())
        disc_sim_pred = self.disc_S(sim_batch)
        disc_fake_sim_pred = self.disc_S(fake_sim.detach())
        
        # Calculate losses
        loss_dict = self.criterion(
            real_batch, sim_batch,
            fake_sim, fake_real,
            cycled_sim, cycled_real,
            disc_fake_real_pred, disc_real_pred,
            disc_fake_sim_pred, disc_sim_pred
        )
        
        return loss_dict, fake_real, fake_sim

    def train_epoch(self, epoch):
        self.gen_S2R.train()
        self.gen_R2S.train()
        self.disc_S.train()
        self.disc_R.train()
        
        total_loss = 0
        pbar = tqdm(self.train_dataloader, desc=f'Epoch {epoch}')
        
        for batch in pbar:
            real_batch = batch['real'].to(self.device)
            sim_batch = batch['sim'].to(self.device)
            
            # Train discriminators
            self.optimizer_D.zero_grad()
            loss_dict, fake_real, fake_sim = self.train_step(real_batch, sim_batch)
            loss_dict['total_loss'].backward()
            self.optimizer_D.step()
            
            # Train generators
            self.optimizer_G.zero_grad()
            loss_dict, _, _ = self.train_step(real_batch, sim_batch)
            loss_dict['total_loss'].backward()
            self.optimizer_G.step()
            
            total_loss += loss_dict['total_loss'].item()
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})
            
            # Log metrics
            if wandb.run:
                wandb.log(loss_dict)
        
        return total_loss / len(self.train_dataloader)

    def validate(self):
        self.gen_S2R.eval()
        self.gen_R2S.eval()
        self.disc_S.eval()
        self.disc_R.eval()
        
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                real_batch = batch['real'].to(self.device)
                sim_batch = batch['sim'].to(self.device)
                
                loss_dict, _, _ = self.train_step(real_batch, sim_batch)
                total_val_loss += loss_dict['total_loss'].item()
        
        return total_val_loss / len(self.val_dataloader)

    def train(self):
        best_val_loss = float('inf')
        
        for epoch in range(self.config['num_epochs']):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            # Update learning rates
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint({
                    'gen_S2R': self.gen_S2R.state_dict(),
                    'gen_R2S': self.gen_R2S.state_dict(),
                    'disc_S': self.disc_S.state_dict(),
                    'disc_R': self.disc_R.state_dict(),
                    'optimizer_G': self.optimizer_G.state_dict(),
                    'optimizer_D': self.optimizer_D.state_dict(),
                    'epoch': epoch
                }, self.config['checkpoint_dir'], 'best_model.pth')
