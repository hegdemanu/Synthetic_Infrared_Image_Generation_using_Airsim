import torch
import math

class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine decay."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, initial_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        
        self.current_epoch = 0
        
    def step(self):
        """Update learning rate based on current epoch."""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.initial_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        self.current_epoch += 1
        return lr
