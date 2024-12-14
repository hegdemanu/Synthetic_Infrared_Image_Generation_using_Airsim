import torch
import os
import logging

logger = logging.getLogger(__name__)

def save_checkpoint(state, filepath):
    """Save model and training state."""
    try:
        torch.save(state, filepath)
        logger.info(f"Checkpoint saved successfully at {filepath}")
    except Exception as e:
        logger.error(f"Error saving checkpoint: {str(e)}")

def load_checkpoint(filepath, models, optimizers=None):
    """
    Load model and training state.
    
    Args:
        filepath: Path to the checkpoint file
        models: Dictionary containing models {'G_S2R': g_s2r, 'G_R2S': g_r2s, 'D_S': d_s, 'D_R': d_r}
        optimizers: Optional dictionary containing optimizers {'G': opt_g, 'D': opt_d}
    
    Returns:
        start_epoch: The epoch to resume training from
    """
    try:
        checkpoint = torch.load(filepath)
        
        # Load model states
        models['G_S2R'].load_state_dict(checkpoint['G_S2R_state_dict'])
        models['G_R2S'].load_state_dict(checkpoint['G_R2S_state_dict'])
        models['D_S'].load_state_dict(checkpoint['D_S_state_dict'])
        models['D_R'].load_state_dict(checkpoint['D_R_state_dict'])
        
        # Load optimizer states if provided
        if optimizers is not None:
            optimizers['G'].load_state_dict(checkpoint['optimizer_G_state_dict'])
            optimizers['D'].load_state_dict(checkpoint['optimizer_D_state_dict'])
        
        start_epoch = checkpoint['epoch']
        logger.info(f"Checkpoint loaded successfully from {filepath}")
        return start_epoch
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        return 0

def save_best_model(state, filepath, current_metric, best_metric, higher_better=True):
    """
    Save model if it performs better than the previous best.
    
    Args:
        state: Model state to save
        filepath: Path to save the model
        current_metric: Current performance metric
        best_metric: Best performance metric so far
        higher_better: Whether higher metric is better (True) or lower is better (False)
    
    Returns:
        best_metric: Updated best metric value
    """
    is_best = (higher_better and current_metric > best_metric) or \
              (not higher_better and current_metric < best_metric)
              
    if is_best:
        save_checkpoint(state, filepath)
        logger.info(f"New best model saved with metric: {current_metric}")
        return current_metric
    
    return best_metric
