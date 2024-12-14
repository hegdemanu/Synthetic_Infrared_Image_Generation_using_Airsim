# 

import argparse
import yaml
import wandb
from train import SIRGANTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize wandb
    wandb.init(project="sir-gan", config=config)
    
    # Initialize trainer
    trainer = SIRGANTrainer(config)
    
    # Train model
    trainer.train()

if __name__ == '__main__':
    main()
