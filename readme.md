# AirSim-IR & SIR-GAN: Synthetic Infrared Image Generation and Refinement

This repository contains two complementary projects for synthetic infrared image generation and refinement:

1. **AirSim-IR**: A framework for generating synthetic infrared images using Microsoft's AirSim plugin
2. **SIR-GAN**: An implementation of "Synthetic IR Image Refinement using Adversarial Learning with Bidirectional Mappings" for refining synthetic IR images

## Overview

The combined framework provides an end-to-end solution for generating and refining synthetic infrared images. AirSim-IR generates the initial synthetic data, while SIR-GAN refines these images to better match real-world infrared characteristics. This approach addresses the challenge of limited real infrared image data availability while ensuring high-quality synthetic data for training and testing purposes.

## Core Components

### SIR-GAN Features

#### Architecture
- Bidirectional mapping between simulated and real IR domains
- Custom U-Net generator architecture with skip connections
- Discriminators with dilated convolutions for spatial hierarchy preservation
- Novel loss function combining adversarial, cycle consistency, and refinement components
- Support for unpaired and imbalanced dataset training

#### Loss Functions
- SIR adversarial loss for domain adaptation
- SIR cycle consistency loss for bidirectional mapping
- SIR refinement loss combining:
  - Infrared characteristics preservation
  - Structural information retention

### AirSim-IR Features

#### Environment Simulation
- African landscape environment simulation
- Dynamic time-of-day systems
- Weather variation support
- Accurate thermal signature reproduction
- Real-time environmental response

#### Image Generation
- Multi-angle IR image capture
- Automated segmentation map creation
- Thermal radiation modeling
- Real-time data capture
- Configurable camera parameters

## Prerequisites

### Global Requirements
- Python >= 3.8
- NVIDIA GPU with CUDA support
- CUDA >= 10.2 (for GPU support)
- Git LFS for handling large model files

### AirSim-IR Specific
- Unreal Engine 4.25+
- Microsoft AirSim plugin
- OGRE rendering engine
- Windows 10+ or Ubuntu 18.04+

### SIR-GAN Specific
- PyTorch >= 1.7.0
- torchvision >= 0.8.0
- numpy >= 1.19.2
- opencv-python >= 4.5.0
- matplotlib >= 3.3.4
- scipy >= 1.6.0

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/airsim-ir-gan.git
cd airsim-ir-gan
```

2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # Linux/Mac
# or
.\env\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure AirSim settings:
```bash
python setup/configure_airsim.py
```

## Project Structure
```
airsim-ir-gan/
├── airsim_ir/          # AirSim-IR implementation
│   ├── environment/    # Environment configuration
│   ├── capture/        # Image capture utilities
│   └── thermal/        # Thermal modeling
sir-gan/
├── configs/           # Configuration files
├── data/             # Dataset storage
├── models/           # Network architectures
│   ├── generator.py
│   └── discriminator.py
├── losses/           # Loss function implementations
├── utils/            # Utility functions
├── train.py          # Training script
├── test.py           # Testing script
└── evaluate.py       # Evaluation metrics
```

## Usage Guide

### Generate Synthetic IR Images
```bash
python scripts/capture_ir_segmentation.py --config configs/default.yaml
```

### Refine Synthetic Images
```bash
python scripts/refine_ir_images.py --input_dir data/synthetic --output_dir data/refined
```

### Training SIR-GAN
```bash
python train.py --config configs/training.yaml
```

## Training Pipeline

1. Data Preparation:
   - Generate synthetic IR data using AirSim-IR
   - Collect and preprocess real IR dataset
   - Configure data paths in config files

2. Model Training:
   - Initialize network weights
   - Configure training parameters
   - Execute training pipeline
   - Monitor convergence

3. Evaluation:
   - Compute quantitative metrics
   - Perform qualitative assessment
   - Generate comparison visualizations

## Evaluation Metrics

- Visual quality assessment
- Perceptual studies (AMT scores)
- FCN-score for semantic evaluation
- Infrared characteristics comparison
- Structural similarity metrics
- Cross-domain consistency measures

## Results

Our model achieves:
- Improved visual quality of synthetic IR images
- Higher AMT perceptual study scores
- Better FCN-score metrics compared to baselines
- Preserved thermal characteristics
- Robust structural consistency

## Contributing

We welcome contributions to enhance the framework's capabilities. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Citation

If you use this code in your research, please cite:
```bibtex
@article{zhang2019synthetic,
    title={Synthetic IR Image Refinement using Adversarial Learning with Bidirectional Mappings},
    author={Zhang, Ruiheng and Mu, Chengpo and Xu, Min and Xu, Lixin and Shi, Qiaolin and Wang, Junbo},
    journal={IEEE Access},
    year={2019}
}
```

## Acknowledgments

- Dr. Aparna Akula (CSIR-CSIO) for project guidance
- Elizabeth Bondi for AirSim-W and Birdsai reference implementation
- Microsoft for the AirSim plugin
- BITS Pilani for project support
- GPU resources provided by CSIO Chandigarh

## Contact

For questions or support:
- Open an issue in the repository
- Contact the project maintainers: hegdemanu22@gmail.com

## Development Roadmap

### Short Term
- [ ] Add support for additional thermal scenarios
- [ ] Implement real-time refinement pipeline
- [ ] Expand dataset diversity

### Long Term
- [ ] Improve training efficiency
- [ ] Add more evaluation metrics
- [ ] Develop web interface for model demo
- [ ] Create comprehensive documentation

## Troubleshooting

Common issues and solutions:
1. CUDA out of memory: Reduce batch size or image resolution
2. AirSim connection issues: Check simulator settings
3. Training instability: Adjust learning rates and loss weights
4. Environment setup: Verify dependency versions

## Publications

For more details about the methodology and results, please refer to our publications:
1. Original SIR-GAN paper
2. AirSim-IR technical documentation
