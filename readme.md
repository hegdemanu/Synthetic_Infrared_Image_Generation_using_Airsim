# AirSim-IR & SIR-GAN: Synthetic Infrared Image Generation and Refinement

This repository contains two complementary projects for synthetic infrared image generation and refinement:

1. **AirSim-IR**: A framework for generating synthetic infrared images using Microsoft's AirSim plugin
2. **SIR-GAN**: An implementation of "Synthetic IR Image Refinement using Adversarial Learning with Bidirectional Mappings" for refining synthetic IR images

## Overview

The combined framework provides an end-to-end solution for generating and refining synthetic infrared images. AirSim-IR generates the initial synthetic data, while SIR-GAN refines these images to better match real-world infrared characteristics.

## SIR-GAN Features

### Architecture
- Bidirectional mapping between simulated and real IR domains
- Custom U-Net generator with skip connections
- Discriminator with dilated convolutions
- Novel loss function incorporating:
  - SIR adversarial loss
  - SIR cycle consistency loss
  - SIR refinement loss (Infrared and Structure losses)

## AirSim-IR Features

### Environment Simulation
- African landscape environment simulation
- Dynamic time-of-day systems
- Weather variation support
- Accurate thermal signature reproduction

### Image Generation
- Multi-angle IR image capture
- Automated segmentation map creation
- Thermal radiation modeling
- Real-time data capture

## Prerequisites

### Global Requirements
- Python >= 3.8
- NVIDIA GPU with CUDA support
- CUDA >= 10.2 (for GPU support)

### AirSim-IR Specific
- Unreal Engine 4.25+
- Microsoft AirSim plugin
- OGRE rendering engine

### SIR-GAN Specific
- PyTorch >= 1.7.0
- torchvision >= 0.8.0
- numpy >= 1.19.2
- opencv-python >= 4.5.0

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/airsim-ir-gan.git
cd airsim-ir-gan
