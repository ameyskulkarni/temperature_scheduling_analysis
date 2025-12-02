# Temperature Scheduling for Neural Network Training

## Overview

This repository investigates the effects of **temperature scheduling** during neural network training on model calibration, accuracy, and robustness. The core hypothesis is that dynamically adjusting the softmax temperature throughout training can improve model properties beyond what constant temperature training achieves.

## Thesis

Standard neural network training uses a fixed temperature (T=1) in the softmax function for cross-entropy loss. However, temperature affects the "sharpness" of probability distributions:
- **Low temperature (T < 1)**: Produces sharper, more confident predictions
- **High temperature (T > 1)**: Produces softer, more uniform predictions

By scheduling temperature throughout training, we hypothesize that:
1. **Early high temperature** encourages exploration and prevents premature convergence to local minima
2. **Late low temperature** sharpens decision boundaries for final predictions
3. **Cyclic schedules** may help escape local minima and improve generalization

## Experiments

### Temperature Schedules Implemented

| Schedule | Description |
|----------|-------------|
| `constant` | Fixed temperature throughout training |
| `cosine` | Cosine annealing from T_max to T_min |
| `linear` | Linear decay from T_max to T_min |
| `reverse_cosine` | Starts at T_min, increases to T_max |
| `cyclic_sinusoidal` | Oscillates between T_min and T_max |
| `cyclic_triangular` | Triangle wave between T_min and T_max |

### Evaluation Metrics

- **Test Accuracy**: Standard classification accuracy
- **ECE (Expected Calibration Error)**: Measures probability calibration
- **Corruption Accuracy**: Robustness on CIFAR-C corrupted images
- **Shape vs Texture Bias**: Evaluates reliance on shape vs texture cues

### Key Results (CIFAR-100, ResNet-32)
Upcoming shortly.

## Usage

### Training

```bash
python train.py \
    --exp-name my_experiment \
    --dataset cifar100 \
    --model resnet32 \
    --epochs 200 \
    --temp-schedule cosine \
    --temp-max 2.0 \
    --temp-min 0.07 \
    --lr-schedule cosine \
    --eval-corruptions \
    --eval-shape-texture
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--temp-schedule` | Temperature schedule type | `constant` |
| `--temp-max` | Maximum/initial temperature | 1.0 |
| `--temp-min` | Minimum/final temperature | 1.0 |
| `--temp-cycle-length` | Cycle length for cyclic schedules | 40 |
| `--lr-schedule` | Learning rate schedule | `constant` |
| `--eval-corruptions` | Evaluate on CIFAR-C | False |
| `--eval-shape-texture` | Evaluate shape/texture bias | False |

## Project Structure

```
├── train.py                    # Main training script
├── models.py                   # Model architectures (ResNet, WRN, ViT)
├── datasets.py                 # Dataset loading (CIFAR-10/100, long-tailed)
├── temperature_schedulers.py   # Temperature scheduling implementations
├── lr_scheduler.py             # Learning rate schedulers
├── shape_texture_bias.py       # Shape vs texture bias evaluation
├── utils.py                    # Utilities (ECE, corruption evaluation)
└── Temperature_Scheduling_Results.xlsx  # Experiment results
```

## Requirements

- PyTorch
- torchvision
- wandb (for logging)
- numpy
- opencv-python
- tqdm

## Related Work

This project builds on concepts from:
- Knowledge distillation (Hinton et al., 2015)
- Label smoothing and calibration
- Curriculum learning