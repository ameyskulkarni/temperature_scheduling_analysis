"""
Model Evaluation Script

Clean, modular evaluation for:
- CIFAR-10/100 test sets
- CIFAR-10/100-C corrupted test sets
- Extensible for future datasets

Usage:
    python eval_checkpoint.py --checkpoint model.pth --dataset cifar100
    python eval_checkpoint.py --checkpoint model.pth --dataset cifar100-c --severity 3
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import os
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

from models import get_model


# =============================================================================
# Data Classes for Clean Results
# =============================================================================

@dataclass
class EvalResult:
    """Container for evaluation results."""
    accuracy: float
    loss: float
    ece: float = None

    def to_dict(self) -> Dict:
        result = {'accuracy': self.accuracy, 'loss': self.loss}
        if self.ece is not None:
            result['ece'] = self.ece
        return result


@dataclass
class CorruptionResult:
    """Container for corruption evaluation results."""
    per_corruption: Dict[str, EvalResult]
    mean_accuracy: float
    mean_corruption_error: float
    mean_ece: float

    def to_dict(self) -> Dict:
        return {
            'per_corruption': {k: v.to_dict() for k, v in self.per_corruption.items()},
            'mean_accuracy': self.mean_accuracy,
            'mean_corruption_error': self.mean_corruption_error,
            'mean_ece': self.mean_ece,
        }


# =============================================================================
# Dataset Configurations
# =============================================================================

DATASET_CONFIGS = {
    'cifar10': {
        'num_classes': 10,
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2023, 0.1994, 0.2010],
        'dataset_class': torchvision.datasets.CIFAR10,
        'corruption_dir': 'CIFAR-10-C',
    },
    'cifar100': {
        'num_classes': 100,
        'mean': [0.5071, 0.4867, 0.4408],
        'std': [0.2675, 0.2565, 0.2761],
        'dataset_class': torchvision.datasets.CIFAR100,
        'corruption_dir': 'CIFAR-100-C',
    },
}

CORRUPTION_TYPES = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]


# =============================================================================
# Core Evaluation Functions
# =============================================================================

def compute_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute classification accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total


def compute_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute average cross-entropy loss."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item()
            total_samples += targets.size(0)

    return total_loss / total_samples


def compute_ece(model: nn.Module, loader: DataLoader, device: torch.device,
                n_bins: int = 15) -> float:
    """Compute Expected Calibration Error."""
    model.eval()
    all_probs, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            probs = F.softmax(model(inputs), dim=1)
            all_probs.append(probs.cpu())
            all_targets.append(targets)

    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)

    confidences, predictions = all_probs.max(dim=1)
    accuracies = predictions.eq(all_targets)

    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            acc_in_bin = accuracies[in_bin].float().mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += torch.abs(conf_in_bin - acc_in_bin) * prop_in_bin

    return ece.item()


def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device,
                    compute_calibration: bool = True) -> EvalResult:
    """Evaluate model on a single data loader."""
    accuracy = compute_accuracy(model, loader, device)
    loss = compute_loss(model, loader, device)
    ece = compute_ece(model, loader, device) if compute_calibration else None

    return EvalResult(accuracy=accuracy, loss=loss, ece=ece)


# =============================================================================
# Dataset Loaders
# =============================================================================

def get_test_loader(dataset_name: str, data_root: str = './data',
                    batch_size: int = 128, num_workers: int = 4, eval_split: str = 'val') -> DataLoader:
    """
    Get test set loader for CIFAR-10 or CIFAR-100.

    Args:
        dataset_name: 'cifar10' or 'cifar100'
        data_root: Root directory for data
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        DataLoader for the test set
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[dataset_name]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config['mean'], config['std'])
    ])

    # Load full training dataset (without augmentation first for splitting)
    full_train_dataset = config['dataset_class'](root=data_root, train=True, download=True,
                                       transform=transform)

    # Load test dataset
    test_dataset = config['dataset_class'](root=data_root, train=False, download=True,
                                 transform=transform)

    # Get targets for stratified splitting
    targets = np.array(full_train_dataset.targets)
    indices = np.arange(len(full_train_dataset))

    # Stratified split: 90% train, 10% val
    # Using sklearn's train_test_split with stratify ensures equal class proportions
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.1,
        random_state=100,  # Fixed seed for reproducibility
        stratify=targets  # Ensures equal proportion from each class
    )


    # For validation, we need test-time transforms (no augmentation)
    # Create a separate dataset with test transforms for validation
    val_base_dataset = config['dataset_class'](root=data_root, train=True, download=False,
                                     transform=transform)
    val_dataset = Subset(val_base_dataset, val_indices)

    # DataLoaders
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    if eval_split == 'val':
        return val_loader
    elif eval_split == 'test':
        return test_loader


def get_corruption_loaders(dataset_name: str, data_root: str = './data',
                           batch_size: int = 128, num_workers: int = 4,
                           severity: int = 3,
                           corruption_types: List[str] = None) -> Dict[str, DataLoader]:
    """
    Get data loaders for CIFAR-C corrupted datasets.

    Args:
        dataset_name: 'cifar10' or 'cifar100'
        data_root: Root directory for data
        batch_size: Batch size
        num_workers: Number of data loading workers
        severity: Corruption severity (1-5)
        corruption_types: List of corruptions to load (default: all)

    Returns:
        Dictionary mapping corruption_name -> DataLoader
    """
    from PIL import Image

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    config = DATASET_CONFIGS[dataset_name]
    corruption_dir = os.path.join(data_root, config['corruption_dir'])

    if not os.path.exists(corruption_dir):
        raise FileNotFoundError(
            f"{config['corruption_dir']} not found at {corruption_dir}. "
            f"Download from: https://zenodo.org/record/3555552"
        )

    # Load labels
    labels = np.load(os.path.join(corruption_dir, 'labels.npy'))[:10000]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config['mean'], config['std'])
    ])

    if corruption_types is None:
        corruption_types = CORRUPTION_TYPES

    loaders = {}
    for corruption in corruption_types:
        data_path = os.path.join(corruption_dir, f'{corruption}.npy')
        if not os.path.exists(data_path):
            print(f"Warning: {corruption}.npy not found, skipping")
            continue

        # Extract data for specified severity
        all_data = np.load(data_path)
        start_idx = (severity - 1) * 10000
        data = all_data[start_idx:start_idx + 10000]

        dataset = _CorruptedDataset(data, labels, transform)
        loaders[corruption] = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

    return loaders


class _CorruptedDataset(torch.utils.data.Dataset):
    """Simple dataset wrapper for corrupted images."""

    def __init__(self, data: np.ndarray, labels: np.ndarray, transform):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        from PIL import Image
        img = self.transform(Image.fromarray(self.data[idx]))
        return img, int(self.labels[idx])


# =============================================================================
# Model Loading
# =============================================================================

def load_model(checkpoint_path: str, model_name: str, num_classes: int,
               device: torch.device = None) -> nn.Module:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model_name: Model architecture name
        num_classes: Number of output classes
        device: Device to load model to

    Returns:
        Model in eval mode
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(model_name, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def load_model_from_state_dict(state_dict: dict, model_name: str, num_classes: int,
                                device: torch.device = None) -> nn.Module:
    """Load model from state dict (for merged models)."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(model_name, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


# =============================================================================
# High-Level Evaluation Functions
# =============================================================================

def evaluate_on_testset(model: nn.Module, dataset_name: str, device: torch.device,
                        data_root: str = './data', batch_size: int = 128,
                        num_workers: int = 4, eval_split: str = 'val', verbose: bool = True) -> EvalResult:
    """
    Evaluate model on standard test set.

    Args:
        model: Model to evaluate
        dataset_name: 'cifar10' or 'cifar100'
        device: Device for evaluation
        data_root: Data root directory
        batch_size: Batch size
        num_workers: Data loading workers
        verbose: Print results

    Returns:
        EvalResult with accuracy, loss, and ECE
    """
    loader = get_test_loader(dataset_name, data_root, batch_size, num_workers, eval_split)
    result = evaluate_loader(model, loader, device)

    if verbose:
        print(f"{dataset_name.upper()} Test Set:")
        print(f"  Accuracy: {result.accuracy:.2f}%")
        print(f"  Loss: {result.loss:.4f}")
        print(f"  ECE: {result.ece:.4f}")

    return result


def evaluate_on_corruptions(model: nn.Module, dataset_name: str, device: torch.device,
                            data_root: str = './data', batch_size: int = 128,
                            num_workers: int = 4, severity: int = 3,
                            verbose: bool = True) -> CorruptionResult:
    """
    Evaluate model on corrupted test set (CIFAR-C).

    Args:
        model: Model to evaluate
        dataset_name: 'cifar10' or 'cifar100'
        device: Device for evaluation
        data_root: Data root directory
        batch_size: Batch size
        num_workers: Data loading workers
        severity: Corruption severity (1-5)
        verbose: Print results

    Returns:
        CorruptionResult with per-corruption and mean metrics
    """
    loaders = get_corruption_loaders(dataset_name, data_root, batch_size,
                                      num_workers, severity)

    per_corruption = {}
    accuracies = []
    eces = []

    iterator = tqdm(loaders.items(), desc='Evaluating corruptions') if verbose else loaders.items()

    for corruption_name, loader in iterator:
        result = evaluate_loader(model, loader, device, compute_calibration=True)
        per_corruption[corruption_name] = result
        accuracies.append(result.accuracy)
        eces.append(result.ece)

    mean_acc = np.mean(accuracies)
    mean_ece = np.mean(eces)

    result = CorruptionResult(
        per_corruption=per_corruption,
        mean_accuracy=mean_acc,
        mean_corruption_error=100.0 - mean_acc,
        mean_ece=mean_ece,
    )

    if verbose:
        print(f"\n{dataset_name.upper()}-C (severity={severity}):")
        print(f"  Mean Accuracy: {result.mean_accuracy:.2f}%")
        print(f"  Mean Corruption Error: {result.mean_corruption_error:.2f}%")
        print(f"  Mean ECE: {result.mean_ece:.4f}")

    return result


def evaluate_checkpoint(checkpoint_path: str, model_name: str, dataset_name: str,
                        include_corruptions: bool = True, corruption_severity: int = 3,
                        data_root: str = './data', batch_size: int = 128,
                        num_workers: int = 4, device: torch.device = None, eval_split: str = 'val',
                        verbose: bool = True) -> Dict:
    """
    Full evaluation of a checkpoint on test set and optionally corruptions.

    Args:
        checkpoint_path: Path to model checkpoint
        model_name: Model architecture
        dataset_name: 'cifar10' or 'cifar100'
        include_corruptions: Whether to evaluate on CIFAR-C
        corruption_severity: Severity level for CIFAR-C
        data_root: Data root directory
        batch_size: Batch size
        num_workers: Data loading workers
        device: Device for evaluation
        verbose: Print results

    Returns:
        Dictionary with all results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = DATASET_CONFIGS[dataset_name]
    model = load_model(checkpoint_path, model_name, config['num_classes'], device)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating: {checkpoint_path}")
        print(f"{'='*60}\n")

    results = {
        'checkpoint': str(checkpoint_path),
        'model': model_name,
        'dataset': dataset_name,
    }

    # Test set evaluation
    test_result = evaluate_on_testset(model, dataset_name, device, data_root,
                                       batch_size, num_workers, eval_split, verbose)
    results['test'] = test_result.to_dict()

    # Corruption evaluation
    if include_corruptions:
        if verbose:
            print()
        corruption_result = evaluate_on_corruptions(model, dataset_name, device,
                                                     data_root, batch_size, num_workers,
                                                     corruption_severity, verbose)
        results['corruptions'] = corruption_result.to_dict()

    return results


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate model checkpoint')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='resnet32',
                        help='Model architecture')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100'],
                        help='Dataset to evaluate on')
    parser.add_argument('--no-corruptions', action='store_true',
                        help='Skip corruption evaluation')
    parser.add_argument('--severity', type=int, default=3,
                        help='Corruption severity (1-5)')
    parser.add_argument('--data-root', type=str, default='./data',
                        help='Data root directory')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--eval-split', type=str, default='val',
                        choices=['val', 'test'],
                        help='Which split to use for evaluation during training (val or test)')

    args = parser.parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        dataset_name=args.dataset,
        include_corruptions=not args.no_corruptions,
        corruption_severity=args.severity,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        eval_split=args.eval_split,
        verbose=True
    )

    return results


if __name__ == '__main__':
    main()