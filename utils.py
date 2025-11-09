"""
Utility Functions
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms


def compute_ece(predictions, targets, n_bins=15):
    """
    Compute Expected Calibration Error

    Args:
        predictions: (N, C) tensor of softmax probabilities
        targets: (N,) tensor of ground truth labels
        n_bins: number of bins

    Returns:
        ECE value (float)
    """
    confidences, predicted_labels = predictions.max(dim=1)
    accuracies = predicted_labels.eq(targets)

    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()


class CIFARCDataset(Dataset):
    """CIFAR-C corrupted dataset"""

    def __init__(self, data, labels, transform):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


def evaluate_on_corruptions(model, dataset_name, data_root, batch_size,
                           num_workers, device, temperature=1.0, severity=3):
    """
    Evaluate model on CIFAR-C corruptions

    Args:
        model: PyTorch model
        dataset_name: 'cifar10' or 'cifar100'
        data_root: path to data directory
        batch_size: batch size
        num_workers: number of workers
        device: torch device
        temperature: temperature to use for evaluation
        severity: corruption severity level (1-5)

    Returns:
        dict with corruption results
    """

    corruption_types = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    ]

    # Get normalization
    if dataset_name == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        cifar_c_path = os.path.join(data_root, 'CIFAR-10-C')
    elif dataset_name == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        cifar_c_path = os.path.join(data_root, 'CIFAR-100-C')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load labels (same for all corruptions)
    labels = np.load(os.path.join(cifar_c_path, 'labels.npy'))

    model.eval()
    results = {'per_corruption': {}}
    all_accuracies = []
    all_eces = []

    for corruption in tqdm(corruption_types, desc='Evaluating corruptions'):
        try:
            # Load corrupted data
            data_path = os.path.join(cifar_c_path, f'{corruption}.npy')
            if not os.path.exists(data_path):
                print(f"Warning: {corruption}.npy not found")
                continue

            all_data = np.load(data_path)

            # Extract data for specific severity
            start_idx = (severity - 1) * 10000
            end_idx = severity * 10000
            data = all_data[start_idx:end_idx]

            # Create dataset and loader
            dataset = CIFARCDataset(data, labels, transform)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

            # Evaluate
            correct = 0
            total = 0
            all_probs = []
            all_targets = []

            with torch.no_grad():
                for inputs, targets in loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    logits = model(inputs)

                    # Use specified temperature for evaluation
                    probs = F.softmax(logits / temperature, dim=1)

                    _, predicted = logits.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    all_probs.append(probs.cpu())
                    all_targets.append(targets.cpu())

            accuracy = 100.0 * correct / total
            all_probs = torch.cat(all_probs)
            all_targets = torch.cat(all_targets)
            ece = compute_ece(all_probs, all_targets)

            results['per_corruption'][corruption] = {
                'accuracy': accuracy,
                'ece': ece
            }

            all_accuracies.append(accuracy)
            all_eces.append(ece)

        except Exception as e:
            print(f"Error evaluating {corruption}: {e}")
            continue

    # Compute mean metrics
    if len(all_accuracies) > 0:
        results['mean_acc'] = np.mean(all_accuracies)
        results['mean_ece'] = np.mean(all_eces)
    else:
        results['mean_acc'] = 0.0
        results['mean_ece'] = 0.0

    return results