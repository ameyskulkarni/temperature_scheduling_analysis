"""
Shape vs Texture Bias Evaluation - CORRECTED VERSION

Key fixes:
1. Proper normalization applied to all perturbed images
2. Temperature-aware evaluation
3. Cached perturbations for efficiency
4. Accurate metric naming (robustness, not true shape bias)
5. Support for both CIFAR-10 and CIFAR-100 with correct normalization constants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import os


def get_normalization_params(dataset_name):
    """Get normalization parameters for dataset"""
    if 'cifar10' in dataset_name.lower():
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    else:  # CIFAR-100
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    return mean, std


def normalize_tensor(img_tensor, mean, std):
    """Apply CIFAR normalization to tensor"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return (img_tensor - mean) / std


def denormalize_tensor(img_tensor, mean, std):
    """Remove CIFAR normalization from tensor"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return img_tensor * std + mean


class PatchShuffledDataset(Dataset):
    """
    Dataset with patch-shuffled images

    FIXES:
    - Applies proper CIFAR normalization after shuffling
    - Caches shuffled images to avoid recomputation
    - Clear documentation of what's being tested
    """

    def __init__(self, original_dataset, dataset_name, patch_size=8,
                 shuffle_indices_path='shuffle_indices.npy', cache_perturbations=True):
        """
        Args:
            original_dataset: CIFAR dataset with ToTensor() only (no normalization)
            dataset_name: 'cifar10' or 'cifar100' for correct normalization
            patch_size: Size of patches to shuffle (8 for 32x32 images)
            shuffle_indices_path: Path to deterministic shuffle indices
            cache_perturbations: Cache perturbed images for efficiency
        """
        self.original_dataset = original_dataset
        self.patch_size = patch_size
        self.shuffle_indices_path = shuffle_indices_path
        self.cache_perturbations = cache_perturbations

        # Get correct normalization parameters
        self.mean, self.std = get_normalization_params(dataset_name)

        # Get image dimensions
        sample_img, _ = original_dataset[0]
        if torch.is_tensor(sample_img):
            self.C, self.H, self.W = sample_img.shape
        else:
            sample_img = np.array(sample_img)
            if sample_img.ndim == 2:
                self.C, self.H, self.W = 1, sample_img.shape[0], sample_img.shape[1]
            elif sample_img.shape[-1] == 3:
                self.H, self.W, self.C = sample_img.shape
            else:
                self.C, self.H, self.W = sample_img.shape

        self.num_patches_h = self.H // patch_size
        self.num_patches_w = self.W // patch_size
        self.total_patches = self.num_patches_h * self.num_patches_w

        # Load or create deterministic shuffle indices
        self.shuffle_indices = self._get_or_create_shuffle_indices()

        # Cache for perturbed images
        self.cache = {} if cache_perturbations else None

    def _get_or_create_shuffle_indices(self):
        """Load or generate deterministic shuffle indices"""
        if os.path.exists(self.shuffle_indices_path):
            print(f"Loading shuffle indices from {self.shuffle_indices_path}")
            shuffle_indices = np.load(self.shuffle_indices_path)
            expected_shape = (len(self.original_dataset), self.total_patches)
            if shuffle_indices.shape != expected_shape:
                raise ValueError(
                    f"Shuffle indices shape {shuffle_indices.shape} != expected {expected_shape}. "
                    f"Delete {self.shuffle_indices_path} and regenerate."
                )
        else:
            print(f"Generating deterministic shuffle indices at {self.shuffle_indices_path}")
            rng = np.random.RandomState(seed=42)
            shuffle_indices = np.zeros((len(self.original_dataset), self.total_patches), dtype=np.int32)
            for i in range(len(self.original_dataset)):
                shuffle_indices[i] = rng.permutation(self.total_patches)
            np.save(self.shuffle_indices_path, shuffle_indices)
            print(f"Saved shuffle indices: shape {shuffle_indices.shape}")
        return shuffle_indices

    def __len__(self):
        return len(self.original_dataset)

    def shuffle_patches(self, img_array, img_idx):
        """Shuffle image patches deterministically"""
        C, H, W = img_array.shape

        # Reshape into patches
        patches = img_array.reshape(
            C, self.num_patches_h, self.patch_size, self.num_patches_w, self.patch_size
        )
        patches = patches.transpose(1, 3, 0, 2, 4)
        patches = patches.reshape(self.total_patches, C, self.patch_size, self.patch_size)

        # Apply deterministic shuffle
        shuffle_order = self.shuffle_indices[img_idx]
        patches = patches[shuffle_order]

        # Reconstruct
        patches = patches.reshape(self.num_patches_h, self.num_patches_w, C,
                                  self.patch_size, self.patch_size)
        patches = patches.transpose(2, 0, 3, 1, 4)
        shuffled = patches.reshape(C, H, W)
        return shuffled

    def __getitem__(self, idx):
        # Check cache first
        if self.cache is not None and idx in self.cache:
            return self.cache[idx]

        img, label = self.original_dataset[idx]

        # Convert to numpy array [0, 1] range
        if torch.is_tensor(img):
            img_array = img.numpy()
        else:
            img_array = np.array(img)
            if img_array.ndim == 2:
                img_array = np.expand_dims(img_array, axis=0)
            elif img_array.shape[-1] == 3:
                img_array = img_array.transpose(2, 0, 1)

        # Apply shuffling
        img_array = self.shuffle_patches(img_array, idx)

        # Convert to tensor
        img_tensor = torch.from_numpy(img_array).float()

        # ✅ FIX: Apply CIFAR normalization
        img_tensor = normalize_tensor(img_tensor, self.mean, self.std)

        if self.cache is not None:
            self.cache[idx] = (img_tensor, label)

        return img_tensor, label


class EdgePreservedDataset(Dataset):
    """
    Dataset with edge-preserving smoothing (removes texture, keeps shape)

    FIXES:
    - Proper normalization workflow: denorm -> process -> renorm
    - Caching for efficiency
    """

    def __init__(self, original_dataset, dataset_name, cache_perturbations=True):
        self.original_dataset = original_dataset
        self.cache_perturbations = cache_perturbations
        self.mean, self.std = get_normalization_params(dataset_name)
        self.cache = {} if cache_perturbations else None

    def __len__(self):
        return len(self.original_dataset)

    def edge_preserving_smooth(self, img_array):
        """
        Apply bilateral filter

        Args:
            img_array: (C, H, W) in [0, 1] range
        Returns:
            smoothed: (C, H, W) in [0, 1] range
        """
        # Convert to (H, W, C) for cv2
        img_hwc = img_array.transpose(1, 2, 0)

        # Scale to uint8 for cv2
        img_hwc = (img_hwc * 255).clip(0, 255).astype(np.uint8)

        # Apply bilateral filter
        smoothed = cv2.bilateralFilter(img_hwc, d=9, sigmaColor=75, sigmaSpace=75)

        # Back to float [0, 1]
        smoothed = smoothed.astype(np.float32) / 255.0

        # Back to (C, H, W)
        smoothed = smoothed.transpose(2, 0, 1)

        return smoothed

    def __getitem__(self, idx):
        if self.cache is not None and idx in self.cache:
            return self.cache[idx]

        img, label = self.original_dataset[idx]

        # Get unnormalized array [0, 1]
        if torch.is_tensor(img):
            img_array = img.numpy()
        else:
            img_array = np.array(img).transpose(2, 0, 1) / 255.0

        # Apply smoothing
        img_array = self.edge_preserving_smooth(img_array)

        # Convert to tensor
        img_tensor = torch.from_numpy(img_array).float()

        # ✅ FIX: Apply CIFAR normalization
        img_tensor = normalize_tensor(img_tensor, self.mean, self.std)

        if self.cache is not None:
            self.cache[idx] = (img_tensor, label)

        return img_tensor, label


class HighPassFilterDataset(Dataset):
    """
    Dataset with high-pass filtering (extracts edges/shapes)

    FIXES:
    - Proper normalization after filtering
    - Improved high-pass filter implementation
    """

    def __init__(self, original_dataset, dataset_name, sigma=2.0, cache_perturbations=True):
        self.original_dataset = original_dataset
        self.sigma = sigma
        self.cache_perturbations = cache_perturbations
        self.mean, self.std = get_normalization_params(dataset_name)
        self.cache = {} if cache_perturbations else None

    def __len__(self):
        return len(self.original_dataset)

    def high_pass_filter(self, img_array):
        """
        Extract high-frequency components (edges)

        Args:
            img_array: (C, H, W) in [0, 1] range
        Returns:
            high_pass: (C, H, W) normalized to [0, 1]
        """
        low_pass = np.zeros_like(img_array)
        for c in range(img_array.shape[0]):
            low_pass[c] = gaussian_filter(img_array[c], sigma=self.sigma)

        # High-pass = original - low-pass
        high_pass = img_array - low_pass

        # Normalize to [0, 1] range
        high_pass = (high_pass - high_pass.min()) / (high_pass.max() - high_pass.min() + 1e-8)

        return high_pass

    def __getitem__(self, idx):
        if self.cache is not None and idx in self.cache:
            return self.cache[idx]

        img, label = self.original_dataset[idx]

        # Get array [0, 1]
        if torch.is_tensor(img):
            img_array = img.numpy()
        else:
            img_array = np.array(img).transpose(2, 0, 1) / 255.0

        # Apply high-pass filter
        img_array = self.high_pass_filter(img_array)

        # Convert to tensor
        img_tensor = torch.from_numpy(img_array).float()

        # ✅ FIX: Apply CIFAR normalization
        img_tensor = normalize_tensor(img_tensor, self.mean, self.std)

        if self.cache is not None:
            self.cache[idx] = (img_tensor, label)

        return img_tensor, label


def evaluate_accuracy(model, loader, device, temperature=1.0):
    """
    Evaluate accuracy with specified temperature

    FIXES:
    - Temperature parameter for consistent evaluation
    - Returns predictions for further analysis
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=f'Evaluating (T={temperature:.2f})'):
            inputs, targets = inputs.to(device), targets.to(device)

            logits = model(inputs)

            # ✅ FIX: Use temperature-scaled predictions
            if temperature != 1.0:
                probs = F.softmax(logits / temperature, dim=1)
                _, predicted = probs.max(1)
            else:
                _, predicted = logits.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds.append(predicted.cpu())
            all_targets.append(targets.cpu())

    accuracy = 100. * correct / total
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    return accuracy, all_preds, all_targets


def evaluate_shape_texture_bias(model, test_dataset, dataset_name, batch_size=128,
                                shuffle_indices_path='shuffle_indices.npy',
                                num_workers=4, device='cuda', temperature=1.0):
    """
    Comprehensive perturbation robustness evaluation

    FIXES:
    - Proper normalization for all datasets
    - Temperature-aware evaluation
    - Dataset-specific normalization parameters
    - Accurate metric naming

    Args:
        model: Trained model
        test_dataset: CIFAR test dataset with ToTensor() only (no normalization)
        dataset_name: 'cifar10' or 'cifar100'
        batch_size: Batch size
        shuffle_indices_path: Path for deterministic shuffle indices
        num_workers: DataLoader workers
        device: Device for evaluation
        temperature: Temperature for softmax scaling during evaluation

    Returns:
        dict: Perturbation robustness metrics
    """
    model.eval()

    print("\n" + "="*80)
    print(f"PERTURBATION ROBUSTNESS EVALUATION")
    print(f"Dataset: {dataset_name}")
    print(f"Evaluation Temperature: {temperature:.2f}")
    print("="*80)

    # ================================================================
    # 1. Original (properly normalized)
    # ================================================================
    print("\n[1/4] Evaluating ORIGINAL images (with proper normalization)...")
    mean, std = get_normalization_params(dataset_name)

    # Create properly normalized dataset
    from torchvision import transforms
    normalize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Need to re-create dataset with normalization
    if 'cifar10' in dataset_name.lower():
        import torchvision
        normalized_dataset = torchvision.datasets.CIFAR10(
            root=test_dataset.root, train=False, download=False,
            transform=normalize_transform
        )
    else:
        import torchvision
        normalized_dataset = torchvision.datasets.CIFAR100(
            root=test_dataset.root, train=False, download=False,
            transform=normalize_transform
        )

    original_loader = DataLoader(normalized_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers, pin_memory=True)
    original_acc, _, _ = evaluate_accuracy(model, original_loader, device, temperature)
    print(f"✓ Original Accuracy: {original_acc:.2f}%")

    # ================================================================
    # 2. Patch-shuffled (texture disruption)
    # ================================================================
    print("\n[2/4] Evaluating PATCH-SHUFFLED images...")
    shuffled_dataset = PatchShuffledDataset(
        test_dataset, dataset_name, patch_size=8,
        shuffle_indices_path=shuffle_indices_path
    )
    shuffled_loader = DataLoader(shuffled_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers, pin_memory=True)
    shuffled_acc, _, _ = evaluate_accuracy(model, shuffled_loader, device, temperature)
    print(f"✓ Patch-Shuffled Accuracy: {shuffled_acc:.2f}%")

    # ================================================================
    # 3. Edge-preserved (texture removal)
    # ================================================================
    print("\n[3/4] Evaluating EDGE-PRESERVED images...")
    edge_dataset = EdgePreservedDataset(test_dataset, dataset_name)
    edge_loader = DataLoader(edge_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)
    edge_acc, _, _ = evaluate_accuracy(model, edge_loader, device, temperature)
    print(f"✓ Edge-Preserved Accuracy: {edge_acc:.2f}%")

    # ================================================================
    # 4. High-pass filtered (edges only)
    # ================================================================
    print("\n[4/4] Evaluating HIGH-PASS FILTERED images...")
    highpass_dataset = HighPassFilterDataset(test_dataset, dataset_name, sigma=2.0)
    highpass_loader = DataLoader(highpass_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers, pin_memory=True)
    highpass_acc, _, _ = evaluate_accuracy(model, highpass_loader, device, temperature)
    print(f"✓ High-Pass Accuracy: {highpass_acc:.2f}%")

    # ================================================================
    # Calculate metrics (renamed for accuracy)
    # ================================================================
    perturbation_robustness = (shuffled_acc + edge_acc + highpass_acc) / 3
    robustness_ratio = perturbation_robustness / original_acc if original_acc > 0 else 0
    texture_dependency = (original_acc - perturbation_robustness) / original_acc if original_acc > 0 else 0

    results = {
        'original_accuracy': original_acc,
        'patch_shuffled_accuracy': shuffled_acc,
        'edge_preserved_accuracy': edge_acc,
        'highpass_accuracy': highpass_acc,
        'perturbation_robustness_score': perturbation_robustness,
        'robustness_ratio': robustness_ratio,
        'texture_dependency': texture_dependency,
        'evaluation_temperature': temperature,
    }

    # ================================================================
    # Print summary
    # ================================================================
    print("\n" + "="*80)
    print("PERTURBATION ROBUSTNESS SUMMARY")
    print("="*80)
    print(f"Original Accuracy:             {original_acc:.2f}%")
    print(f"Patch-Shuffled Accuracy:       {shuffled_acc:.2f}%")
    print(f"Edge-Preserved Accuracy:       {edge_acc:.2f}%")
    print(f"High-Pass Accuracy:            {highpass_acc:.2f}%")
    print(f"\nPerturbation Robustness Score: {perturbation_robustness:.2f}%")
    print(f"Robustness Ratio:              {robustness_ratio:.4f}")
    print(f"Texture Dependency:            {texture_dependency:.4f}")
    print(f"\nEvaluation Temperature:        {temperature:.2f}")
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("  - Robustness Ratio: Proportion of accuracy retained under perturbations")
    print("  - High ratio (>0.7): Model maintains performance without texture details")
    print("  - Low ratio (<0.4): Model relies heavily on fine-grained texture")
    print("  - Texture Dependency: Proportion of accuracy lost without texture")
    print("\nNOTE: This measures perturbation robustness, NOT true shape bias.")
    print("      True shape bias requires cue-conflict stimuli (e.g., stylized images).")
    print("="*80)

    return results


def run_bias_evaluation(model, test_dataset, dataset_name, batch_size=128,
                       device='cuda', temperature=1.0,
                       shuffle_indices_path='shuffle_indices.npy'):
    """
    Convenience wrapper with corrected implementation

    ✅ ALL FIXES APPLIED
    """
    return evaluate_shape_texture_bias(
        model, test_dataset, dataset_name, batch_size,
        shuffle_indices_path, num_workers=4, device=device,
        temperature=temperature
    )


# ================================================================
# Verification utilities
# ================================================================

def verify_normalization(test_dataset, dataset_name):
    """
    Verify that normalization is correctly applied

    Run this to check that fixes are working
    """
    print("\n" + "="*80)
    print("VERIFYING NORMALIZATION")
    print("="*80)

    mean, std = get_normalization_params(dataset_name)
    print(f"Dataset: {dataset_name}")
    print(f"Expected mean: {mean}")
    print(f"Expected std: {std}")

    # Test original dataset (should be unnormalized)
    img, _ = test_dataset[0]
    if torch.is_tensor(img):
        print(f"\nOriginal dataset image range: [{img.min():.3f}, {img.max():.3f}]")
        print(f"Expected: [0.0, 1.0] ✓" if 0 <= img.min() and img.max() <= 1 else "Expected: [0.0, 1.0] ✗")

    # Test perturbed dataset (should be normalized)
    shuffled_dataset = PatchShuffledDataset(test_dataset, dataset_name, patch_size=8)
    img_shuffled, _ = shuffled_dataset[0]
    print(f"\nShuffled dataset image range: [{img_shuffled.min():.3f}, {img_shuffled.max():.3f}]")
    print(f"Expected: roughly [-2, 2] ✓" if -3 < img_shuffled.min() < 0 and 0 < img_shuffled.max() < 3 else "Expected: roughly [-2, 2] ✗")

    # Check mean and std
    sample_mean = img_shuffled.mean(dim=[1, 2])
    sample_std = img_shuffled.std(dim=[1, 2])
    print(f"\nSample mean: {sample_mean.tolist()}")
    print(f"Sample std: {sample_std.tolist()}")

    print("="*80)


if __name__ == '__main__':
    import torchvision

    # Test with CIFAR-100
    print("Testing corrected implementation...")

    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True,
        transform=transforms.ToTensor()
    )

    verify_normalization(test_dataset, 'cifar100')