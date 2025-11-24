"""
Shape vs Texture Bias Evaluation
Implements multiple methods to assess model's reliance on shape vs texture cues
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


class PatchShuffledDataset(Dataset):
    """
    Dataset that creates jigsaw-puzzled versions of images with DETERMINISTIC shuffling

    Key: Shuffle indices are computed once and reused across all runs
    This ensures every model is evaluated on the EXACT SAME shuffled dataset
    """

    def __init__(self, original_dataset, patch_size=4, shuffle_indices_path='shuffle_indices.npy'):
        """
        Args:
            original_dataset: Original CIFAR dataset
            patch_size: Size of patches to shuffle (4x4 for 32x32 images = 64 patches)
            shuffle_indices_path: Path to save/load deterministic shuffle indices
        """
        self.original_dataset = original_dataset
        self.patch_size = patch_size
        self.shuffle_indices_path = shuffle_indices_path

        # Get image dimensions from first sample
        sample_img, _ = original_dataset[0]
        if torch.is_tensor(sample_img):
            C, H, W = sample_img.shape
        else:
            sample_img = np.array(sample_img)
            C, H, W = sample_img.shape

        self.img_shape = (C, H, W)
        self.num_patches_h = H // patch_size
        self.num_patches_w = W // patch_size
        self.total_patches = self.num_patches_h * self.num_patches_w

        # Load or generate deterministic shuffle indices
        self.shuffle_indices = self._get_or_create_shuffle_indices()

    def _get_or_create_shuffle_indices(self):
        """
        Load shuffle indices if they exist, otherwise create and save them
        This ensures the SAME shuffling pattern is used across ALL experiments
        """
        import os

        if os.path.exists(self.shuffle_indices_path):
            print(f"Loading existing shuffle indices from {self.shuffle_indices_path}")
            shuffle_indices = np.load(self.shuffle_indices_path)
            print(f"Loaded shuffle indices shape: {shuffle_indices.shape}")

            # Verify shape matches current configuration
            expected_shape = (len(self.original_dataset), self.total_patches)
            if shuffle_indices.shape != expected_shape:
                raise ValueError(
                    f"Loaded shuffle indices shape {shuffle_indices.shape} "
                    f"doesn't match expected shape {expected_shape}. "
                    f"Delete {self.shuffle_indices_path} and regenerate."
                )
        else:
            print(f"Generating NEW deterministic shuffle indices...")
            print(f"These will be saved to {self.shuffle_indices_path}")
            print(f"Dataset size: {len(self.original_dataset)}, Patches per image: {self.total_patches}")

            # Use a FIXED seed for reproducibility across all experiments
            rng = np.random.RandomState(seed=42)

            # Generate unique shuffle for each image in the dataset
            shuffle_indices = np.zeros((len(self.original_dataset), self.total_patches), dtype=np.int32)

            for i in range(len(self.original_dataset)):
                # Each image gets a unique but deterministic shuffle
                shuffle_indices[i] = rng.permutation(self.total_patches)

            # Save for future use
            np.save(self.shuffle_indices_path, shuffle_indices)
            print(f"Saved shuffle indices to {self.shuffle_indices_path}")
            print(f"Shape: {shuffle_indices.shape}")

        return shuffle_indices

    def __len__(self):
        return len(self.original_dataset)

    def shuffle_patches(self, img_array, img_idx):
        """
        Shuffle image patches using pre-computed deterministic indices

        Args:
            img_array: Image tensor (C, H, W)
            img_idx: Index of image in dataset (determines shuffle pattern)
        """
        C, H, W = img_array.shape
        patch_size = self.patch_size

        # Reshape into patches
        patches = img_array.reshape(
            C, self.num_patches_h, patch_size, self.num_patches_w, patch_size
        )
        patches = patches.transpose(1, 3, 0, 2, 4)  # (num_h, num_w, C, patch_h, patch_w)
        patches = patches.reshape(self.total_patches, C, patch_size, patch_size)

        # Apply pre-computed shuffle for this specific image
        shuffle_order = self.shuffle_indices[img_idx]
        patches = patches[shuffle_order]

        # Reconstruct image
        patches = patches.reshape(self.num_patches_h, self.num_patches_w, C, patch_size, patch_size)
        patches = patches.transpose(2, 0, 3, 1, 4)  # (C, num_h, patch_h, num_w, patch_w)
        shuffled = patches.reshape(C, H, W)

        return shuffled

    def __getitem__(self, idx):
        img, label = self.original_dataset[idx]

        # Convert to numpy if tensor
        if torch.is_tensor(img):
            img_array = img.numpy()
        else:
            img_array = np.array(img)
            if img_array.ndim == 2:  # Grayscale
                img_array = np.expand_dims(img_array, axis=0)
            elif img_array.shape[-1] == 3:  # (H, W, C) -> (C, H, W)
                img_array = img_array.transpose(2, 0, 1)

        # Apply deterministic shuffle
        img_array = self.shuffle_patches(img_array, idx)

        # Convert back to tensor
        img_tensor = torch.from_numpy(img_array).float()

        return img_tensor, label


class EdgePreservedDataset(Dataset):
    """
    Dataset that removes texture while preserving shape using edge-preserving smoothing
    High accuracy = shape bias (recognizes smoothed objects)
    Low accuracy = texture bias (needs texture details)
    """

    def __init__(self, original_dataset, sigma_spatial=15, sigma_range=0.1):
        """
        Args:
            original_dataset: Original CIFAR dataset
            sigma_spatial: Spatial smoothing strength
            sigma_range: Range smoothing strength
        """
        self.original_dataset = original_dataset
        self.sigma_spatial = sigma_spatial
        self.sigma_range = sigma_range

    def __len__(self):
        return len(self.original_dataset)

    def edge_preserving_smooth(self, img_array):
        """Apply bilateral filter to preserve edges while smoothing texture"""
        # img_array shape: (C, H, W)
        # Need to convert to (H, W, C) for cv2
        img_hwc = img_array.transpose(1, 2, 0)

        # Denormalize if normalized (assuming CIFAR normalization)
        img_hwc = (img_hwc * 255).astype(np.uint8)

        # Apply bilateral filter
        smoothed = cv2.bilateralFilter(
            img_hwc,
            d=9,
            sigmaColor=75,
            sigmaSpace=75
        )

        # Normalize back
        smoothed = smoothed.astype(np.float32) / 255.0

        # Convert back to (C, H, W)
        smoothed = smoothed.transpose(2, 0, 1)

        return smoothed

    def __getitem__(self, idx):
        img, label = self.original_dataset[idx]

        # Convert to numpy
        if torch.is_tensor(img):
            img_array = img.numpy()
        else:
            img_array = np.array(img)

        # Apply edge-preserving smoothing
        img_array = self.edge_preserving_smooth(img_array)

        # Convert back to tensor
        img_tensor = torch.from_numpy(img_array).float()

        return img_tensor, label


class HighPassFilterDataset(Dataset):
    """
    Dataset that extracts edges/shapes using high-pass filtering
    High accuracy = shape bias (recognizes edge-only images)
    Low accuracy = texture bias
    """

    def __init__(self, original_dataset, sigma=1.0):
        """
        Args:
            original_dataset: Original CIFAR dataset
            sigma: Gaussian blur sigma for high-pass filter
        """
        self.original_dataset = original_dataset
        self.sigma = sigma

    def __len__(self):
        return len(self.original_dataset)

    def high_pass_filter(self, img_array):
        """Extract high-frequency components (edges/shapes)"""
        # Apply Gaussian blur (low-pass)
        low_pass = np.zeros_like(img_array)
        for c in range(img_array.shape[0]):
            low_pass[c] = gaussian_filter(img_array[c], sigma=self.sigma)

        # High-pass = Original - Low-pass
        high_pass = img_array - low_pass

        # Normalize to [0, 1]
        high_pass = (high_pass - high_pass.min()) / (high_pass.max() - high_pass.min() + 1e-8)

        return high_pass

    def __getitem__(self, idx):
        img, label = self.original_dataset[idx]

        # Convert to numpy
        if torch.is_tensor(img):
            img_array = img.numpy()
        else:
            img_array = np.array(img)

        # Apply high-pass filter
        img_array = self.high_pass_filter(img_array)

        # Convert back to tensor
        img_tensor = torch.from_numpy(img_array).float()

        return img_tensor, label


def evaluate_shape_texture_bias(model, test_dataset, batch_size=128, shuffle_indices_path='shuffle_indices_cifar100.npy',
                                num_workers=4, device='cuda'):
    """
    Comprehensive shape vs texture bias evaluation

    Returns:
        dict: Contains accuracies on original, patch-shuffled, edge-preserved,
              and high-pass filtered images, plus bias metrics
    """
    model.eval()

    # 1. Original accuracy (baseline)
    print("\n" + "=" * 60)
    print("Evaluating on ORIGINAL images...")
    print("=" * 60)
    original_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)
    original_acc = evaluate_accuracy(model, original_loader, device)
    print(f"Original Accuracy: {original_acc:.2f}%")

    # 2. Patch-shuffled (disrupts texture, preserves global statistics)
    print("\n" + "=" * 60)
    print("Evaluating on PATCH-SHUFFLED images (patch_size=4)...")
    print("=" * 60)
    shuffled_dataset = PatchShuffledDataset(
        test_dataset,
        patch_size=4,
        shuffle_indices_path=shuffle_indices_path
    )
    shuffled_loader = DataLoader(shuffled_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=4)
    shuffled_acc = evaluate_accuracy(model, shuffled_loader, device)
    print(f"Patch-Shuffled Accuracy: {shuffled_acc:.2f}%")

    # 3. Edge-preserved (removes texture, keeps shape)
    print("\n" + "=" * 60)
    print("Evaluating on EDGE-PRESERVED images...")
    print("=" * 60)
    edge_dataset = EdgePreservedDataset(test_dataset)
    edge_loader = DataLoader(edge_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    edge_acc = evaluate_accuracy(model, edge_loader, device)
    print(f"Edge-Preserved Accuracy: {edge_acc:.2f}%")

    # 4. High-pass filtered (shape/edge information only)
    print("\n" + "=" * 60)
    print("Evaluating on HIGH-PASS FILTERED images...")
    print("=" * 60)
    highpass_dataset = HighPassFilterDataset(test_dataset, sigma=2.0)
    highpass_loader = DataLoader(highpass_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)
    highpass_acc = evaluate_accuracy(model, highpass_loader, device)
    print(f"High-Pass Filtered Accuracy: {highpass_acc:.2f}%")

    # Calculate bias metrics
    # Shape Bias Score: Average of shape-reliant metrics relative to original
    shape_score = (shuffled_acc + edge_acc + highpass_acc) / 3
    shape_bias_ratio = shape_score / original_acc if original_acc > 0 else 0

    # Texture Dependency: How much accuracy drops without texture
    texture_dependency = (original_acc - shape_score) / original_acc if original_acc > 0 else 0

    results = {
        'original_accuracy': original_acc,
        'patch_shuffled_accuracy': shuffled_acc,
        'edge_preserved_accuracy': edge_acc,
        'highpass_accuracy': highpass_acc,
        'shape_bias_score': shape_score,
        'shape_bias_ratio': shape_bias_ratio,
        'texture_dependency': texture_dependency,
    }

    # Print summary
    print("\n" + "=" * 60)
    print("SHAPE VS TEXTURE BIAS SUMMARY")
    print("=" * 60)
    print(f"Original Accuracy:           {original_acc:.2f}%")
    print(f"Patch-Shuffled Accuracy:     {shuffled_acc:.2f}%")
    print(f"Edge-Preserved Accuracy:     {edge_acc:.2f}%")
    print(f"High-Pass Accuracy:          {highpass_acc:.2f}%")
    print(f"\nShape Bias Score:            {shape_score:.2f}%")
    print(f"Shape Bias Ratio:            {shape_bias_ratio:.4f}")
    print(f"Texture Dependency:          {texture_dependency:.4f}")
    print("\nInterpretation:")
    print(f"  - Shape Bias Ratio > 0.7: Strong shape bias")
    print(f"  - Shape Bias Ratio 0.4-0.7: Moderate shape bias")
    print(f"  - Shape Bias Ratio < 0.4: Strong texture bias")
    print(f"  - Texture Dependency > 0.6: Highly texture-dependent")
    print("=" * 60)

    return results


def evaluate_accuracy(model, loader, device):
    """Helper function to compute accuracy"""
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100. * correct / total


# Example usage function
def run_bias_evaluation(model, test_dataset, batch_size=128, device='cuda',
                       shuffle_indices_path='shuffle_indices_cifar100.npy'):
    """
    Convenience function to run all bias evaluations

    Args:
        model: Trained PyTorch model
        test_dataset: Test dataset (CIFAR-10/100)
        batch_size: Batch size for evaluation
        device: Device to run evaluation on

    Returns:
        dict: Comprehensive bias evaluation results
    """
    return evaluate_shape_texture_bias(
        model, test_dataset, batch_size, shuffle_indices_path, num_workers=4, device=device
    )


def verify_determinism(test_dataset, patch_size=4, num_samples=5):
    """
    Verify that shuffled dataset is deterministic across multiple loads

    Run this once to confirm shuffle indices work correctly
    """
    print("\n" + "=" * 60)
    print("VERIFYING DETERMINISM")
    print("=" * 60)

    # Create dataset twice
    dataset1 = PatchShuffledDataset(test_dataset, patch_size=patch_size)
    dataset2 = PatchShuffledDataset(test_dataset, patch_size=patch_size)

    # Check samples
    all_match = True
    for i in range(num_samples):
        img1, label1 = dataset1[i]
        img2, label2 = dataset2[i]

        if not torch.equal(img1, img2):
            print(f"❌ Sample {i}: Images DON'T match!")
            all_match = False
        else:
            print(f"✓ Sample {i}: Images match perfectly")

    if all_match:
        print("\n✓✓✓ SUCCESS: Dataset is fully deterministic!")
    else:
        print("\n❌❌❌ FAILURE: Dataset has randomness!")

    print("=" * 60)

    return all_match


if __name__ == '__main__':
    import torchvision
    import torchvision.transforms as transforms

    # Load test dataset
    test_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=test_transform
    )

    # Verify determinism
    verify_determinism(test_dataset)