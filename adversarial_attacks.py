"""
Adversarial Attack Implementations with ECE Calibration Metrics
Supports PGD (Projected Gradient Descent) and C&W (Carlini-Wagner) attacks

Updated to follow the methodology from:
"Exploring the Impact of Temperature Scaling in Softmax for Classification and Adversarial Robustness"
(arXiv:2502.20604)

Key changes:
1. Attack generation ALWAYS uses T=1 (no temperature attenuation)
2. Added ECE calculations for both clean and adversarial examples
3. Evaluation uses T=1 by default for paper-comparable results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


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


class PGDAttack:
    """
    Projected Gradient Descent (PGD) Attack

    Reference:
    Madry et al. "Towards Deep Learning Models Resistant to Adversarial Attacks"
    ICLR 2018

    Note: Per the temperature scaling paper (arXiv:2502.20604), attacks are
    ALWAYS generated with T=1 to ensure gradients are not attenuated.
    """

    def __init__(self, model, epsilon=8 / 255, alpha=2 / 255, num_iter=20,
                 device='cuda'):
        """
        Args:
            model: Target model to attack
            epsilon: Maximum perturbation (L_infinity bound)
            alpha: Step size per iteration
            num_iter: Number of iterations
            device: Device to run on

        Note: Temperature parameter removed - attacks always use T=1
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.device = device

    def generate(self, images, labels):
        """
        Generate adversarial examples using PGD

        Args:
            images: Clean images (B, C, H, W)
            labels: True labels (B,)

        Returns:
            Adversarial images (B, C, H, W)
        """
        images = images.to(self.device)
        labels = labels.to(self.device)

        # Initialize perturbation with random noise
        delta = torch.zeros_like(images).uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad = True

        for i in range(self.num_iter):
            # Forward pass
            adv_images = images + delta
            logits = self.model(adv_images)

            # CRITICAL: Always use T=1 for attack generation
            # This ensures attack gradients are not attenuated
            loss = F.cross_entropy(logits, labels)

            # Backward pass
            loss.backward()

            # Update perturbation
            grad = delta.grad.detach()
            delta.data = delta.data + self.alpha * grad.sign()

            # Project to epsilon ball
            delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)

            # Project to valid image range [0, 1]
            delta.data = torch.clamp(images.data + delta.data, 0, 1) - images.data

            # Zero gradients
            delta.grad.zero_()

        return (images + delta).detach()


class CWAttack:
    """
    Carlini-Wagner (C&W) L_infinity Attack

    Reference:
    Carlini & Wagner "Towards Evaluating the Robustness of Neural Networks"
    IEEE S&P 2017

    Note: This is a simplified L_inf version for efficiency.
    Attacks always use T=1 per the temperature scaling paper methodology.
    """

    def __init__(self, model, epsilon=8 / 255, c=1.0, kappa=0, num_iter=100,
                 learning_rate=0.01, device='cuda'):
        """
        Args:
            model: Target model to attack
            epsilon: Maximum perturbation (L_infinity bound)
            c: Trade-off parameter between adversarial loss and perturbation
            kappa: Confidence parameter (margin)
            num_iter: Number of optimization iterations
            learning_rate: Learning rate for optimization
            device: Device to run on

        Note: Temperature parameter removed - attacks always use T=1
        """
        self.model = model
        self.epsilon = epsilon
        self.c = c
        self.kappa = kappa
        self.num_iter = num_iter
        self.learning_rate = learning_rate
        self.device = device

    def generate(self, images, labels):
        """
        Generate adversarial examples using C&W attack

        Args:
            images: Clean images (B, C, H, W)
            labels: True labels (B,)

        Returns:
            Adversarial images (B, C, H, W)
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        batch_size = images.shape[0]

        # Initialize perturbation
        delta = torch.zeros_like(images, requires_grad=True)

        # Optimizer for delta
        optimizer = torch.optim.Adam([delta], lr=self.learning_rate)

        for i in range(self.num_iter):
            # Get adversarial images
            adv_images = torch.clamp(images + delta, 0, 1)

            # Forward pass - ALWAYS use T=1 (no temperature scaling)
            logits = self.model(adv_images)

            # C&W loss: maximize the difference between target class and other classes
            # f(x) = max(max{Z(x)_i : i ≠ y} - Z(x)_y, -κ)

            # Get true class logits
            true_logits = logits.gather(1, labels.unsqueeze(1))

            # Get max logit among other classes
            mask = torch.ones_like(logits).scatter_(1, labels.unsqueeze(1), 0.0)
            other_logits = (logits * mask + (1 - mask) * -1e9).max(1, keepdim=True)[0]

            # C&W loss (we want to maximize this, so minimize negative)
            adv_loss = torch.clamp(other_logits - true_logits + self.kappa, min=0)

            # L_infinity constraint (we want to minimize perturbation)
            perturbation_loss = torch.max(torch.abs(delta))

            # Combined loss
            loss = adv_loss.mean() + self.c * perturbation_loss

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Project to epsilon ball
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
                delta.data = torch.clamp(images + delta.data, 0, 1) - images

        return torch.clamp(images + delta, 0, 1).detach()


def evaluate_adversarial_robustness(model, test_loader, attack_type='pgd',
                                    epsilon=8 / 255,
                                    batch_size=128, num_workers=4, device='cuda'):
    """
    Evaluate model robustness against adversarial attacks WITH ECE metrics

    Per the temperature scaling paper (arXiv:2502.20604):
    - All attacks are generated with T=1
    - All evaluations (clean and adversarial) use T=1

    This ensures attack gradients are not attenuated and reflects
    the model's true sensitivity to perturbations.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        attack_type: 'pgd' or 'cw'
        epsilon: Maximum perturbation
        batch_size: Batch size
        num_workers: Number of workers
        device: Device

    Returns:
        dict: {
            'clean_accuracy': float,
            'clean_ece': float,
            'adversarial_accuracy': float,
            'adversarial_ece': float,
            'attack_success_rate': float,
            'confidence_drop': float  # How much confidence drops under attack
        }
    """
    model.eval()

    # Create attack (no temperature parameter - always T=1)
    if attack_type.lower() == 'pgd':
        attack = PGDAttack(
            model=model,
            epsilon=epsilon,
            alpha=2 / 255,  # Standard: epsilon/4
            num_iter=20,
            device=device
        )
    elif attack_type.lower() == 'cw':
        attack = CWAttack(
            model=model,
            epsilon=epsilon,
            c=1.0,
            kappa=0,
            num_iter=100,  # C&W needs more iterations
            learning_rate=0.01,
            device=device
        )
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

    # Storage for ECE computation
    clean_probs_list = []
    clean_targets_list = []
    adv_probs_list = []
    adv_targets_list = []

    clean_correct = 0
    total = 0

    print(f"\nRunning {attack_type.upper()} attack (ε={epsilon:.4f}, T=1.0)...")

    # First pass: evaluate clean accuracy and collect probabilities
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Clean evaluation'):
            images, labels = images.to(device), labels.to(device)

            # Always use T=1 for evaluation
            logits = model(images)
            probs = F.softmax(logits, dim=1)

            _, predicted = logits.max(1)
            clean_correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            # Store for ECE
            clean_probs_list.append(probs.cpu())
            clean_targets_list.append(labels.cpu())

    clean_accuracy = 100.0 * clean_correct / total

    # Compute clean ECE
    clean_probs = torch.cat(clean_probs_list)
    clean_targets = torch.cat(clean_targets_list)
    clean_ece = compute_ece(clean_probs, clean_targets)
    clean_mean_confidence = clean_probs.max(dim=1)[0].mean().item()

    # Second pass: generate adversarial examples and evaluate
    adv_correct = 0
    total = 0

    for images, labels in tqdm(test_loader, desc=f'{attack_type.upper()} attack'):
        images, labels = images.to(device), labels.to(device)

        # Generate adversarial examples (always with T=1)
        adv_images = attack.generate(images, labels)

        # Evaluate on adversarial examples (always with T=1)
        with torch.no_grad():
            logits = model(adv_images)
            probs = F.softmax(logits, dim=1)

            _, predicted = logits.max(1)
            adv_correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            # Store for ECE
            adv_probs_list.append(probs.cpu())
            adv_targets_list.append(labels.cpu())

    adv_accuracy = 100.0 * adv_correct / total

    # Compute adversarial ECE
    adv_probs = torch.cat(adv_probs_list)
    adv_targets = torch.cat(adv_targets_list)
    adv_ece = compute_ece(adv_probs, adv_targets)
    adv_mean_confidence = adv_probs.max(dim=1)[0].mean().item()

    # Compute attack success rate
    attack_success_rate = 100.0 * (1 - adv_accuracy / clean_accuracy) if clean_accuracy > 0 else 100.0

    # Compute confidence drop (how much confidence decreases under attack)
    confidence_drop = clean_mean_confidence - adv_mean_confidence

    results = {
        'clean_accuracy': clean_accuracy,
        'clean_ece': clean_ece,
        'clean_mean_confidence': clean_mean_confidence * 100,  # As percentage
        'adversarial_accuracy': adv_accuracy,
        'adversarial_ece': adv_ece,
        'adversarial_mean_confidence': adv_mean_confidence * 100,  # As percentage
        'attack_success_rate': attack_success_rate,
        'confidence_drop': confidence_drop * 100,  # As percentage points
        'ece_increase': adv_ece - clean_ece,  # How much ECE worsens under attack
    }

    return results


def evaluate_both_attacks(model, dataset, data_root, batch_size=128,
                          num_workers=4, device='cuda', epsilon=8 / 255):
    """
    Evaluate model on both PGD and C&W attacks with ECE metrics

    Per the temperature scaling paper (arXiv:2502.20604):
    - All attacks generated with T=1
    - All evaluations use T=1

    Args:
        model: Model to evaluate
        dataset: Dataset name ('cifar10' or 'cifar100')
        data_root: Data root directory
        batch_size: Batch size
        num_workers: Number of workers
        device: Device
        epsilon: Maximum perturbation

    Returns:
        dict: Results for both attacks including ECE metrics
    """
    import torchvision
    import torchvision.transforms as transforms

    # Load test dataset
    if dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        dataset_class = torchvision.datasets.CIFAR10
    elif dataset == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        dataset_class = torchvision.datasets.CIFAR100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_dataset = dataset_class(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    results = {}

    # Evaluate PGD attack
    print("\n" + "=" * 70)
    print(f"PGD-20 Attack Evaluation (ε={epsilon:.4f}, T=1.0)")
    print("=" * 70)
    pgd_results = evaluate_adversarial_robustness(
        model, test_loader, attack_type='pgd',
        epsilon=epsilon, batch_size=batch_size,
        num_workers=num_workers, device=device
    )
    results['pgd'] = pgd_results

    print(f"\nPGD-20 Results:")
    print(f"  Clean Accuracy:           {pgd_results['clean_accuracy']:.2f}%")
    print(f"  Clean ECE:                {pgd_results['clean_ece']:.4f}")
    print(f"  Clean Mean Confidence:    {pgd_results['clean_mean_confidence']:.2f}%")
    print(f"  Adversarial Accuracy:     {pgd_results['adversarial_accuracy']:.2f}%")
    print(f"  Adversarial ECE:          {pgd_results['adversarial_ece']:.4f}")
    print(f"  Adversarial Mean Conf:    {pgd_results['adversarial_mean_confidence']:.2f}%")
    print(f"  Attack Success Rate:      {pgd_results['attack_success_rate']:.2f}%")
    print(f"  Confidence Drop:          {pgd_results['confidence_drop']:.2f}pp")
    print(f"  ECE Increase:             {pgd_results['ece_increase']:.4f}")

    # Evaluate C&W attack
    print("\n" + "=" * 70)
    print(f"C&W Attack Evaluation (ε={epsilon:.4f}, T=1.0)")
    print("=" * 70)
    cw_results = evaluate_adversarial_robustness(
        model, test_loader, attack_type='cw',
        epsilon=epsilon, batch_size=batch_size,
        num_workers=num_workers, device=device
    )
    results['cw'] = cw_results

    print(f"\nC&W Results:")
    print(f"  Clean Accuracy:           {cw_results['clean_accuracy']:.2f}%")
    print(f"  Clean ECE:                {cw_results['clean_ece']:.4f}")
    print(f"  Clean Mean Confidence:    {cw_results['clean_mean_confidence']:.2f}%")
    print(f"  Adversarial Accuracy:     {cw_results['adversarial_accuracy']:.2f}%")
    print(f"  Adversarial ECE:          {cw_results['adversarial_ece']:.4f}")
    print(f"  Adversarial Mean Conf:    {cw_results['adversarial_mean_confidence']:.2f}%")
    print(f"  Attack Success Rate:      {cw_results['attack_success_rate']:.2f}%")
    print(f"  Confidence Drop:          {cw_results['confidence_drop']:.2f}pp")
    print(f"  ECE Increase:             {cw_results['ece_increase']:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("ADVERSARIAL ROBUSTNESS & CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"Epsilon: {epsilon:.4f} ({int(epsilon * 255)}/255)")
    print(f"Evaluation Temperature: T=1.0 (per paper methodology)")

    print(f"\n{'Metric':<25} {'PGD-20':<15} {'C&W':<15}")
    print("-" * 55)
    print(f"{'Clean Accuracy (%)':<25} {pgd_results['clean_accuracy']:<15.2f} {cw_results['clean_accuracy']:<15.2f}")
    print(f"{'Clean ECE':<25} {pgd_results['clean_ece']:<15.4f} {cw_results['clean_ece']:<15.4f}")
    print(f"{'Adversarial Acc (%)':<25} {pgd_results['adversarial_accuracy']:<15.2f} {cw_results['adversarial_accuracy']:<15.2f}")
    print(f"{'Adversarial ECE':<25} {pgd_results['adversarial_ece']:<15.4f} {cw_results['adversarial_ece']:<15.4f}")
    print(f"{'Attack Success (%)':<25} {pgd_results['attack_success_rate']:<15.2f} {cw_results['attack_success_rate']:<15.2f}")
    print(f"{'Confidence Drop (pp)':<25} {pgd_results['confidence_drop']:<15.2f} {cw_results['confidence_drop']:<15.2f}")
    print(f"{'ECE Increase':<25} {pgd_results['ece_increase']:<15.4f} {cw_results['ece_increase']:<15.4f}")
    print("=" * 70)

    print("\nINTERPRETATION:")
    print("  - ECE Increase > 0: Model becomes LESS calibrated under attack")
    print("  - ECE Increase < 0: Model becomes MORE calibrated under attack (rare)")
    print("  - High Confidence Drop + High ECE Increase: Model is overconfident on wrong predictions")
    print("  - Low Confidence Drop + Low ECE: Model maintains calibration under attack (good)")
    print("=" * 70)

    return results


if __name__ == '__main__':
    """Test adversarial attacks with ECE"""
    import torchvision
    from models import get_model

    # Load a simple model for testing
    model = get_model('resnet20', num_classes=10).cuda()

    # Load CIFAR-10
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                         [0.2023, 0.1994, 0.2010])
    ])

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False
    )

    # Test PGD attack
    print("Testing PGD attack with ECE...")
    pgd_attack = PGDAttack(model, epsilon=8 / 255, device='cuda')

    images, labels = next(iter(test_loader))
    adv_images = pgd_attack.generate(images, labels)

    print(f"Original image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Adversarial image range: [{adv_images.min():.3f}, {adv_images.max():.3f}]")
    print(f"Perturbation: {(adv_images - images.cuda()).abs().max():.4f}")

    # Quick ECE test
    with torch.no_grad():
        clean_logits = model(images.cuda())
        adv_logits = model(adv_images)

        clean_probs = F.softmax(clean_logits, dim=1)
        adv_probs = F.softmax(adv_logits, dim=1)

        clean_ece = compute_ece(clean_probs.cpu(), labels)
        adv_ece = compute_ece(adv_probs.cpu(), labels)

        print(f"\nClean ECE: {clean_ece:.4f}")
        print(f"Adversarial ECE: {adv_ece:.4f}")
        print(f"ECE Increase: {adv_ece - clean_ece:.4f}")

    print("\nTests passed!")