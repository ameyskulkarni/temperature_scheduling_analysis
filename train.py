"""
Temperature Scheduling Training Script
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import wandb
from tqdm import tqdm

from models import get_model
from lr_scheduler import get_lr_scheduler
from temperature_schedulers import get_temperature_scheduler
from utils import compute_ece, evaluate_on_corruptions
from shape_texture_bias import run_bias_evaluation
from adversarial_attacks import evaluate_both_attacks


def parse_args():
    parser = argparse.ArgumentParser(description='Temperature Scheduling Training')

    # Required
    parser.add_argument('--exp-name', type=str, required=True,
                        help='Experiment name for W&B logging')

    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100'],
                        help='Dataset to use')

    parser.add_argument('--data-root', type=str, default='./data',
                        help='Path to dataset root')

    # Model
    parser.add_argument('--model', type=str, default='resnet32',
                        choices=['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet50_cifar', 'resnet110', 'resnet164', 'wrn28_10'],
                        help='Model architecture')

    # Training
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')

    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')

    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate (constant)')
    parser.add_argument('--lr-schedule', type=str, default='constant',
                        choices=['constant', 'cosine', 'cosine_warmup'],
                        help='Learning rate schedule type')
    parser.add_argument('--lr-min', type=float, default=0.0,
                        help='Minimum learning rate for cosine annealing')

    parser.add_argument('--warmup-epochs', type=int, default=10,
                        help='Number of warmup epochs for cosine_warmup schedule')

    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay')

    # Temperature
    parser.add_argument('--temp-schedule', type=str, default='constant',
                        choices=['constant', 'cosine', 'linear', 'reverse_cosine', 'cyclic_sinusoidal', 'cyclic_triangular'],
                        help='Temperature schedule type')
    parser.add_argument('--temp-max', type=float, default=1.0,
                        help='Maximum/initial temperature')
    parser.add_argument('--temp-min', type=float, default=1.0,
                        help='Minimum/final temperature')
    parser.add_argument('--temp-cycle-length', type=int, default=40,
                        help='Cycle length (epochs) for cyclic schedules')

    # Evaluation
    parser.add_argument('--eval-corruptions', action='store_true',
                        help='Evaluate on CIFAR-C for best checkpoint')

    parser.add_argument('--eval-shape-texture', action='store_true',
                        help='Evaluate shape vs texture bias on best checkpoint')

    parser.add_argument('--eval-adversarial', action='store_true',
                        help='Evaluate adversarial robustness (PGD-20 and C&W) on best checkpoint')

    parser.add_argument('--adv-epsilon', type=float, default=8 / 255,
                        help='Adversarial perturbation budget (default: 8/255)')

    # System
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    return parser.parse_args()


def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_data_loaders(dataset_name, data_root, batch_size, num_workers, generator):
    """Load CIFAR dataset"""

    # Normalization
    if dataset_name == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        num_classes = 10
        dataset_class = torchvision.datasets.CIFAR10
    elif dataset_name == 'cifar100':  # cifar100
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        num_classes = 100
        dataset_class = torchvision.datasets.CIFAR100

    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Datasets
    train_dataset = dataset_class(root=data_root, train=True, download=True,
                                   transform=train_transform)
    test_dataset = dataset_class(root=data_root, train=False, download=True,
                                  transform=test_transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True,
                             worker_init_fn=seed_worker, generator=generator)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, num_classes


def temperature_scaled_loss(logits, targets, temperature):
    """Cross-entropy with temperature scaling"""
    scaled_logits = logits / temperature
    return nn.CrossEntropyLoss()(scaled_logits, targets)


def train_epoch(model, train_loader, optimizer, temperature, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    grad_norms = []

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = temperature_scaled_loss(logits, targets, temperature)
        loss.backward()

        grad_norm = compute_gradient_norm(model)
        grad_norms.append(grad_norm)

        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}',
                         'acc': f'{100.*correct/total:.2f}%',
                         'T': f'{temperature:.2f}'})

        mean_grad_norm = sum(grad_norms) / len(grad_norms)

    return total_loss / total, 100. * correct / total, mean_grad_norm


def validate(model, test_loader, temperature, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            logits = model(inputs)
            loss = temperature_scaled_loss(logits, targets, temperature)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            probs = torch.softmax(logits / temperature, dim=1)
            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())

    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)

    return total_loss / total, 100. * correct / total, all_probs, all_targets

def compute_gradient_norm(model):
    """Compute L2 norm of gradients"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set seed
    generator = set_seed(args.seed)

    # Initialize W&B
    wandb.init(project="temperature-scheduling-resnet50-cifar", name=args.exp_name, config=vars(args))

    # create_ckpt_directory
    save_dir = f'models/{args.exp_name}'
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    train_loader, test_loader, num_classes = get_data_loaders(
        args.dataset, args.data_root, args.batch_size, args.num_workers, generator
    )

    print(f'Dataset: {args.dataset} ({num_classes} classes)')
    print(f'Model: {args.model}')
    print(f'Device: {device}')

    # Create model
    model = get_model(args.model, num_classes=num_classes).to(device)

    # Optimizer (constant LR)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                         momentum=args.momentum, weight_decay=args.weight_decay)

    # Learning rate scheduler
    lr_scheduler = get_lr_scheduler(
        args.lr_schedule,
        lr_max=args.lr,
        lr_min=args.lr_min,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs
    )

    # Temperature scheduler
    temp_kwargs = {}
    if args.temp_schedule == 'cyclic_sinusoidal' or args.temp_schedule == 'cyclic_triangular':
        temp_kwargs = {'cycle_length': args.temp_cycle_length}
    temp_scheduler = get_temperature_scheduler(
        args.temp_schedule, T_max=args.temp_max, T_min=args.temp_min,
        total_epochs=args.epochs, **temp_kwargs
    )

    # Training loop
    best_acc = 0.0
    best_epoch = 0
    for epoch in range(args.epochs):
        current_temp = temp_scheduler.get_temperature(epoch)

        current_lr = lr_scheduler.get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Train
        train_loss, train_acc, grad_norm = train_epoch(
            model, train_loader, optimizer, current_temp, device, epoch
        )

        # Validate
        val_loss, val_acc, val_probs, val_targets = validate(
            model, test_loader, current_temp, device
        )

        # Compute ECE every 10 epochs
        val_ece = None
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            val_ece = compute_ece(val_probs, val_targets)

        # Log to W&B
        log_dict = {
            'hyperparams/epoch': epoch,
            'hyperparams/temperature': current_temp,
            'hyperparams/learning_rate': current_lr,
            'train/loss': train_loss,
            'train/accuracy': train_acc,
            'train/grad_norm': grad_norm,
            'val/loss': val_loss,
            'val/accuracy': val_acc,
        }
        if val_ece is not None:
            log_dict['val/ece'] = val_ece

        wandb.log(log_dict)

        # Print summary
        print(f'Epoch {epoch}: LR={current_lr:.6f}, T={current_temp:.2f}, Train Acc={train_acc:.2f}%, '
              f'Val Acc={val_acc:.2f}%', end='')
        if val_ece is not None:
            print(f', ECE={val_ece:.4f}')
        else:
            print()

        # Save best model with training temperature
        if val_acc > best_acc:
            best_epoch = epoch
            best_acc = val_acc
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'train_temperature': current_temp,  # Save the temperature used during training
                'val_accuracy': val_acc,
            }
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))

        # Save model checkpoint every few epochs
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'train_temperature': current_temp,  # Save the temperature used during training
                'val_accuracy': val_acc,
            }
            torch.save(checkpoint, f'models/{args.exp_name}/model_{epoch}.pth')
            print(f'Checkpoint saved at epoch {epoch}.')

        # Save model checkpoint every few epochs
        if epoch > 159 and epoch % 2 == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'train_temperature': current_temp,  # Save the temperature used during training
                'val_accuracy': val_acc,
            }
            torch.save(checkpoint, f'models/{args.exp_name}/model_{epoch}.pth')
            print(f'Checkpoint saved at epoch {epoch}.')

    print(f'\nTraining complete! Best accuracy: {best_acc:.2f}% (epoch {best_epoch})')

    # Evaluate best model
    print('\n' + '='*60)
    print('Evaluating best checkpoint...')
    print('='*60)

    # Load checkpoint
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    train_temp = checkpoint['train_temperature']

    print(f'Best checkpoint from epoch {checkpoint["epoch"]}')
    print(f'Training temperature at that epoch: {train_temp:.2f}')

    # ================================================================
    # Evaluation 1: With training temperature (T_train)
    # ================================================================
    print(f'\n--- Evaluation with Training Temperature (T={train_temp:.2f}) ---')

    test_loss_Ttrain, test_acc_Ttrain, test_probs_Ttrain, test_targets = validate(
        model, test_loader, train_temp, device
    )
    test_ece_Ttrain = compute_ece(test_probs_Ttrain, test_targets)

    print(f'Test Accuracy (T={train_temp:.2f}): {test_acc_Ttrain:.2f}%')
    print(f'Test ECE (T={train_temp:.2f}): {test_ece_Ttrain:.4f}')

    # ================================================================
    # Evaluation 2: With standard temperature (T=1.0)
    # ================================================================
    print(f'\n--- Evaluation with Standard Temperature (T=1.0) ---')

    test_loss_T1, test_acc_T1, test_probs_T1, test_targets = validate(
        model, test_loader, 1.0, device
    )
    test_ece_T1 = compute_ece(test_probs_T1, test_targets)

    print(f'Test Accuracy (T=1.0): {test_acc_T1:.2f}%')
    print(f'Test ECE (T=1.0): {test_ece_T1:.4f}')

    # ================================================================
    # Comparison
    # ================================================================
    print(f'\n--- Comparison ---')
    acc_diff = test_acc_T1 - test_acc_Ttrain
    ece_diff = test_ece_T1 - test_ece_Ttrain
    print(f'Accuracy difference (T=1.0 - T={train_temp:.2f}): {acc_diff:+.2f}%')
    print(f'ECE difference (T=1.0 - T={train_temp:.2f}): {ece_diff:+.4f}')

    # Log final metrics to W&B
    final_metrics = {
        # Metadata
        'hyperparams/final_train_temperature': train_temp,
        'hyperparams/best_epoch': checkpoint['epoch'],
        'test/best_val_accuracy_Ttrain': best_acc,

        # Evaluation with training temperature
        'test/test_accuracy_Ttrain': test_acc_Ttrain,
        'test/test_ece_Ttrain': test_ece_Ttrain,

        # Evaluation with T=1.0 (standard comparison)
        'test/test_accuracy_T1': test_acc_T1,
        'test/test_ece_T1': test_ece_T1,

        # Differences
        'test/accuracy_diff_T1_vs_Ttrain': acc_diff,
        'test/ece_diff_T1_vs_Ttrain': ece_diff,
    }

    # ================================================================
    # Corruption evaluation (if requested)
    # ================================================================
    if args.eval_corruptions:
        print('\n' + '='*60)
        print('Evaluating on corruptions...')
        print('='*60)

        # Evaluate with training temperature
        print(f'\n--- Corruptions with Training Temperature (T={train_temp:.2f}) ---')
        corruption_results_Ttrain = evaluate_on_corruptions(
            model, args.dataset, args.data_root, args.batch_size,
            args.num_workers, device, temperature=train_temp
        )

        print(f'Mean Corruption Accuracy (T={train_temp:.2f}): {corruption_results_Ttrain["mean_acc"]:.2f}%')
        print(f'Mean Corruption ECE (T={train_temp:.2f}): {corruption_results_Ttrain["mean_ece"]:.4f}')

        # Evaluate with T=1.0
        print(f'\n--- Corruptions with Standard Temperature (T=1.0) ---')
        corruption_results_T1 = evaluate_on_corruptions(
            model, args.dataset, args.data_root, args.batch_size,
            args.num_workers, device, temperature=1.0
        )

        print(f'Mean Corruption Accuracy (T=1.0): {corruption_results_T1["mean_acc"]:.2f}%')
        print(f'Mean Corruption ECE (T=1.0): {corruption_results_T1["mean_ece"]:.4f}')

        # Comparison
        corr_acc_diff = corruption_results_T1['mean_acc'] - corruption_results_Ttrain['mean_acc']
        corr_ece_diff = corruption_results_T1['mean_ece'] - corruption_results_Ttrain['mean_ece']

        print(f'\n--- Corruption Comparison ---')
        print(f'Accuracy difference (T=1.0 - T={train_temp:.2f}): {corr_acc_diff:+.2f}%')
        print(f'ECE difference (T=1.0 - T={train_temp:.2f}): {corr_ece_diff:+.4f}')

        # Add to metrics
        final_metrics.update({
            # Corruptions with training temperature
            'corruption/corruption_accuracy_Ttrain': corruption_results_Ttrain['mean_acc'],
            'corruption/corruption_ece_Ttrain': corruption_results_Ttrain['mean_ece'],

            # Corruptions with T=1.0
            'corruption/corruption_accuracy_T1': corruption_results_T1['mean_acc'],
            'corruption/corruption_ece_T1': corruption_results_T1['mean_ece'],

            # Differences
            'corruption/corruption_accuracy_diff_T1_vs_Ttrain': corr_acc_diff,
            'corruption/corruption_ece_diff_T1_vs_Ttrain': corr_ece_diff,
        })

        # Per-corruption results with both temperatures
        for corruption in corruption_results_Ttrain['per_corruption'].keys():
            if corruption in corruption_results_T1['per_corruption']:
                # Training temperature
                final_metrics[f'corruption_Ttrain/{corruption}_acc'] = \
                    corruption_results_Ttrain['per_corruption'][corruption]['accuracy']
                final_metrics[f'corruption_Ttrain/{corruption}_ece'] = \
                    corruption_results_Ttrain['per_corruption'][corruption]['ece']

                # T=1.0
                final_metrics[f'corruption_T1/{corruption}_acc'] = \
                    corruption_results_T1['per_corruption'][corruption]['accuracy']
                final_metrics[f'corruption_T1/{corruption}_ece'] = \
                    corruption_results_T1['per_corruption'][corruption]['ece']

    # ================================================================
    # Shape vs Texture evaluation (if requested)
    # ================================================================

    if args.eval_shape_texture:
        print('\n' + '=' * 60)
        print('Evaluating Perturbation Robustness (Shape/Texture Proxy)...')
        print('=' * 60)

        indices_path = 'shuffle_indices_cifar100.npy'

        print(f"\n--- Evaluating with Training Temperature (T={train_temp:.2f}) ---")
        bias_results_Ttrain = run_bias_evaluation(
            model, args.dataset, args.data_root, args.batch_size * 2, device,
            temperature=train_temp,  # ✅ Use training temperature
            shuffle_indices_path=indices_path
        )
        patch_shuffled_ECE_Ttrain = compute_ece(bias_results_Ttrain['shuffled_all_probs'], bias_results_Ttrain['shuffled_all_targets'])

        print(f"\n--- Evaluating with Standard Temperature (T=1.0) ---")
        bias_results_T1 = run_bias_evaluation(
            model, args.dataset, args.data_root, args.batch_size * 2, device,
            temperature=1.0,  # ✅ Standard temperature
            shuffle_indices_path=indices_path
        )
        patch_shuffled_ECE_T1 = compute_ece(bias_results_T1['shuffled_all_probs'],
                                                bias_results_T1['shuffled_all_targets'])

        final_metrics.update({
            # Training temperature results
            'bias_Ttrain/original_accuracy': bias_results_Ttrain['original_accuracy'],
            'bias_Ttrain/patch_shuffled_accuracy': bias_results_Ttrain['patch_shuffled_accuracy'],
            'bias_Ttrain/patch_shuffled_ECE_Ttrain': patch_shuffled_ECE_Ttrain,

            # T=1.0 results
            'bias_T1/original_accuracy': bias_results_T1['original_accuracy'],
            'bias_T1/patch_shuffled_accuracy': bias_results_T1['patch_shuffled_accuracy'],
            'bias_T1/patch_shuffled_ECE_T1': patch_shuffled_ECE_T1,

        })

        # Print comparison
        print(f"\n{'Metric':<30} {'T={train_temp:.2f}':<15} {'T=1.0':<15} {'Diff':<15}")
        print('-' * 75)

    # ================================================================
    # Adversarial robustness evaluation (if requested)
    # ================================================================

    if args.eval_adversarial:
        print('\n' + '=' * 70)
        print('Evaluating Adversarial Robustness with Calibration Metrics...')
        print('=' * 70)
        print('\nNote: Per the temperature scaling paper (arXiv:2502.20604),')
        print('all attacks are generated and evaluated with T=1.0')



        # Evaluate with T=1.0 only (per paper methodology)
        # Attacks MUST use T=1 for proper comparison
        adv_results = evaluate_both_attacks(
            model, args.dataset, args.data_root, args.batch_size,
            args.num_workers, device, epsilon=args.adv_epsilon
        )

        # Extract results
        pgd = adv_results['pgd']
        cw = adv_results['cw']

        # Print detailed results
        print(f'\n' + '=' * 70)
        print('DETAILED ADVERSARIAL RESULTS')
        print('=' * 70)

        print(f'\nPGD-20 Attack (ε={args.adv_epsilon:.4f}):')
        print(f'  Clean Accuracy:           {pgd["clean_accuracy"]:.2f}%')
        print(f'  Clean ECE:                {pgd["clean_ece"]:.4f}')
        print(f'  Clean Mean Confidence:    {pgd["clean_mean_confidence"]:.2f}%')
        print(f'  Adversarial Accuracy:     {pgd["adversarial_accuracy"]:.2f}%')
        print(f'  Adversarial ECE:          {pgd["adversarial_ece"]:.4f}')
        print(f'  Adversarial Mean Conf:    {pgd["adversarial_mean_confidence"]:.2f}%')
        print(f'  Attack Success Rate:      {pgd["attack_success_rate"]:.2f}%')
        print(f'  Confidence Drop:          {pgd["confidence_drop"]:.2f}pp')
        print(f'  ECE Increase:             {pgd["ece_increase"]:.4f}')

        print(f'\nC&W Attack (ε={args.adv_epsilon:.4f}):')
        print(f'  Clean Accuracy:           {cw["clean_accuracy"]:.2f}%')
        print(f'  Clean ECE:                {cw["clean_ece"]:.4f}')
        print(f'  Clean Mean Confidence:    {cw["clean_mean_confidence"]:.2f}%')
        print(f'  Adversarial Accuracy:     {cw["adversarial_accuracy"]:.2f}%')
        print(f'  Adversarial ECE:          {cw["adversarial_ece"]:.4f}')
        print(f'  Adversarial Mean Conf:    {cw["adversarial_mean_confidence"]:.2f}%')
        print(f'  Attack Success Rate:      {cw["attack_success_rate"]:.2f}%')
        print(f'  Confidence Drop:          {cw["confidence_drop"]:.2f}pp')
        print(f'  ECE Increase:             {cw["ece_increase"]:.4f}')

        # Add to metrics
        final_metrics.update({
            # PGD metrics
            'adversarial/pgd_clean_accuracy': pgd['clean_accuracy'],
            'adversarial/pgd_clean_ece': pgd['clean_ece'],
            'adversarial/pgd_clean_confidence': pgd['clean_mean_confidence'],
            'adversarial/pgd_adv_accuracy': pgd['adversarial_accuracy'],
            'adversarial/pgd_adv_ece': pgd['adversarial_ece'],
            'adversarial/pgd_adv_confidence': pgd['adversarial_mean_confidence'],
            'adversarial/pgd_attack_success': pgd['attack_success_rate'],
            'adversarial/pgd_confidence_drop': pgd['confidence_drop'],
            'adversarial/pgd_ece_increase': pgd['ece_increase'],

            # C&W metrics
            'adversarial/cw_clean_accuracy': cw['clean_accuracy'],
            'adversarial/cw_clean_ece': cw['clean_ece'],
            'adversarial/cw_clean_confidence': cw['clean_mean_confidence'],
            'adversarial/cw_adv_accuracy': cw['adversarial_accuracy'],
            'adversarial/cw_adv_ece': cw['adversarial_ece'],
            'adversarial/cw_adv_confidence': cw['adversarial_mean_confidence'],
            'adversarial/cw_attack_success': cw['attack_success_rate'],
            'adversarial/cw_confidence_drop': cw['confidence_drop'],
            'adversarial/cw_ece_increase': cw['ece_increase'],

            # Summary metrics
            'adversarial/mean_adv_accuracy': (pgd['adversarial_accuracy'] + cw['adversarial_accuracy']) / 2,
            'adversarial/mean_adv_ece': (pgd['adversarial_ece'] + cw['adversarial_ece']) / 2,
            'adversarial/mean_ece_increase': (pgd['ece_increase'] + cw['ece_increase']) / 2,
        })

        # Store for summary
        pgd_adv_acc = pgd['adversarial_accuracy']
        cw_adv_acc = cw['adversarial_accuracy']
        pgd_adv_ece = pgd['adversarial_ece']
        cw_adv_ece = cw['adversarial_ece']

    # ================================================================
    # Updated Summary Section (replace the existing summary)
    # ================================================================
    wandb.log(final_metrics)
    print(f'\n' + '=' * 70)
    print('EVALUATION SUMMARY')
    print('=' * 70)
    print(f'\nModel trained with temperature schedule: {args.temp_schedule}')
    print(f'Temperature at best checkpoint (epoch {checkpoint["epoch"]}): {train_temp:.2f}')

    print(f'\n--- Standard Evaluation ---')
    print(f'{"Metric":<30} {"T_train":<15} {"T=1.0":<15} {"Diff":<15}')
    print('-' * 75)
    print(f'{"Test Accuracy (%)":<30} {test_acc_Ttrain:<15.2f} {test_acc_T1:<15.2f} {acc_diff:<+15.2f}')
    print(f'{"Test ECE":<30} {test_ece_Ttrain:<15.4f} {test_ece_T1:<15.4f} {ece_diff:<+15.4f}')

    if args.eval_corruptions:
        print(
            f'{"Corruption Accuracy (%)":<30} {corruption_results_Ttrain["mean_acc"]:<15.2f} {corruption_results_T1["mean_acc"]:<15.2f} {corr_acc_diff:<+15.2f}')
        print(
            f'{"Corruption ECE":<30} {corruption_results_Ttrain["mean_ece"]:<15.4f} {corruption_results_T1["mean_ece"]:<15.4f} {corr_ece_diff:<+15.4f}')

    if args.eval_adversarial:
        print(f'\n--- Adversarial Evaluation (T=1.0 only, per paper) ---')
        print(f'{"Metric":<30} {"PGD-20":<15} {"C&W":<15}')
        print('-' * 60)
        print(f'{"Clean Accuracy (%)":<30} {pgd["clean_accuracy"]:<15.2f} {cw["clean_accuracy"]:<15.2f}')
        print(f'{"Clean ECE":<30} {pgd["clean_ece"]:<15.4f} {cw["clean_ece"]:<15.4f}')
        print(
            f'{"Adversarial Accuracy (%)":<30} {pgd["adversarial_accuracy"]:<15.2f} {cw["adversarial_accuracy"]:<15.2f}')
        print(f'{"Adversarial ECE":<30} {pgd["adversarial_ece"]:<15.4f} {cw["adversarial_ece"]:<15.4f}')
        print(f'{"Attack Success Rate (%)":<30} {pgd["attack_success_rate"]:<15.2f} {cw["attack_success_rate"]:<15.2f}')
        print(f'{"Confidence Drop (pp)":<30} {pgd["confidence_drop"]:<15.2f} {cw["confidence_drop"]:<15.2f}')
        print(f'{"ECE Increase":<30} {pgd["ece_increase"]:<15.4f} {cw["ece_increase"]:<15.4f}')

    print()
    print('NOTES:')
    print('  - Standard eval: Diff = (T=1.0) - (T_train)')
    print('  - Adversarial eval: Always T=1.0 per paper methodology')
    print('  - ECE Increase > 0 means worse calibration under attack')
    print('  - High ECE + High Attack Success = Overconfident wrong predictions (bad)')
    print('=' * 70)


if __name__ == '__main__':
    main()