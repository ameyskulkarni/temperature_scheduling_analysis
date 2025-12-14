"""
Greedy Model Soup

Implements the Greedy Soup algorithm from Wortsman et al., 2022:
"Model soups: averaging weights of multiple fine-tuned models
improves accuracy without increasing inference time"

Algorithm:
    1. Sort checkpoints by validation accuracy (descending)
    2. Start with empty soup
    3. For each checkpoint:
       - Try adding it to the soup
       - Keep it if it improves (or maintains) validation accuracy
    4. Return the averaged weights of all kept checkpoints

Usage:
    python greedy_model_soup.py \
        --checkpoint-dir ./models/temp_30 \
        --model resnet32 \
        --dataset cifar100 \
        --top-k 40 \
        --output-dir ./results/soup
"""

import argparse
import json
import copy
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from models import get_model
from evaluate_checkpoint import (
    evaluate_on_testset,
    evaluate_on_corruptions,
    evaluate_loader,
    get_test_loader,
    load_model_from_state_dict,
    DATASET_CONFIGS,
    EvalResult,
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CheckpointInfo:
    """Information about a checkpoint."""
    path: Path
    epoch: int
    val_accuracy: float
    train_temperature: float

    def to_dict(self) -> Dict:
        return {
            'path': str(self.path),
            'epoch': self.epoch,
            'val_accuracy': self.val_accuracy,
            'train_temperature': self.train_temperature,
        }


@dataclass
class SoupStep:
    """Record of one step in soup construction."""
    step: int
    checkpoint: str
    checkpoint_val_acc: float
    soup_acc_if_added: float
    current_soup_acc: float
    added: bool
    soup_size: int


@dataclass
class SoupResult:
    """Final result of soup construction."""
    num_ingredients: int
    ingredients: List[str]
    final_val_accuracy: float
    construction_history: List[Dict]


# =============================================================================
# Checkpoint Discovery
# =============================================================================

def discover_checkpoints(checkpoint_dir: str, pattern: str = '*.pth') -> List[CheckpointInfo]:
    """
    Find all checkpoints in a directory and extract their metadata.

    Args:
        checkpoint_dir: Directory containing checkpoints
        pattern: Glob pattern for checkpoint files

    Returns:
        List of CheckpointInfo sorted by validation accuracy (descending)
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_files = list(checkpoint_dir.glob(pattern))

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    print(f"Found {len(checkpoint_files)} checkpoint files")

    checkpoints = []
    for path in tqdm(checkpoint_files, desc="Loading metadata"):
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)

            info = CheckpointInfo(
                path=path,
                epoch=ckpt.get('epoch', -1),
                val_accuracy=ckpt.get('val_accuracy', 0.0),
                train_temperature=ckpt.get('train_temperature', 1.0),
            )
            checkpoints.append(info)

        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")

    # Deduplicate by epoch: keep best val_accuracy per epoch
    epoch_to_best = {}
    for c in checkpoints:
        if (c.epoch not in epoch_to_best or
                c.val_accuracy > epoch_to_best[c.epoch].val_accuracy):
            epoch_to_best[c.epoch] = c

    checkpoints = list(epoch_to_best.values())
    # Sort by validation accuracy (descending)
    checkpoints.sort(key=lambda x: x.val_accuracy, reverse=True)

    print(f"Loaded {len(checkpoints)} checkpoints")
    print(f"  Best val acc: {checkpoints[0].val_accuracy:.2f}%")
    print(f"  Worst val acc: {checkpoints[-1].val_accuracy:.2f}%")

    return checkpoints


def filter_checkpoints(checkpoints: List[CheckpointInfo],
                       top_k: int = None,
                       last_n_epochs: int = None) -> List[CheckpointInfo]:
    """
    Filter checkpoints by top-k or last N epochs.

    Args:
        checkpoints: List of checkpoints (already sorted by val_accuracy)
        top_k: Keep only top-k by validation accuracy
        last_n_epochs: Keep only checkpoints from last N epochs

    Returns:
        Filtered list of checkpoints
    """
    if last_n_epochs is not None:
        max_epoch = max(c.epoch for c in checkpoints)
        min_epoch = max_epoch - last_n_epochs + 1
        checkpoints = [c for c in checkpoints if c.epoch >= min_epoch]
        # Re-sort after filtering
        checkpoints.sort(key=lambda x: x.val_accuracy, reverse=True)
        print(f"Filtered to epochs {min_epoch}-{max_epoch}: {len(checkpoints)} checkpoints")

    if top_k is not None:
        checkpoints = checkpoints[:top_k]
        print(f"Filtered to top {len(checkpoints)} checkpoints")

    return checkpoints


# =============================================================================
# Model Averaging
# =============================================================================

def load_state_dict(checkpoint_path: Path) -> OrderedDict:
    """Load state dict from checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'model_state_dict' in ckpt:
        return ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        return ckpt['state_dict']
    else:
        return ckpt


def average_state_dicts(state_dicts: List[OrderedDict]) -> OrderedDict:
    """Average multiple state dictionaries."""
    if len(state_dicts) == 0:
        raise ValueError("Cannot average empty list")

    if len(state_dicts) == 1:
        return copy.deepcopy(state_dicts[0])

    avg = OrderedDict()
    for key in state_dicts[0].keys():
        stacked = torch.stack([sd[key].float() for sd in state_dicts])
        avg[key] = stacked.mean(dim=0)

    return avg


# =============================================================================
# Greedy Soup Builder
# =============================================================================

class GreedySoupBuilder:
    """
    Builds a greedy model soup.

    Greedily adds checkpoints to the soup if they improve validation accuracy.
    """

    def __init__(self, model_name: str, num_classes: int,
                 val_loader: torch.utils.data.DataLoader,
                 device: torch.device):
        self.model_name = model_name
        self.num_classes = num_classes
        self.val_loader = val_loader
        self.device = device

        self.ingredients: List[Path] = []
        self.state_dicts: List[OrderedDict] = []
        self.history: List[SoupStep] = []

    def _evaluate_soup(self, state_dicts: List[OrderedDict]) -> float:
        """Evaluate accuracy of averaged state dicts."""
        if len(state_dicts) == 0:
            return 0.0

        avg_state_dict = average_state_dicts(state_dicts)
        model = load_model_from_state_dict(avg_state_dict, self.model_name,
                                           self.num_classes, self.device)

        result = evaluate_loader(model, self.val_loader, self.device,
                                  compute_calibration=False)
        return result.accuracy

    def build(self, checkpoints: List[CheckpointInfo], verbose: bool = True) -> OrderedDict:
        """
        Build greedy soup from checkpoints.

        Args:
            checkpoints: List of checkpoints sorted by val_accuracy (descending)
            verbose: Print progress

        Returns:
            State dict of the final soup
        """
        if verbose:
            print(f"\n{'='*60}")
            print("Building Greedy Model Soup")
            print(f"{'='*60}")
            print(f"Candidates: {len(checkpoints)}")
            print(f"{'='*60}\n")

        self.ingredients = []
        self.state_dicts = []
        self.history = []
        current_acc = 0.0

        iterator = tqdm(checkpoints, desc="Building soup") if verbose else checkpoints

        for i, ckpt in enumerate(iterator):
            # Load candidate
            candidate_sd = load_state_dict(ckpt.path)

            # Try adding to soup
            trial_sds = self.state_dicts + [candidate_sd]
            trial_acc = self._evaluate_soup(trial_sds)

            # Record step
            step = SoupStep(
                step=i + 1,
                checkpoint=str(ckpt.path.name),
                checkpoint_val_acc=ckpt.val_accuracy,
                soup_acc_if_added=trial_acc,
                current_soup_acc=current_acc,
                added=False,
                soup_size=len(self.ingredients),
            )

            # Greedy decision
            if trial_acc >= current_acc:
                self.ingredients.append(ckpt.path)
                self.state_dicts.append(candidate_sd)
                current_acc = trial_acc
                step.added = True
                step.soup_size = len(self.ingredients)

                if verbose:
                    tqdm.write(f"  [+] Added {ckpt.path.name} | Soup: {len(self.ingredients)} models, Acc: {current_acc:.2f}%")

            self.history.append(step)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Soup Complete: {len(self.ingredients)}/{len(checkpoints)} models")
            print(f"Final Validation Accuracy: {current_acc:.2f}%")
            print(f"{'='*60}\n")

        if self.state_dicts:
            return average_state_dicts(self.state_dicts)
        else:
            # Fallback: return best individual checkpoint
            print("Warning: No ingredients added, returning best checkpoint")
            return load_state_dict(checkpoints[0].path)

    def get_result(self) -> SoupResult:
        """Get soup construction result."""
        final_acc = self._evaluate_soup(self.state_dicts) if self.state_dicts else 0.0

        return SoupResult(
            num_ingredients=len(self.ingredients),
            ingredients=[str(p) for p in self.ingredients],
            final_val_accuracy=final_acc,
            construction_history=[asdict(s) for s in self.history],
        )


# =============================================================================
# Main Experiment Function
# =============================================================================

def run_greedy_soup(
    checkpoint_dir: str,
    model_name: str = 'resnet32',
    dataset_name: str = 'cifar100',
    top_k: int = None,
    last_n_epochs: int = None,
    corruption_severity: int = 3,
    data_root: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
    output_dir: str = None,
    eval_split: str = 'val',
    device: torch.device = None,
) -> Dict:
    """
    Run complete greedy soup experiment.

    Args:
        checkpoint_dir: Directory with checkpoints
        model_name: Model architecture
        dataset_name: 'cifar10' or 'cifar100'
        top_k: Use only top-k checkpoints by val accuracy
        last_n_epochs: Use only checkpoints from last N epochs
        corruption_severity: Severity for CIFAR-C evaluation
        data_root: Data directory
        batch_size: Batch size
        num_workers: Data workers
        output_dir: Directory to save results
        device: Compute device

    Returns:
        Dictionary with all results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = DATASET_CONFIGS[dataset_name]
    num_classes = config['num_classes']

    print(f"\n{'='*70}")
    print("GREEDY MODEL SOUP EXPERIMENT")
    print(f"{'='*70}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")

    # 1. Discover and filter checkpoints
    checkpoints = discover_checkpoints(checkpoint_dir)
    checkpoints = filter_checkpoints(checkpoints, top_k, last_n_epochs)

    # 2. Get validation loader for soup construction
    # Using test set as validation for soup (in practice, use held-out val set)
    val_loader = get_test_loader(dataset_name, data_root, batch_size, num_workers, eval_split)
    test_loader = get_test_loader(dataset_name, data_root, batch_size, num_workers, 'test')

    # 3. Build soup
    builder = GreedySoupBuilder(model_name, num_classes, val_loader, device)
    soup_state_dict = builder.build(checkpoints, verbose=True)
    soup_result = builder.get_result()

    # 4. Create soup model
    soup_model = load_model_from_state_dict(soup_state_dict, model_name,
                                             num_classes, device)

    # 5. Evaluate soup
    print(f"\n{'='*70}")
    print("SOUP EVALUATION")
    print(f"{'='*70}\n")

    soup_test = evaluate_on_testset(soup_model, dataset_name, device,
                                     data_root, batch_size, num_workers, eval_split='test')
    soup_corrupt = evaluate_on_corruptions(soup_model, dataset_name, device,
                                            data_root, batch_size, num_workers,
                                            corruption_severity)

    # 6. Evaluate best individual checkpoint for comparison
    print(f"\n{'='*70}")
    print("BASELINE (Best Individual Checkpoint)")
    print(f"{'='*70}\n")

    best_sd = load_state_dict(checkpoints[0].path)
    best_model = load_model_from_state_dict(best_sd, model_name, num_classes, device)

    best_test = evaluate_on_testset(best_model, dataset_name, device,
                                     data_root, batch_size, num_workers, eval_split='test')
    best_corrupt = evaluate_on_corruptions(best_model, dataset_name, device,
                                            data_root, batch_size, num_workers,
                                            corruption_severity)

    # 7. Compile results
    results = {
        'config': {
            'checkpoint_dir': str(checkpoint_dir),
            'model_name': model_name,
            'dataset': dataset_name,
            'top_k': top_k,
            'last_n_epochs': last_n_epochs,
            'corruption_severity': corruption_severity,
            'timestamp': datetime.now().isoformat(),
        },
        'soup': {
            'num_ingredients': soup_result.num_ingredients,
            'ingredients': soup_result.ingredients,
            'test': soup_test.to_dict(),
            'corruptions': soup_corrupt.to_dict(),
        },
        'baseline': {
            'checkpoint': str(checkpoints[0].path),
            'test': best_test.to_dict(),
            'corruptions': best_corrupt.to_dict(),
        },
        'comparison': {
            'test_accuracy': {
                'soup': soup_test.accuracy,
                'baseline': best_test.accuracy,
                'improvement': soup_test.accuracy - best_test.accuracy,
            },
            'test_ece': {
                'soup': soup_test.ece,
                'baseline': best_test.ece,
                'improvement': soup_test.ece - best_test.ece,
            },
            'corruption_accuracy': {
                'soup': soup_corrupt.mean_accuracy,
                'baseline': best_corrupt.mean_accuracy,
                'improvement': soup_corrupt.mean_accuracy - best_corrupt.mean_accuracy,
            },
            'corruption_ece': {
                'soup': soup_corrupt.mean_ece,
                'baseline': best_corrupt.mean_ece,
                'improvement': soup_corrupt.mean_ece - best_corrupt.mean_ece,
            },
        },
        'construction_history': soup_result.construction_history,
    }

    # 8. Print summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\nSoup: {soup_result.num_ingredients} models")
    print(f"\n{'Metric':<25} {'Soup':>12} {'Baseline':>12} {'Δ':>12}")
    print("-" * 55)
    print(f"{'Test Accuracy':<25} {soup_test.accuracy:>11.2f}% {best_test.accuracy:>11.2f}% {soup_test.accuracy - best_test.accuracy:>+11.2f}%")
    print(f"{'Test ECE':<25} {soup_test.ece:>12.4f} {best_test.ece:>12.4f} {soup_test.ece - best_test.ece:>+12.4f}")
    print(f"{'Corruption Accuracy':<25} {soup_corrupt.mean_accuracy:>11.2f}% {best_corrupt.mean_accuracy:>11.2f}% {soup_corrupt.mean_accuracy - best_corrupt.mean_accuracy:>+11.2f}%")
    print(f"{'Corruption ECE':<25} {soup_corrupt.mean_ece:>12.4f} {best_corrupt.mean_ece:>12.4f} {soup_corrupt.mean_ece - best_corrupt.mean_ece:>+12.4f}")
    print(f"{'='*70}\n")

    # 9. Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save results JSON
        results_path = output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {results_path}")

        # Save soup model
        soup_path = output_dir / 'soup_model.pth'
        torch.save({
            'model_state_dict': soup_state_dict,
            'num_ingredients': soup_result.num_ingredients,
            'ingredients': soup_result.ingredients,
        }, soup_path)
        print(f"Soup model saved to: {soup_path}")

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Build Greedy Model Soup')

    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Directory containing checkpoints')
    parser.add_argument('--model', type=str, default='resnet32',
                        help='Model architecture')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100'],
                        help='Dataset')
    parser.add_argument('--top-k', type=int, default=None,
                        help='Use only top-k checkpoints by val accuracy')
    parser.add_argument('--last-n-epochs', type=int, default=None,
                        help='Use only checkpoints from last N epochs')
    parser.add_argument('--severity', type=int, default=3,
                        help='Corruption severity (1-5)')
    parser.add_argument('--data-root', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Data workers')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--device', type=str, default=None,
                        help='Device')
    parser.add_argument('--eval-split', type=str, default='val',
                        choices=['val', 'test'],
                        help='Which split to use for evaluation during training (val or test)')

    args = parser.parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_greedy_soup(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model,
        dataset_name=args.dataset,
        top_k=args.top_k,
        last_n_epochs=args.last_n_epochs,
        corruption_severity=args.severity,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        eval_split=args.eval_split,
        device=device,
    )


if __name__ == '__main__':
    main()