"""
Learning Rate Schedulers with Warmup
"""
import math


class CosineLRScheduler:
    """Cosine annealing learning rate scheduler with optional warmup"""

    def __init__(self, lr_max, lr_min=0.0, total_epochs=200, warmup_epochs=0):
        """
        Args:
            lr_max: Maximum learning rate (after warmup)
            lr_min: Minimum learning rate
            total_epochs: Total number of training epochs
            warmup_epochs: Number of epochs for linear warmup
        """
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs

    def get_lr(self, epoch):
        """Get learning rate for given epoch"""
        if epoch < self.warmup_epochs:
            # Linear warmup: 0 -> lr_max
            return self.lr_max * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing: lr_max -> lr_min
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * progress))


class ConstantLRScheduler:
    """Constant learning rate (baseline)"""

    def __init__(self, lr):
        self.lr = lr

    def get_lr(self, epoch):
        return self.lr


def get_lr_scheduler(schedule_type, lr_max, lr_min=0.0, total_epochs=200, warmup_epochs=5):
    """Factory function to get learning rate scheduler"""
    if schedule_type == 'constant':
        return ConstantLRScheduler(lr_max)
    elif schedule_type in ['cosine', 'cosine_warmup']:
        warmup = warmup_epochs if schedule_type == 'cosine_warmup' else 0
        return CosineLRScheduler(lr_max, lr_min, total_epochs, warmup)
    else:
        raise ValueError(f"Unknown LR schedule: {schedule_type}")