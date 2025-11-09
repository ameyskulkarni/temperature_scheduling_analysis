"""
Temperature Schedulers
"""

import numpy as np


class TemperatureScheduler:
    """Base class for temperature schedulers"""

    def __init__(self, T_max=1.0, T_min=1.0, total_epochs=200):
        self.T_max = T_max
        self.T_min = T_min
        self.total_epochs = total_epochs

    def get_temperature(self, epoch):
        raise NotImplementedError


class ConstantTemperature(TemperatureScheduler):
    """Constant temperature"""

    def get_temperature(self, epoch):
        return self.T_max


class CosineTemperature(TemperatureScheduler):
    """Cosine annealing from T_max to T_min"""

    def get_temperature(self, epoch):
        return self.T_min + 0.5 * (self.T_max - self.T_min) * (
                1 + np.cos(np.pi * epoch / self.total_epochs)
        )


class LinearTemperature(TemperatureScheduler):
    """Linear decay from T_max to T_min"""

    def get_temperature(self, epoch):
        return self.T_max - (self.T_max - self.T_min) * epoch / self.total_epochs


def get_temperature_scheduler(schedule_type, T_max=1.0, T_min=1.0, total_epochs=200):
    """
    Factory function to create temperature scheduler
    """
    schedulers = {
        'constant': ConstantTemperature,
        'cosine': CosineTemperature,
        'linear': LinearTemperature,
    }

    if schedule_type not in schedulers:
        raise ValueError(f"Unknown schedule: {schedule_type}")

    return schedulers[schedule_type](T_max, T_min, total_epochs)