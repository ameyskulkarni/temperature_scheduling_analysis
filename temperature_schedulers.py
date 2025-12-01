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

class ReverseCosineTemperature(TemperatureScheduler):
    """
    Reverse cosine: Starts at T_min and increases to T_max
    Useful for gradually increasing temperature (confidence reduction over time)
    """

    def get_temperature(self, epoch):
        # Cosine goes from 1 to -1, we want to go from T_min to T_max
        return self.T_max - 0.5 * (self.T_max - self.T_min) * (
                1 + np.cos(np.pi * epoch / self.total_epochs)
        )

class LinearTemperature(TemperatureScheduler):
    """Linear decay from T_max to T_min"""

    def get_temperature(self, epoch):
        return self.T_max - (self.T_max - self.T_min) * epoch / self.total_epochs


class CyclicSinusoidalTemperature(TemperatureScheduler):
    """
    Cyclic sinusoidal temperature: oscillates between T_min and T_max

    Args:
        cycle_length: Number of epochs per complete cycle (default: 40)
        phase: Phase shift in radians (default: 0)
    """

    def __init__(self, T_max=1.0, T_min=1.0, total_epochs=200, cycle_length=40, phase=0.0):
        super().__init__(T_max, T_min, total_epochs)
        self.cycle_length = cycle_length
        self.phase = phase

    def get_temperature(self, epoch):
        """
        Sinusoidal oscillation: T(t) = T_mean + amplitude * sin(2π * t / period + phase)
        where T_mean = (T_max + T_min) / 2
              amplitude = (T_max - T_min) / 2
        """
        T_mean = (self.T_max + self.T_min) / 2
        amplitude = (self.T_max - self.T_min) / 2

        # Sine oscillates between -1 and 1
        oscillation = np.sin(2 * np.pi * epoch / self.cycle_length + self.phase)

        return T_mean + amplitude * oscillation


class CyclicTriangularTemperature(TemperatureScheduler):
    """
    Cyclic triangular temperature: Linear oscillation between T_min and T_max
    Forms a sawtooth/triangle wave pattern

    Args:
        cycle_length: Number of epochs per complete cycle (default: 20)
    """

    def __init__(self, T_max=1.0, T_min=1.0, total_epochs=200,
                 cycle_length=40):
        super().__init__(T_max, T_min, total_epochs)
        self.cycle_length = cycle_length

    def get_temperature(self, epoch):
        # Position within current cycle (0 to 1)
        cycle_position = (epoch % self.cycle_length) / self.cycle_length

        # Symmetric triangle wave
        if cycle_position < 0.5:
            # First half: increase from T_min to T_max
            progress = cycle_position * 2  # 0 to 1
        else:
            # Second half: decrease from T_max to T_min
            progress = 2 - cycle_position * 2  # 1 to 0

        return self.T_min + (self.T_max - self.T_min) * progress


def get_temperature_scheduler(schedule_type, T_max=1.0, T_min=1.0, total_epochs=200, **kwargs):
    """
    Factory function to create temperature scheduler
    """
    schedulers = {
        'constant': ConstantTemperature,
        'cosine': CosineTemperature,
        'linear': LinearTemperature,
        'reverse_cosine': ReverseCosineTemperature,
        'cyclic_sinusoidal': CyclicSinusoidalTemperature,
        'cyclic_triangular': CyclicTriangularTemperature,
    }

    if schedule_type not in schedulers:
        raise ValueError(f"Unknown schedule: {schedule_type}")

    return schedulers[schedule_type](T_max, T_min, total_epochs, **kwargs)