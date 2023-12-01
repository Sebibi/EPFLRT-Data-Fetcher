import numpy as np

from src.backend.state_estimation.config.vehicle_params import VehicleParams


def measure_wheel_speeds(motor_speeds: np.ndarray) -> np.ndarray:
    return motor_speeds * np.pi / (30.0 * VehicleParams.gear_ratio)


