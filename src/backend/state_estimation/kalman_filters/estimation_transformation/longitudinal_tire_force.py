import numpy as np

from src.backend.state_estimation.config.vehicle_params import VehicleParams
from src.backend.state_estimation.kalman_filters.estimation_transformation.normal_forces import estimate_normal_force, \
    estimate_normal_forces


def estimate_longitudinal_tire_force(x: np.array, wheel_id: int) -> np.ndarray:
    normal_force = estimate_normal_force(x, wheel_id=wheel_id)
    mu = VehicleParams.magic_formula(slip_ratio=x[5 + wheel_id])
    return np.array([normal_force * mu])


def estimate_longitudinal_tire_forces(x: np.array) -> np.ndarray:
    normal_forces = estimate_normal_forces(x)
    mu = np.array([VehicleParams.magic_formula(s) for s in x[5:9]])
    return normal_forces * mu
