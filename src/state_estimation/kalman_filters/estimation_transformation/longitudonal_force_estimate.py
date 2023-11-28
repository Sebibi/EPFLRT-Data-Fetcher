import numpy as np

from src.state_estimation.config.vehicle_params import VehicleParams
from src.state_estimation.kalman_filters.estimation_transformation.normal_forces_estimation import get_normal_force


def get_longitudonal_force_estimate(x: np.array, wheel_id: int) -> float:
    normal_force = get_normal_force(x, wheel_id=wheel_id)
    mu = VehicleParams.inverse_magic_formula()
    return normal_force * mu