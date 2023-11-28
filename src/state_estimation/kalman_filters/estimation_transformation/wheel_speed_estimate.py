import numpy as np

from src.state_estimation.config.vehicle_params import VehicleParams
from src.state_estimation.kalman_filters.estimation_transformation.longitudonal_speed import get_longitudonal__speed


def get_wheel_speed_estimate(x: np.array, steering_deltas: np.array, wheel_id: int) -> float:
    assert wheel_id in [0, 1, 2, 3]
    vl = get_longitudonal__speed(x, steering_deltas, wheel_id)
    slip_ratio = x[5 + 1]
    wheel_speed = vl * (1 - slip_ratio) / VehicleParams.Rw
    return wheel_speed