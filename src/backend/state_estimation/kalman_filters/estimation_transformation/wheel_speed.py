import numpy as np

from src.backend.state_estimation.config.vehicle_params import VehicleParams
from src.backend.state_estimation.kalman_filters.estimation_transformation.longitudonal_speed import estimate_longitudinal_velocity


def estimate_wheel_speed(x: np.array, steering_deltas: np.array, wheel_id: int) -> np.ndarray:
    assert wheel_id in [0, 1, 2, 3]
    vl = estimate_longitudinal_velocity(x, steering_deltas, wheel_id)
    slip_ratio = x[5 + wheel_id]
    # if slip_ratio < 0:
    #     wheel_speed = (vl / VehicleParams.Rw) * (1 + slip_ratio)
    # else:
    #     wheel_speed = vl / (VehicleParams.Rw * (1 - slip_ratio))
    wheel_speed = (vl / VehicleParams.Rw) * (1 + slip_ratio)
    return np.array([wheel_speed])


def estimate_wheel_speeds(x: np.array, steering_deltas: np.array) -> np.ndarray:
    ws = np.array([estimate_wheel_speed(x, steering_deltas, wheel_id) for wheel_id in range(4)]).flatten()
    return ws