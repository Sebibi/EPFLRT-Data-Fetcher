import numpy as np

from src.backend.state_estimation.config.vehicle_params import VehicleParams

px = [VehicleParams.lf, VehicleParams.lf, -VehicleParams.lr, -VehicleParams.lr]
py = [-VehicleParams.a / 2, VehicleParams.a / 2, -VehicleParams.b / 2, VehicleParams.b / 2]


def estimate_longitudinal_velocity(x: np.array, steering_deltas: np.array, wheel_id: int) -> np.ndarray:
    assert wheel_id in [0, 1, 2, 3]
    vx, vy, yaw_rate = x[0], x[1], x[4]
    delta = steering_deltas[wheel_id]
    steering_effect = np.array([np.cos(delta), np.sin(delta)])
    wheel_long_speed = np.array([vx - yaw_rate * py[wheel_id], vy + yaw_rate * px[wheel_id]])
    vl = steering_effect @ wheel_long_speed
    return vl


def estimate_longitudinal_velocities(x: np.array, steering_deltas: np.array) -> np.ndarray:
    vls = np.array([estimate_longitudinal_velocity(x, steering_deltas, wheel_id) for wheel_id in range(4)]).flatten()
    return vls
