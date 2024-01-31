import numpy as np

from src.backend.state_estimation.config.vehicle_params import VehicleParams
from src.backend.state_estimation.kalman_filters.estimation_transformation.normal_forces import \
    estimate_aero_focre_one_tire


def get_rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def wheel_to_car_frame(Fl, Flat, wheel_delta):
    F_wheel = np.array([Fl, Flat])
    return np.matmul(get_rotation_matrix(wheel_delta), F_wheel)


def car_to_wheel_frame(Fx, Fy, wheel_delta) -> np.array:
    F_car = np.array([Fx, Fy])
    return np.matmul(get_rotation_matrix(-wheel_delta), F_car)


def measure_fsum(long_tire_forces: np.ndarray, wheel_deltas: np.ndarray, state: np.ndarray) -> np.array:
    Fx, Fy = np.sum([wheel_to_car_frame(long_tire_forces[i], 0, wheel_deltas[i]) for i in range(4)], axis=0)
    F_drag = estimate_aero_focre_one_tire(state) * 4.0
    Fx -= F_drag
    return np.array([Fx, Fy])


def measure_acc_fsum(long_tire_forces: np.ndarray, wheel_deltas: np.ndarray, state: np.ndarray) -> np.array:
    vx, vy, yaw_rate = state[0], state[1], state[4]
    Fx, Fy = measure_fsum(long_tire_forces, wheel_deltas, state)
    accX = (Fx / VehicleParams.m_car) - vy * yaw_rate  # Account for rotating axes
    accY = (Fy / VehicleParams.m_car) + vx * yaw_rate
    return np.array([accX, accY])


if __name__ == '__main__':
    lf = np.array([1, 2, 3, 4])
    wd = np.array([0, 0, 0, 0])
    state = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    print(measure_acc_fsum(lf, wd, state))
