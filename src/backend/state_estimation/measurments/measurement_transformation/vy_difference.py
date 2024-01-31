import numpy as np

from src.backend.state_estimation.config.vehicle_params import VehicleParams
from src.backend.state_estimation.measurments.measurement_transformation.sum_of_forces import measure_fsum


def _measure_velocities(ax_body: float, ay_body: float, ax_world: float, ay_world: float, yaw_rate: float) -> np.array:
    #  accX = (Fx / VehicleParams.m_car) - vy * yaw_rate
    #  vy = ((Fx / m) - accX) / yaw_rate

    #  accY = (Fy / VehicleParams.m_car) + vx * yaw_rate
    # vx = -((Fy / m) - accY) / yaw_rate

    """
    Measure body velocities by comparing body and world acceleration and yaw rate at center of gravity
    :param ax_body: float
    :param ay_body: float
    :param ax_world: float
    :param ay_world: float
    :param yaw_rate: float
    :return: np.array
    """

    vx = (ay_body - ay_world) / yaw_rate
    vy = (ax_world - ax_body) / yaw_rate
    return np.array([vx, vy])


def measure_velocities(long_tire_forces: np.ndarray, wheel_deltas: np.ndarray, state: np.ndarray) -> np.array:
    vx, vy, yaw_rate = state[0], state[1], state[4]
    Fx, Fy = measure_fsum(long_tire_forces, wheel_deltas, state)
    ax_world = Fx / VehicleParams.m_car
    ay_world = Fy / VehicleParams.m_car
    ax_body, ay_body = state[2], state[3]
    return _measure_velocities(ax_body, ay_body, ax_world, ay_world, yaw_rate)
