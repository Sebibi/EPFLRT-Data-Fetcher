import numpy as np


def measure_delta_wheel_angle(steering_angle: float) -> np.ndarray:
    """
    Convert steering angle (deg) to delta wheel angle (rad)
    :param steering_angle: (deg)
    :return: np.array([delta_FL, delta_FR, delta_RL, delta_RR]) (rad)
    """
    delta_wheels = np.zeros(4)
    res0 = 0.165 * steering_angle - 9.5e-4 * steering_angle ** 2
    res1 = 0.207 * steering_angle + 1.02e-4 * steering_angle ** 2
    delta_wheels[0] = res0 if steering_angle > 0 else res1
    delta_wheels[1] = res0 if steering_angle < 0 else res1
    return np.deg2rad(delta_wheels)