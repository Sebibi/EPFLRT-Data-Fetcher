import numpy as np


def measure_delta_wheel_angle(steering_angle: float) -> np.ndarray:
    delta_wheels = np.zeros(4)  # Delta FL - Delta FR - Delta RL - Delta RR
    res0 = 0.165 * steering_angle - 9.5e-4 * steering_angle ** 2
    res1 = 0.207 * steering_angle + 1.02e-4 * steering_angle ** 2
    res0 *= np.pi / 180.0
    res1 *= np.pi / 180.0
    delta_wheels[0] = res0 if steering_angle > 0 else res1
    delta_wheels[1] = res0 if steering_angle < 0 else res1
    return delta_wheels