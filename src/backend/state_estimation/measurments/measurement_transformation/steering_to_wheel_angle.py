import numpy as np
import matplotlib.pyplot as plt


def measure_delta_wheel_angle_old(steering_angle: float) -> np.ndarray:
    """
    Convert steering angle (deg) to delta wheel angle (rad)
    :param steering_angle: (deg)
    :return: np.array([delta_FL, delta_FR, delta_RL, delta_RR]) (rad)
    """
    delta_wheels = np.zeros(4)
    res1 = 0.165 * steering_angle - 9.5e-4 * (steering_angle ** 2) * np.sign(steering_angle)
    res0 = 0.207 * steering_angle + 1.02e-4 * (steering_angle ** 2) * np.sign(steering_angle)
    delta_wheels[0] = res0 if steering_angle > 0 else res1
    delta_wheels[1] = res0 if steering_angle < 0 else res1
    return np.deg2rad(delta_wheels)


def measure_delta_wheel_angle(steering_angle: float) -> np.ndarray:
    """
    Convert steering angle (deg) to delta wheel angle (rad)
    :param steering_angle: (deg)
    :return: np.array([delta_FL, delta_FR, delta_RL, delta_RR]) (rad)
    """
    delta_wheels = np.zeros(4)
    k_FL = [1.24529686449354e-08, 2.15639847160840e-06, 0.000194484114611891, 0.208689090578083, 0.0378881485270762]
    k_FR = [-1.24529686449398e-08, 2.15639847160880e-06, -0.000194484114611844, 0.208689090578078, -0.0378881485268889]
    x = np.array([steering_angle ** 4, steering_angle ** 3, steering_angle ** 2, steering_angle, 1])
    delta_wheels[0] = np.dot(k_FL, x)
    delta_wheels[1] = np.dot(k_FR, x)
    return np.deg2rad(delta_wheels)


if __name__ == '__main__':
    steering_angles = np.linspace(-80, 80, 100)
    delta_wheel_angles = np.array([measure_delta_wheel_angle_old(steering_angle) for steering_angle in steering_angles])

    delta_wheel_angles = np.rad2deg(delta_wheel_angles)
    delta_FL = delta_wheel_angles[:, 0]
    delta_FR = delta_wheel_angles[:, 1]
    delta_RL = delta_wheel_angles[:, 2]
    delta_RR = delta_wheel_angles[:, 3]

    _ = plt.figure(figsize=(12, 6))
    plt.plot(steering_angles, delta_FL, label='FL')
    plt.plot(steering_angles, delta_FR, label='FR')
    plt.plot(steering_angles, delta_RL, label='RL')
    plt.plot(steering_angles, delta_RR, label='RR')
    plt.legend()
    plt.xlabel('Steering angle [deg]')
    plt.ylabel('Delta wheel angle [deg]')
    plt.show()
