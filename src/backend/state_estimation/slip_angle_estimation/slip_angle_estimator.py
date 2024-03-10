import numpy as np
from scipy.linalg import expm

from src.backend.state_estimation.config.vehicle_params import VehicleParams

lf = VehicleParams.lf
lr = VehicleParams.lr
m = VehicleParams.m_car
Iz = VehicleParams.lzz

kr = VehicleParams.kr
kf = VehicleParams.kf


class EKF_slip_angle:

    def __init__(self):
        self.dt = 0.01
        self.dim_x = 3
        self.dim_z = 1
        self.Q = np.diag([0.1, 0.1, 0.1])
        self.R = np.array([[0.1]]) * self.dt

        self.x = np.zeros(self.dim_x)
        self.P = np.eye(self.dim_x) * 1e-6

    def f(self, x: np.ndarray, steering_angle: float, ax: float) -> np.ndarray:
        """
        Predict the state with the EKF
        :param ax:
        :param x: state
        :param steering_angle: (rad)
        """
        yr, beta, vx = x

        yr_dot = ((lf ** 2 * kf + lr ** 2 * kr) * yr / (Iz * vx)
                  + (lf * kf - lr * kr) * beta / Iz
                  - lf * kf * steering_angle / Iz)

        beta_dot = ((((lf * kf - lr * kr) / (m * vx ** 2)) - 1) * yr
                    + (kf + kr) * beta / (m * vx)
                    - kf * steering_angle / (m * vx))

        vx_dot = ax + beta * yr * vx
        return np.array([yr_dot, beta_dot, vx_dot])

    def predict(self, steering_angle: float, ax: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict the state with the EKF
        :param P: covariance matrix
        :param steering_angle: (rad)
        :param state: [yaw rate, slip angle, vx]
        """
        yr, beta, vx = self.x

        F = np.array([
            [(lf ** 2 * kf + lr ** 2 * kr) / (Iz * vx),
             (lf * kf - lr * kr) / Iz,
             -(lf ** 2 * kf + lr ** 2 * kr) * yr / (Iz * vx ** 2)],

            [((lf * kf - lr * kr) / (m * vx ** 2)) - 1,
             (kf + kr) / (m * vx),
             -2 * (lf * kf - lr * kr) * yr / (m * vx ** 3)
             - (kf + kr) * beta / (m * vx ** 2)
             + kf * steering_angle / (m * vx ** 2)],

            [beta * vx, yr * vx, beta * yr]
        ])

        phi = expm(F * self.dt)

        self.x += self.f(self.x, steering_angle, ax) * self.dt
        self.P = phi @ self.P @ phi.T + self.Q
        return self.x, self.P

    def h(self, steering_angle: float) -> np.ndarray:
        """
        Predict the state with the EKF
        :param steering_angle: (rad)
        """
        yr, beta, vx = self.x
        return ((lf * kf - lr * kr) * yr / (m * vx)
                + (kf + kr) / m
                - kf * steering_angle / m)

    def update(self, steering_angle: float, ay: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Update the state with the EKF
        :param ay: lateral acceleration
        """
        yr, beta, vx = self.x
        H = np.array([(lf * kf - lr * kr) / (m * vx), (kf + kr) / m, -(kf * lf - kr * lr) * yr / (m * vx ** 2)])

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ (ay - self.h(steering_angle))
        self.P = (np.eye(self.dim_x) - K @ H) @ self.P
        return self.x, self.P

    def predict_update(self, steering_angle: float, ax: float, ay: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict and update the state with the EKF
        :param ay: lateral acceleration
        :param ax: longitudinal acceleration
        :param steering_angle: (rad)
        """
        self.predict(steering_angle, ax)
        self.update(steering_angle, ay)
        return self.x, self.P
