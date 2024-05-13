from collections import deque

import numpy as np
from filterpy import kalman

from src.backend.state_estimation.config.state_estimation_param import SE_param
from src.backend.state_estimation.config.vehicle_params import VehicleParams


class LKF:
    dim_x = SE_param.dim_x
    dim_z = 3
    history_size = 5  # Number of measurements to use for bias estimation

    def __init__(self):
        # INS measurements
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[:, [2, 3, 4]] = np.eye(self.dim_z)  # observation matrix
        self.R = SE_param.ins_measurement_noise.copy()

        # Velocity and slip reset
        self.H_vy_reset = np.zeros((1, self.dim_x))
        self.H_vy_reset[0, 1] = 1
        self.H_slip_reset = np.eye(4, self.dim_x, self.dim_x - 4)
        self.H_reset = np.concatenate((self.H_vy_reset, self.H_slip_reset))
        self.R_vy_reset = SE_param.vy_reset_noise.copy()
        self.R_reset = np.diag([0.1, 0.1, 0.1, 0.1, 0.1])

        # Bias estimation
        self.ax_hist = deque([0.0 for _ in range(self.history_size)], maxlen=self.history_size)
        self.ay_hist = deque([0.0 for _ in range(self.history_size)], maxlen=self.history_size)
        self.yaw_hist = deque([0.0 for _ in range(self.history_size)], maxlen=self.history_size * 50)
        self.ax_bias = 0
        self.ay_bias = 0

    def update_bias(self, wheel_speeds: np.ndarray) -> bool:
        """
        If the vehicle is not moving, estimate the bias of the accelerometer
        :param wheel_speeds:
        :return: (bool) True if the bias was estimated else False
        """
        if np.sum(wheel_speeds) == 0:
            self.ax_bias = np.mean(self.ax_hist)
            self.ay_bias = np.mean(self.ay_hist)
            return True
        return False

    def update(self, x: np.array, P: np.ndarray, ins: np.ndarray, attitude: np.ndarray) -> (np.array, np.ndarray):
        """
        Compute the update step of the LKF
        :param x: shape(n,)
        :param P: shape(n,n)
        :param ins: [ax ay yaw rate] np.array
        :return: x, P
        """
        ax, ay, _, _, _, yaw_rate = ins
        self.ax_hist.append(ax)
        self.ay_hist.append(ay)
        self.yaw_hist.append(yaw_rate)
        ax = ax - self.ax_bias - np.sin(attitude[0]) * VehicleParams.g
        ay = ay - self.ay_bias + np.sin(attitude[1]) * np.cos(attitude[0]) * VehicleParams.g
        x, P = kalman.update(x, P, np.array([ax, ay, yaw_rate]), self.R, self.H)
        return x, P

    def update_vy_reset(self, x: np.ndarray, P: np.ndarray):
        """
        Compute the update step of LKF setting vy at 0
        :param x: shape(n,)
        :param P: shape(n,n)
        :param ins: [ax ay yaw rate] np.ndarray
        :return: x, P
        """
        vx = x[0]
        if np.abs(vx) < 0.01:
            R = self.R_vy_reset
            H = self.H_vy_reset
            z = np.zeros(len(R))
            x, P = kalman.update(x, P, z=z, R=R, H=H)
        return x, P


if __name__ == '__main__':
    lkf = LKF()
    x = np.zeros(lkf.dim_x)
    P = np.eye(lkf.dim_x)
    z = np.ones(lkf.dim_z)
    x, P = lkf.update(x, P, z)
    print(x.round(3))

    z = np.array([0, 0, 0])
    x, P = lkf.update_vy_reset(x, P)
    print(x.round(3))

