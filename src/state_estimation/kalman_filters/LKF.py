from collections import deque

import numpy as np
from filterpy import kalman


class LKF:
    dim_x = 9
    dim_z = 3
    history_size = 5  # Number of measurements to use for bias estimation
    measurement_cov = [0.9, 0.9, 0.1]

    def __init__(self):
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[:, [2, 3, 4]] = np.eye(self.dim_z)
        self.R = np.diag(self.measurement_cov)
        self.ax_queue = deque([0.0 for _ in range(self.history_size)], maxlen=self.history_size)
        self.ay_queue = deque([0.0 for _ in range(self.history_size)], maxlen=self.history_size)
        self.ax_bias = 0
        self.ay_bias = 0

    def update_bias(self, motor_speeds: np.ndarray):
        if np.sum(motor_speeds) == 0:
            self.ax_bias = np.mean(self.ax_queue)
            self.ay_bias = np.mean(self.ay_queue)

    def update(self, x: np.array, P: np.ndarray, ins: np.ndarray):
        """
        Compute the update step of the LKF
        :param x: vy vy ax ay yaw rate slip
        :param P: covariance matrix
        :param ins: ax ay yaw rate
        :return: x, P
        """
        self.ax_queue.append(ins[0])
        self.ay_queue.append(ins[1])
        ax = ins[0] - self.ax_bias
        ay = ins[1] - self.ay_bias
        yaw_rate = ins[2]
        x, P = kalman.update(x, P, np.array([ax, ay, yaw_rate]), self.R, self.H)
        return x, P


if __name__ == '__main__':
    lkf = LKF()
    x = np.zeros(lkf.dim_x)
    P = np.eye(lkf.dim_x)
    z = np.ones(lkf.dim_z)
    x, P = lkf.update(x, P, z)
    print(x)
    print(P)
