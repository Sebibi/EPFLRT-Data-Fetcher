import numpy as np

from src.state_estimation.kalman_filters.EKF import EKF
from src.state_estimation.kalman_filters.LKF import LKF
from src.state_estimation.kalman_filters.UKF import UKF


class MKF:

    def __init__(self):
        self.dt = 0.01
        self.dim_x = 9
        self.lkf = LKF()
        self.ukf = UKF()
        self.ekf = EKF()

    def predict(self, x: np.array, P: np.ndarray):
        x, P = self.ekf.predict(x, P)
        return x, P

    def update(
            self,
            x: np.array,
            P: np.ndarray,
            ins: np.ndarray,
            wheel_speeds: np.ndarray,
            steering_deltas: np.array,
            torques: np.ndarray,
            bp: np.ndarray,
            wheel_acc: np.ndarray,
    ):
        x, P = self.lkf.update(x, P, ins)
        x, P = self.ukf.update1(x, P, wheel_speeds, steering_deltas)
        x, P = self.ukf.update2(x, P, torques, bp, wheel_speeds, wheel_acc)
        return x, P
