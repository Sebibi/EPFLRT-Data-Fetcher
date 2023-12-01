import numpy as np

from src.backend.state_estimation.config.state_estimation_param import SE_param
from src.backend.state_estimation.config.vehicle_params import VehicleParams
from src.backend.state_estimation.kalman_filters.EKF import EKF
from src.backend.state_estimation.kalman_filters.LKF import LKF
from src.backend.state_estimation.kalman_filters.UKF import UKF


class MKF:

    def __init__(self):
        self.dt = VehicleParams.dt
        self.dim_x = SE_param.dim_x
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
        x, P = self.lkf.update(x, P, ins)  # Update ax, ay, yaw_rate from ins
        x, P = self.lkf.update_vy_reset(x, P)  # Update vy == 0 if steady state
        # x, P = self.ukf.update1(x, P, wheel_speeds, steering_deltas)  # Update vx, vy, yaw_rate from wheel speeds
        # x, P = self.ukf.update2(x, P, torques, bp, wheel_speeds, wheel_acc)  # Update ax, ay, yaw_rate from wheel acc
        return x, P


if __name__ == '__main__':
    mkf = MKF()
    x = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    P = np.eye(mkf.dim_x)

    ins = np.array([0, 0, 0], dtype=float)
    wheel_speeds = np.array([0, 0, 0, 0], dtype=float)
    steering_deltas = np.array([0, 0, 0, 0], dtype=float)
    torques = np.array([0, 0, 0, 0], dtype=float)
    bp = np.array([0, 0, 0, 0], dtype=float)
    wheel_acc = np.array([0, 0, 0, 0], dtype=float)

    x, P = mkf.predict(x, P)
    print(x)

    x, P = mkf.update(x, P, ins, wheel_speeds, steering_deltas, torques, bp, wheel_acc)
    print(x)