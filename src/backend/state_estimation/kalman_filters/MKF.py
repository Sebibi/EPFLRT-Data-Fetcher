import numpy as np

from src.backend.state_estimation.config.state_estimation_param import SE_param
from src.backend.state_estimation.config.vehicle_params import VehicleParams
from src.backend.state_estimation.kalman_filters.EKF import EKF
from src.backend.state_estimation.kalman_filters.LKF import LKF
from src.backend.state_estimation.kalman_filters.UKF import UKF, UKF_one
from src.backend.state_estimation.attitude_estimation.attitude_estimation import AttitudeEstimation
from src.backend.state_estimation.attitude_estimation.attitude_estimation_simple import AttitudeEstimationSimple
from src.backend.state_estimation.attitude_estimation.attitude_estimator_speed import AttitudeEstimationSpeed



class MKF:

    def __init__(self, independent_updates: bool = False):
        self.dt = VehicleParams.dt
        self.dim_x = SE_param.dim_x
        self.lkf = LKF()
        self.ukf = UKF_one() if independent_updates else UKF()
        self.ekf = EKF()
        self.attitude_ekf = AttitudeEstimationSpeed()

        self.vx_prev = 0
        self.vy_prev = 0
        self.ax_prev = 0
        self.ay_prev = 0
        self.az_prev = 0

    # def update_attitude(self, x: np.ndarray, ins: np.ndarray):
    #     self.attitude_ekf.predict(ins[[4, 3]])
    #     vx = x[0]
    #     vy = x[1]
    #     yaw_rate = x[2]
    #     ax = ins[0] * 0.01 + self.ax_prev * 0.99
    #     ay = ins[1] * 0.01 + self.ay_prev * 0.99
    #     az = -ins[2] * 0.01 + self.az_prev * 0.99
    #     vx_dot = (vx - self.vx_prev) / self.dt
    #     vy_dot = (vy - self.vy_prev) / self.dt
    #     ax_delta = ax - vx_dot + yaw_rate * vy
    #     ay_delta = ay - vy_dot - yaw_rate * vx
#
    #     z = np.array([ax_delta, ay_delta, az])
    #     self.attitude_ekf.update(z)
    #     self.vx_prev = vx
    #     self.vy_prev = vy
    #     self.ax_prev = ax
    #     self.ay_prev = ay
    #     self.az_prev = az
    #     return self.attitude_ekf.x

    def update_attitude(self, x: np.ndarray, ins: np.ndarray):
        ax, ay, az, gyro_x, gyro_y, gyro_z = ins
        vx, vy, ax, ay = x[0], x[1], x[2], x[3]
        z = np.array([ax, ay, az, vx, vy, gyro_z])
        self.attitude_ekf.predict([gyro_y, gyro_x])
        self.attitude_ekf.update(z)
        return self.attitude_ekf.x

    def predict(self, x: np.array, P: np.ndarray, wheel_speeds: np.ndarray, wheel_acc: np.ndarray):
        # x, P = self.ekf.predict(x, P)
        x, P = self.ekf.predict(x, P, wheel_speeds, wheel_acc)
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
        attitude = np.zeros(2) #  self.update_attitude(x, ins)
        # print(attitude)
        self.lkf.update_bias(wheel_speeds=wheel_speeds)
        x, P = self.lkf.update(x, P, ins, attitude)  # Update ax, ay, az, yaw_rate from ins
        x, P = self.lkf.update_vy_reset(x, P)  # Update vy == 0 if steady state
        x, P = self.ukf.update1(x, P, wheel_speeds, steering_deltas)  # Update vx, vy, yaw_rate from wheel speeds
        x, P = self.ukf.update2(x, P, torques, bp, wheel_speeds, wheel_acc)  # Update ax, ay, yaw_rate from wheel acc
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