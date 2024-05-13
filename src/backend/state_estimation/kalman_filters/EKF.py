import numpy as np
from scipy.linalg import expm
from src.backend.state_estimation.config.state_estimation_param import SE_param
from src.backend.state_estimation.config.vehicle_params import VehicleParams
import streamlit as st


class EKF:
    dt = VehicleParams.dt
    dim_x = SE_param.dim_x
    dim_z = 3

    def __init__(self):
        B = np.eye(self.dim_x)
        self.Q = SE_param.state_transition_noise.copy()
        self.N = B @ self.Q @ B.T * self.dt

    def predict_x(self, x: np.ndarray, wheel_speed: np.ndarray = None, wheel_accel: np.ndarray = None) -> np.ndarray:
        vx, vy, ax, ay, yaw_rate = x[:5]
        x_pred = x.copy()
        x_pred[0] = vx + (ax + vy * yaw_rate) * self.dt
        x_pred[1] = vy + (ay - vx * yaw_rate) * self.dt
        if wheel_accel is not None:
            for i in range(4):
                slip = x[5 + i]
                slip_dot = get_slip_dot(slip, wheel_accel[i], wheel_speed[i], ax)
                x_pred[5 + i] = slip + slip_dot * self.dt
        return x_pred

    def predict(self, x: np.array, P: np.ndarray, wheel_speeds: np.ndarray = None, wheel_accels: np.ndarray = None):
        """
        Predict the state and covariance matrix with the EKF
        :param x: shape(n,)
        :param P: shape(n,n)
        :return: x, P
        """
        A = np.zeros((self.dim_x, self.dim_x))
        A[0, 1] = x[4]  # yaw_rate
        A[0, 2] = 1
        A[0, 4] = x[1]  # vy
        A[1, 0] = -x[4]  # yaw_rate
        A[1, 3] = 1
        A[1, 4] = -x[0]  # vx

        if wheel_speeds is not None:
            for i in range(4):
                A[5 + i, 2] = get_slip_dot_a_component(x[5 + i], wheel_speeds[i])
                A[5 + i, 5 + i] = get_slip_dot_slip_component(x[5 + i], wheel_speeds[i], wheel_accels[i], x[2])

        phi = expm(A * self.dt)
        # Predict state and covariance
        x = self.predict_x(x)
        P = phi @ P @ phi.T + self.Q

        assert all(A.flatten() != np.nan), "A contains nan values \n {}".format(A)
        assert all(phi.flatten() != np.nan), "phi contains nan values \n {}".format(phi)
        assert all(x.flatten() != np.nan), "x contains nan values \n {}".format(x)
        assert all(P.flatten() != np.nan), "P contains nan values \n {}".format(P)

        # Check for inf
        assert all(A.flatten() != np.inf), "A contains inf values \n {}".format(A)
        assert all(phi.flatten() != np.inf), "phi contains inf values \n {}".format(phi)
        assert all(x.flatten() != np.inf), "x contains inf values \n {}".format(x)
        assert all(P.flatten() != np.inf), "P contains inf values \n {}".format(P)

        # Check for -inf
        assert all(A.flatten() != -np.inf), "A contains -inf values \n {}".format(A)
        assert all(phi.flatten() != -np.inf), "phi contains -inf values \n {}".format(phi)
        assert all(x.flatten() != -np.inf), "x contains -inf values \n {}".format(x)
        assert all(P.flatten() != -np.inf), "P contains -inf values \n {}".format(P)

        print(x)

        return x, P


def get_slip_dot(slip: float, wheel_accel: float, wheel_speed: float, ax: float) -> float:
    if wheel_speed < 1:
        return 0
    slip_dot = (wheel_accel / wheel_speed) * (1 + slip) - (ax / (VehicleParams.Rw * wheel_speed)) * (1 + slip) ** 2
    return slip_dot


def get_slip_dot_a_component(slip: float, wheel_speed: float) -> float:
    if wheel_speed < 1:
        return 0
    return -(1 + slip) ** 2 / (VehicleParams.Rw * wheel_speed)


def get_slip_dot_slip_component(slip: float, wheel_speed: float, wheel_accel: float, ax: float) -> float:
    if wheel_speed < 1:
        return 0
    return (wheel_accel * VehicleParams.Rw - 2 * (1 + slip) * ax) / (VehicleParams.Rw * wheel_speed)


if __name__ == '__main__':
    ekf = EKF()
    x = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0], dtype=float)
    P = np.eye(ekf.dim_x)
    vx = 1
    vy = 1
    yaw_rate = 0
    print(x)
    x, P = ekf.predict(x, P)
    print(x)
    print(P[:4, :4])
    x, P = ekf.predict(x, P)
    print(x)
