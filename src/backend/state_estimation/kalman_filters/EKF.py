import numpy as np
from scipy.linalg import expm
from src.backend.state_estimation.config.state_estimation_param import SE_param
from src.backend.state_estimation.config.vehicle_params import VehicleParams


class EKF:
    dt = VehicleParams.dt
    dim_x = SE_param.dim_x
    dim_z = 3

    def __init__(self):
        B = np.eye(self.dim_x)
        self.Q = SE_param.state_transition_noise.copy()
        self.N = B @ self.Q @ B.T * self.dt

    def predict_x(self, x: np.ndarray) -> np.ndarray:
        vx, vy, ax, ay, yaw_rate = x[:5]
        x_pred = x.copy()
        x_pred[0] = vx + (ax + vy * yaw_rate) * self.dt
        x_pred[1] = vy + (ay - vx * yaw_rate) * self.dt
        return x_pred

    def predict(self, x: np.array, P: np.ndarray):
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
        phi = expm(A * self.dt)
        # Predict state and covariance
        x = self.predict_x(x)
        P = phi @ P @ phi.T + self.Q
        return x, P


def get_slip_dot(slip: float, wheel_accel: float, wheel_speed: float, ax: float, ay: float, wheel_delta: float) -> float:
    slip_dot = (wheel_accel / wheel_speed) * (1 - slip) - ax / (VehicleParams.Rw * wheel_speed)
    return slip_dot



class EKF_with_slip_update:
    dt = VehicleParams.dt
    dim_x = SE_param.dim_x
    dim_z = 3

    def __init__(self):
        B = np.eye(self.dim_x)
        self.Q = SE_param.state_transition_noise.copy()
        self.N = B @ self.Q @ B.T * self.dt

    def predict_x(self, x: np.ndarray, wheel_speeds: np.ndarray, wheel_accels: np.ndarray) -> np.ndarray:
        vx, vy, ax, ay, yaw_rate = x[:5]
        x_pred = x.copy()
        x_pred[0] = vx + (ax + vy * yaw_rate) * self.dt
        x_pred[1] = vy + (ay - vx * yaw_rate) * self.dt

        for i in range(4):
            slip = x[5 + i]
            slip_dot = get_slip_dot(slip, wheel_accels[i], wheel_speeds[i], ax)
            x_pred[5 + i] = slip + slip_dot * self.dt
        return x_pred

    def predict(self, x: np.array, P: np.ndarray, wheel_speeds: np.ndarray, wheel_accels: np.ndarray):
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

        for i in range(4):
            A[5 + i, 5 + i] = get_slip_dot(x[5 + i], wheel_accels[i], wheel_speeds[i], x[2])
        phi = expm(A * self.dt)
        # Predict state and covariance
        x = self.predict_x(x)
        P = phi @ P @ phi.T + self.Q
        return x, P

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
