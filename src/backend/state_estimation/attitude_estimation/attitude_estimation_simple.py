import numpy as np
from filterpy.kalman import ExtendedKalmanFilter

from src.backend.state_estimation.config.state_estimation_param import SE_param
from src.backend.state_estimation.config.vehicle_params import VehicleParams


class AttitudeEstimationSimple:
    dt = 0.01
    dim_x = 2
    dim_z = 2

    Q = np.eye(dim_x) * dt * 0.1
    R = np.eye(dim_z) * 10000

    F = np.eye(dim_x)
    H = np.eye(dim_z)

    def __init__(self):
        self.x = np.array([0, 0])
        self.P = np.eye(self.dim_x) * 0.00001 * self.dt

    @classmethod
    def f(cls, x: np.array, u: np.array):
        """
        Compute the next state of the system
        :param x: [pitch, roll]
        :param u: [pitch_rate, roll_rate]
        :return: x
        """
        pitch_next = x[0] + cls.dt * u[0]
        roll_next = x[1] + cls.dt * u[1]
        return np.array([pitch_next, roll_next])

    @classmethod
    def h(cls, x: np.array):
        """
        Compute the observation of the system
        :param x: [pitch, roll]
        :return: [pitch, roll]
        """
        return x

    @classmethod
    def get_z(self, z: np.array):
        """
        Get the observation of the system
        :param z: [ax, ay, az]
        :return: [pitch, roll]
        """
        ax, ay, az = z
        pitch = np.arctan2(ax, np.sqrt(ay**2 + az**2))
        roll = np.arctan2(-ay, np.sqrt(ax**2 + az**2))
        return np.array([pitch, roll])

    def predict(self, u: np.array):
        """
        Predict the state and covariance matrix with the EKF
        :param u: [pitch_rate, roll_rate]
        """
        self.x = self.f(self.x, u)
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.array):
        """
        Update the state and covariance matrix with the EKF
        :param z: [ax_delta, ay_delta az]
        """
        x = self.x
        P = self.P

        y = self.get_z(z) - self.h(x)
        S = self.H @ P @ self.H.T + self.R
        K = P @ self.H.T @ np.linalg.inv(S)
        self.x = x + K @ y
        self.P = (np.eye(self.dim_x) - K @ self.H) @ P

    def predict_update(self, u: np.array, z: np.array):
        """
        Predict and update the state and covariance matrix with the EKF
        :param u: [pitch_rate, roll_rate]
        :param z: [az]
        :return: x, P (state and covariance)
        """
        self.predict(u)
        self.update(z)

        return self.x, self.P


if __name__ == '__main__':

    estimator = AttitudeEstimationSimple()
    u = np.array([0.01, 0.1])
    z = np.array([0, 0, 9.81])

    for i in range(100 * 100):
        estimator.predict_update(u, z)
        print(estimator.x)
        print(estimator.P)
        print(estimator.h(estimator.x))
        print()
        input()
