from src.backend.state_estimation.config.vehicle_params import VehicleParams
import numpy as np

from src.backend.state_estimation.kalman_filters.estimation_transformation import estimate_normal_forces, \
    estimate_longitudinal_tire_forces
from src.backend.state_estimation.measurments.measurement_transformation import measure_tire_longitudinal_forces, \
    measure_wheel_acceleration, measure_delta_wheel_angle


class MuEstimator:

    def __init__(self, mu_init, init_cov, measurement_noise, steering_noise) -> None:
        self.mu_init = mu_init
        self.measurement_noise = measurement_noise
        self.steering_noise = steering_noise

        self.mu_error = np.array([0])
        self.mu_error_cov = np.array([[init_cov]])
        self.H = np.ones((4, 1)) / 4

    def get_mu(self) -> (float, float):
        return self.mu_init + self.mu_error[0], self.mu_error_cov[0, 0]

    def update_mu(self, x: np.array, torques: np.array, brakes: np.array, wheel_speeds: np.array, steering: float) -> np.array:
        fzs_est = estimate_normal_forces(x)
        fls_est = estimate_longitudinal_tire_forces(x)
        wheel_acc = measure_wheel_acceleration(wheel_speeds)
        fls_meas = measure_tire_longitudinal_forces(torques, brakes, wheel_speeds, wheel_acc)

        wheel_deltas = measure_delta_wheel_angle(steering)

        # calculate the innovation
        error_obs = (fls_meas - fls_est) / fzs_est

        # Perform and LKF update step
        # calculate the kalman gain and return the updated mu
        R = np.eye(4) * self.measurement_noise + np.abs(np.diag(wheel_deltas) * self.steering_noise)

        kalman_gain = self.mu_error_cov @ self.H.T @ np.linalg.inv(self.H @ self.mu_error_cov @ self.H.T + R)
        self.mu_error = self.mu_error + kalman_gain @ error_obs
        self.mu_error_cov = (1 - kalman_gain @ self.H) @ self.mu_error_cov
        return self.get_mu()


if __name__ == '__main__':

    estimator = MuEstimator(0.8, 0.02, 1, 0.1)
    x = np.array([2, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1])
    torques = np.array([10, 10, 10, 10]) * 21.5
    brakes = np.array([0, 0, 0, 0])
    wheel_speeds = np.array([10, 10, 10, 10])
    steering = 0
    print(estimator.get_mu())
    for i in range(100):
        print(estimator.update_mu(x, torques, brakes, wheel_speeds, steering))



