import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter

from src.state_estimation.kalman_filters.estimation_transformation.wheel_speed import estimate_wheel_speed, \
    estimate_wheel_speeds
from src.state_estimation.kalman_filters.estimation_transformation.longitudinal_tire_force import \
    estimate_longitudinal_tire_force, estimate_longitudinal_tire_forces
from src.state_estimation.inputs.measurement_transformation.longitudonal_tire_force import \
    measure_tire_longitudinal_force, measure_tire_longitudinal_forces


class UKF:
    dt = 0.01
    dim_x = 9
    dim_z = 4
    wheel_speeds_meas_noise = 1 * np.eye(4)
    longitudinal_meas_noise = 1 * np.eye(4)
    points = MerweScaledSigmaPoints(n=dim_x, alpha=0.1, beta=2., kappa=-1)

    def __init__(self):
        self.ukf = UnscentedKalmanFilter(
            dim_x=self.dim_x,
            dim_z=self.dim_z,
            dt=self.dt,
            fx=lambda x, dt: x,
            hx=estimate_wheel_speeds,
            points=self.points,
        )

    def update1(self, x: np.ndarray, P: np.ndarray, wheel_speeds: np.ndarray, steering_deltas: np.ndarray):
        hx_params = dict(steering_deltas=steering_deltas)
        self.ukf.x = x
        self.ukf.P = P

        # Compute the ukf update
        self.ukf.compute_process_sigmas(dt=self.dt)  # Recompute the sigma points to reflect the new covariance
        self.ukf.update(z=wheel_speeds, hx=estimate_wheel_speeds, R=self.wheel_speeds_meas_noise, **hx_params)
        return self.ukf.x, self.ukf.P

    def update2(self, x: np.ndarray, P: np.ndarray, torques, bp, wheel_speeds, wheel_acc):
        self.ukf.x = x
        self.ukf.P = P

        # Set the measurement noise to a very high value to prevent the filter from diverging
        R = self.longitudinal_meas_noise.copy()
        out_of_bound_slip_ratios_index = np.where(np.abs(x[5:9]) > 0.1)[0]
        for i in out_of_bound_slip_ratios_index:
            R[i, i] = 1e6

        # Compute the ukf update
        self.ukf.compute_process_sigmas(dt=self.dt)  # Recompute the sigma points to reflect the new covariance
        z = measure_tire_longitudinal_forces(torques, bp, wheel_speeds, wheel_acc)
        self.ukf.update(z=z, hx=estimate_longitudinal_tire_forces, R=R)
        return self.ukf.x, self.ukf.P


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    ukf = UKF()
    x = np.array([0, 0, 0, 0, 0, 0.01, 0.01, 0.01, 0.01], dtype=float)
    P = np.eye(ukf.dim_x)
    steering_deltas = np.array([0, 0, 0, 0])
    wheel_speed_meas = np.array([1, 1, 1, 1])
    print(ukf.ukf.P.round(4))

    for i in range(1000):
        print(x.astype(float).round(4))
        print(estimate_wheel_speeds(x, steering_deltas).round(1))
        print()
        x, P = ukf.update1(x, P, wheel_speed_meas, steering_deltas)
    print(P.round(1))

    x = np.array([0, 0, 0, 0, 0, 0.01, 0.01, 0.01, 0.01], dtype=float)
    torques = np.array([0, 0, 0, 0])
    bp = np.array([0, 0, 0, 0])
    wheel_speeds = np.array([0, 0, 0, 0])
    wheel_acc = np.array([0, 0, 0, 0])

    for i in range(1000):
        print(x.astype(float).round(4))
        print(estimate_longitudinal_tire_forces(x).round(1))
        print()
        x, P = ukf.update2(x, P, torques, bp, wheel_speeds, wheel_acc)
