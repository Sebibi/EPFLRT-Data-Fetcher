import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter

from src.backend.state_estimation.config.state_estimation_param import SE_param
from src.backend.state_estimation.config.vehicle_params import VehicleParams
from src.backend.state_estimation.kalman_filters.estimation_transformation.wheel_speed import estimate_wheel_speeds, \
    estimate_wheel_speed
from src.backend.state_estimation.kalman_filters.estimation_transformation.longitudinal_tire_force import \
    estimate_longitudinal_tire_forces, estimate_longitudinal_tire_force
from src.backend.state_estimation.measurments.measurement_transformation.longitudonal_tire_force import \
    measure_tire_longitudinal_forces


class UKF:
    dt = VehicleParams.dt
    dim_x = SE_param.dim_x
    dim_z = 4
    wheel_speeds_meas_noise = SE_param.wheel_speed_measurement_noise.copy()
    longitudinal_force_meas_noise = SE_param.longitudinal_force_measurement_noise.copy()
    points = MerweScaledSigmaPoints(n=dim_x, alpha=SE_param.alpha, beta=SE_param.beta, kappa=SE_param.kappa)

    error_z_w = []
    error_z_fi = []

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
        """
        Update the UKF with the wheel speed measurements
        :param x: shape(n,)
        :param P: shape(n,n)
        :param wheel_speeds: shape(4,) [fl, fr, rl, rr]
        :param steering_deltas: shape(4,) [fl, fr, rl, rr]
        :return:
        """
        hx_params = dict(steering_deltas=steering_deltas)
        self.ukf.x = x
        self.ukf.P = P

        # Compute the ukf update
        z_est = estimate_wheel_speeds(x, steering_deltas)
        error_z = z_est - wheel_speeds
        self.error_z_w.append(error_z)
        # print(z_est - z, z_est, z)
        self.ukf.compute_process_sigmas(dt=self.dt)  # Recompute the sigma points to reflect the new covariance
        self.ukf.update(z=wheel_speeds, hx=estimate_wheel_speeds, R=self.wheel_speeds_meas_noise, **hx_params)
        # print(np.diag(self.ukf.S))
        # print(np.diag(self.ukf.K).round(4))
        return self.ukf.x, self.ukf.P

    def update2(self, x: np.ndarray, P: np.ndarray, torques, bp, wheel_speeds, wheel_acc):
        """
        Update the UKF with the longitudinal tire force measurements
        :param x: shape(n,)
        :param P: shape(n,n)
        :param torques: shape(4,) [fl, fr, rl, rr]
        :param bp: shape(4,) [fl, fr, rl, rr] => fl == fr, rl == rr
        :param wheel_speeds: shape(4,) [fl, fr, rl, rr]
        :param wheel_acc: shape(4,) [fl, fr, rl, rr]
        :return: x, P
        """
        self.ukf.x = x
        self.ukf.P = P

        # Set the measurement noise to a very high value to prevent the filter from diverging
        R = self.longitudinal_force_meas_noise.copy()
        out_of_bound_slip_ratios_index = np.where(np.abs(x[5:9]) > 0.05)[0]
        for i in out_of_bound_slip_ratios_index:
            R[i, i] = 1e6

        # Compute the ukf update
        self.ukf.compute_process_sigmas(dt=self.dt)  # Recompute the sigma points to reflect the new covariance
        z = measure_tire_longitudinal_forces(torques, bp, wheel_speeds, wheel_acc)
        z_est = estimate_longitudinal_tire_forces(x)
        error_z = z_est - z
        self.error_z_fi.append(error_z)
        # print(error_z, z_est, z_bis, x[5:9].round(3))
        self.ukf.update(z=z, hx=estimate_longitudinal_tire_forces, R=R)
        # print(np.diag(self.ukf.K).round(4))
        # print(np.diag(self.ukf.S))
        return self.ukf.x, self.ukf.P



class UKF_one:
    dt = VehicleParams.dt
    dim_x = SE_param.dim_x
    dim_z = 1
    wheel_speeds_meas_noise = SE_param.wheel_speed_measurement_noise.copy()
    longitudinal_force_meas_noise = SE_param.longitudinal_force_measurement_noise.copy()
    points = MerweScaledSigmaPoints(n=dim_x, alpha=SE_param.alpha, beta=SE_param.beta, kappa=SE_param.kappa)

    error_z_w = []
    error_z_fi = []

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
        """
        Update the UKF with the wheel speed measurements
        :param x: shape(n,)
        :param P: shape(n,n)
        :param wheel_speeds: shape(4,) [fl, fr, rl, rr]
        :param steering_deltas: shape(4,) [fl, fr, rl, rr]
        :return:
        """
        hx_params = dict(steering_deltas=steering_deltas)
        self.ukf.x = x
        self.ukf.P = P
        error_z = np.zeros(4)
        for wheel_id in range(4):
            hx_params['wheel_id'] = wheel_id
            r = self.wheel_speeds_meas_noise[wheel_id, wheel_id]  # measurement noise
            # Compute the ukf update
            z_est = estimate_wheel_speed(x, steering_deltas, wheel_id)
            error_z[wheel_id] = z_est[0] - wheel_speeds[wheel_id]
            wheel_speed = wheel_speeds[wheel_id]
            self.ukf.compute_process_sigmas(dt=self.dt)  # Recompute the sigma points to reflect the new covariance
            self.ukf.update(z=wheel_speed, hx=estimate_wheel_speed, R=r, **hx_params)
        self.error_z_w.append(error_z)
        return self.ukf.x, self.ukf.P

    def update2(self, x: np.ndarray, P: np.ndarray, torques, bp, wheel_speeds, wheel_acc):
        """
        Update the UKF with the longitudinal tire force measurements
        :param x: shape(n,)
        :param P: shape(n,n)
        :param torques: shape(4,) [fl, fr, rl, rr]
        :param bp: shape(4,) [fl, fr, rl, rr] => fl == fr, rl == rr
        :param wheel_speeds: shape(4,) [fl, fr, rl, rr]
        :param wheel_acc: shape(4,) [fl, fr, rl, rr]
        :return: x, P
        """
        self.ukf.x = x
        self.ukf.P = P
        R = self.longitudinal_force_meas_noise.copy()
        z = measure_tire_longitudinal_forces(torques, bp, wheel_speeds, wheel_acc)
        error_z = np.zeros(4)
        for wheel_id in range(4):
            error_z[wheel_id] = 0
            if -0.05 < x[5 + wheel_id] < 0.05: # If the slip ratio is too high, do not update the filter
                # Compute the ukf update
                z_est = estimate_longitudinal_tire_force(x, wheel_id)
                error_z[wheel_id] = z_est - z[wheel_id]
                r = R[wheel_id, wheel_id]  # measurement noise
                self.ukf.compute_process_sigmas(dt=self.dt)  # Recompute the sigma points to reflect the new covariance
                self.ukf.update(z=z[wheel_id], hx=estimate_longitudinal_tire_force, R=r, wheel_id=wheel_id)
        self.error_z_fi.append(error_z)
        return self.ukf.x, self.ukf.P


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    ukf = UKF()
    print(ukf.points)
    x = np.array([3, 0, 1, 0, 0, 0.1, 0.1, 0.1, 0.1], dtype=float)
    P = np.eye(ukf.dim_x)
    steering_deltas = np.array([0, 0, 0, 0])
    wheel_speed_meas = np.array([10, 10, 10, 10])
    wheel_acc_meas = np.array([10, 10, 10, 10])

    torques = np.array([1, 1, 1, 1]) * 200
    bp = np.array([0, 0, 0, 0])

    for i in range(1):
        x, P = ukf.update1(x, P, wheel_speed_meas, steering_deltas)
        print(x.astype(float).round(4))
        print(ukf.ukf.S)
        print(ukf.ukf.P)
        print(ukf.ukf.z)
        print()

    # for i in range(1):
    #     x, P = ukf.update2(x, P, torques, bp, wheel_speed_meas, wheel_acc_meas)
    #     print(x.astype(float).round(4))
    #     print(estimate_longitudinal_tire_forces(x))
    #     print(ukf.ukf.z)
    #     print()


