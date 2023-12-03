import numpy as np

from src.backend.state_estimation.config.vehicle_params import VehicleParams


class SE_param:
    estimated_states_names = (["sensors_vXEst", "sensors_vYEst", "sensors_aXEst", "sensors_aYEst", "sensors_dpsi_est"] +
                              [f"sensors_s_{wheel}_est" for wheel in VehicleParams.wheel_names])

    dt = 0.01
    dim_x = 9

    # LKF
    ins_measurement_noise = np.diag([1.0, 1.0, 0.001])
    vy_reset_noise = np.array([[0.1]])

    # EKF
    state_transition_noise = np.diag([1.0, 0.01, 0.01, 0.01, 1.0, 0.001, 0.001, 0.001, 0.0001])

    # UKF
    alpha, beta, kappa = (0.001, 2., 0.)  # Sigma points parameter
    # alpha, beta, kappa = (0.1, 2., -1)  # Sigma points parameter

    wheel_speed_measurement_noise = np.diag([1.0 for _ in range(4)])
    longitudinal_force_measurement_noise = np.diag([1.0 for _ in range(4)])

    @classmethod
    def set_ins_measurement_noise(cls, ax: float, ay: float, yaw_rate: float):
        cls.ins_measurement_noise = np.diag([ax, ay, yaw_rate])

    @classmethod
    def set_vy_reset_noise(cls, vy_reset_noise: float):
        cls.vy_reset_noise = np.array([[vy_reset_noise]])

    @classmethod
    def set_state_transition_noise(cls, vx: float, vy: float, ax: float, ay: float, yaw_rate: float, slip: float):
        cls.state_transition_noise = np.diag([vx, vy, ax, ay, yaw_rate, slip, slip, slip, slip])

    @classmethod
    def set_sigma_points_param(cls, alpha: float, beta: float, kappa: float):
        cls.alpha = alpha
        cls.beta = beta
        cls.kappa = kappa

    @classmethod
    def set_wheel_speed_measurement_noise(cls, wheel_speed: float):
        cls.wheel_speed_measurement_noise = np.diag([wheel_speed for _ in range(4)])

    @classmethod
    def set_longitudinal_force_measurement_noise(cls, longitudinal_force: float):
        cls.longitudinal_force_measurement_noise = np.diag([longitudinal_force for _ in range(4)])
