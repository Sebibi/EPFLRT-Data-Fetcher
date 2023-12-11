import numpy as np
import pandas as pd

from src.backend.state_estimation.config.state_estimation_param import SE_param
from src.backend.state_estimation.config.vehicle_params import VehicleParams
from src.backend.state_estimation.kalman_filters.estimation_transformation import estimate_longitudinal_tire_forces, \
    estimate_normal_forces, estimate_longitudinal_velocities, estimate_wheel_speeds
from src.backend.state_estimation.measurments.measurement_transformation import measure_wheel_acceleration, \
    measure_delta_wheel_angle, measure_tire_longitudinal_forces


def create_new_features(new_data: pd.DataFrame, motor_torques_cols: list[str], brake_pressure_cols: list[str],
                        motor_speeds_cols: list[str]):
    wheel_acc_cols = [f'wheel_acc_{wheel}' for wheel in VehicleParams.wheel_names]
    wheel_speeds_cols_m_s = [col + '_m_s' for col in motor_speeds_cols]

    # Add 0 line
    new_data['zero'] = 0

    # Reset wheel acceleration
    for i in range(30):
        measure_wheel_acceleration(wheel_speeds=np.array([0, 0, 0, 0], dtype=float))
    new_data[wheel_acc_cols] = [
        measure_wheel_acceleration(wheel_speeds=wheel_speeds)
        for wheel_speeds in new_data[wheel_speeds_cols_m_s].values
    ]

    # Compute delta wheel angle
    new_data[['delta_FL', 'delta_FR']] = [
        measure_delta_wheel_angle(steering_angle=steering_angle)[:2]
        for steering_angle in new_data['sensors_steering_angle'].values
    ]
    new_data[['delta_FL_deg', 'delta_FR_deg']] = new_data[['delta_FL', 'delta_FR']].values * 180 / np.pi

    # Compute longitudinal tire forces
    long_tire_force_cols = [f'long_tire_force_{wheel}' for wheel in VehicleParams.wheel_names]
    new_data[long_tire_force_cols] = [
        measure_tire_longitudinal_forces(torques=torques, bps=bps, wheel_speeds=wheel_speeds,
                                         wheel_acc=wheel_acc)
        for torques, bps, wheel_speeds, wheel_acc in
        zip(new_data[motor_torques_cols].values, new_data[brake_pressure_cols].values,
            new_data[wheel_speeds_cols_m_s].values, new_data[wheel_acc_cols].values)
    ]

    state_estimation_cols = SE_param.estimated_states_names

    # Compute estimated longitudinal tire forces
    long_tire_force_est_cols_est = [name + '_est' for name in long_tire_force_cols]
    new_data[long_tire_force_est_cols_est] = [
        estimate_longitudinal_tire_forces(state_estimation)
        for state_estimation in new_data[state_estimation_cols].values
    ]

    # Compute estimated normal forces
    normal_force_cols = [f'Fz_{wheel}' for wheel in VehicleParams.wheel_names]
    new_data[normal_force_cols] = [
        estimate_normal_forces(state_estimation)
        for state_estimation in new_data[state_estimation_cols].values
    ]

    # Compute estimated wheel speeds
    wheel_speeds_cols_m_s_est = [col + '_m_s_est' for col in motor_speeds_cols]
    new_data[wheel_speeds_cols_m_s_est] = [
        estimate_wheel_speeds(state_estimation,
                              measure_delta_wheel_angle(steering_angle=steering_angle)) * VehicleParams.Rw
        for state_estimation, steering_angle in
        zip(new_data[state_estimation_cols].values, new_data['sensors_steering_angle'].values)
    ]

    # Compute estimated longitudinal velocity
    vl_cols = [f'vl_{wheel}' for wheel in VehicleParams.wheel_names]
    new_data[vl_cols] = [
        estimate_longitudinal_velocities(state_estimation, measure_delta_wheel_angle(steering_angle=steering_angle))
        for state_estimation, steering_angle in
        zip(new_data[state_estimation_cols].values, new_data['sensors_steering_angle'].values)
    ]

    return wheel_acc_cols, long_tire_force_cols, long_tire_force_est_cols_est, normal_force_cols, wheel_speeds_cols_m_s_est, vl_cols