import numpy as np
import pandas as pd

from src.backend.state_estimation.config.state_estimation_param import SE_param
from src.backend.state_estimation.config.vehicle_params import VehicleParams
from src.backend.state_estimation.kalman_filters.estimation_transformation import estimate_longitudinal_tire_forces, \
    estimate_normal_forces, estimate_longitudinal_velocities, estimate_wheel_speeds
from src.backend.state_estimation.measurments.measurement_transformation import measure_wheel_acceleration, \
    measure_delta_wheel_angle, measure_tire_longitudinal_forces, measure_acc_fsum
from src.backend.state_estimation.measurments.measurement_transformation.vy_difference import measure_velocities


def create_new_features(
        new_data: pd.DataFrame,
        motor_torques_cols: list[str],
        brake_pressure_cols: list[str],
):
    new_data['zero'] = 0  # Add 0 line

    # Reset wheel acceleration
    for i in range(30):
        measure_wheel_acceleration(wheel_speeds=np.array([0, 0, 0, 0], dtype=float))

    ws_cols = [f'vWheel_{wheel}' for wheel in VehicleParams.wheel_names]
    dws_cols = [f'accWheel_{wheel}' for wheel in VehicleParams.wheel_names]
    new_data[dws_cols] = [
        measure_wheel_acceleration(wheel_speeds=wheel_speeds)
        for wheel_speeds in new_data[ws_cols].values
    ]

    # Compute delta wheel angle
    new_data[['delta_FL', 'delta_FR']] = [
        measure_delta_wheel_angle(steering_angle=steering_angle)[:2]
        for steering_angle in new_data['sensors_steering_angle'].values
    ]
    new_data[['delta_FL_deg', 'delta_FR_deg']] = np.rad2deg(new_data[['delta_FL', 'delta_FR']].values)

    # Compute longitudinal tire forces
    fl_cols = [f'Fl_{wheel}' for wheel in VehicleParams.wheel_names]
    new_data[fl_cols] = [
        measure_tire_longitudinal_forces(torques=torques, bps=bps, wheel_speeds=wheel_speeds,
                                         wheel_acc=wheel_acc)
        for torques, bps, wheel_speeds, wheel_acc in
        zip(new_data[motor_torques_cols].values, new_data[brake_pressure_cols].values,
            new_data[ws_cols].values, new_data[dws_cols].values)
    ]

    state_estimation_cols = SE_param.estimated_states_names

    # Compute estimated longitudinal tire forces
    fl_est_cols = [name + '_est' for name in fl_cols]
    new_data[fl_est_cols] = [
        estimate_longitudinal_tire_forces(state_estimation)
        for state_estimation in new_data[state_estimation_cols].values
    ]

    # Compute estimated normal forces
    fz_est_cols = [f'Fz_{wheel}_est' for wheel in VehicleParams.wheel_names]
    new_data[fz_est_cols] = [
        estimate_normal_forces(state_estimation)
        for state_estimation in new_data[state_estimation_cols].values
    ]

    # Compute estimated wheel speeds
    ws_est_cols = [col + '_est' for col in ws_cols]
    new_data[ws_est_cols] = [
        estimate_wheel_speeds(state_estimation,
                              measure_delta_wheel_angle(steering_angle=steering_angle)) * VehicleParams.Rw
        for state_estimation, steering_angle in
        zip(new_data[state_estimation_cols].values, new_data['sensors_steering_angle'].values)
    ]

    # Compute estimated longitudinal velocity
    vl_est_cols = [f'vL_{wheel}_est' for wheel in VehicleParams.wheel_names]
    new_data[vl_est_cols] = [
        estimate_longitudinal_velocities(state_estimation, measure_delta_wheel_angle(steering_angle=steering_angle))
        for state_estimation, steering_angle in
        zip(new_data[state_estimation_cols].values, new_data['sensors_steering_angle'].values)
    ]

    # Compute measures acc from longitudinal force sum
    acc_fsum_cols = ['accX_Fsum', 'accY_Fsum']
    new_data[acc_fsum_cols] = [
        measure_acc_fsum(long_tire_forces=long_tire_forces, wheel_deltas=wheel_deltas, state=state)
        for long_tire_forces, wheel_deltas, state in zip(new_data[fl_cols].values, new_data[['delta_FL', 'delta_FR', 'zero', 'zero']].values, new_data[state_estimation_cols].values)
    ]

    acc_fsum_est_cols = ['accX_Fsum_est', 'accY_Fsum_est']
    new_data[acc_fsum_est_cols] = [
        measure_acc_fsum(long_tire_forces=long_tire_forces, wheel_deltas=wheel_deltas, state=state)
        for long_tire_forces, wheel_deltas, state in
        zip(new_data[fl_est_cols].values, new_data[['delta_FL', 'delta_FR', 'zero', 'zero']].values,
            new_data[state_estimation_cols].values)
    ]

    v_adiff = ['vX_acc_diff', 'vY_acc_diff']
    new_data[v_adiff] = [
        measure_velocities(long_tire_forces=long_tire_forces, wheel_deltas=wheel_deltas, state=state)
        for long_tire_forces, wheel_deltas, state in
        zip(new_data[fl_cols].values, new_data[['delta_FL', 'delta_FR', 'zero', 'zero']].values,
            new_data[state_estimation_cols].values)
    ]

    return ws_cols, dws_cols, fl_cols, fl_est_cols, fz_est_cols, ws_est_cols, vl_est_cols, acc_fsum_cols, acc_fsum_est_cols, v_adiff
