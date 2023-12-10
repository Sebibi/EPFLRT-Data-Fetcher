import numpy as np
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from stqdm import stqdm

from src.backend.state_estimation.config.vehicle_params import VehicleParams
from src.backend.state_estimation.config.state_estimation_param import SE_param, tune_param_input
from src.backend.state_estimation.state_estimator_app import StateEstimatorApp
from src.backend.state_estimation.measurments.measurement_transformation.wheel_speed import measure_wheel_speeds
from src.backend.state_estimation.measurments.sensors import get_sensors_from_data, Sensors
from src.backend.sessions.create_sessions import SessionCreator
from src.frontend.plotting.plotting import plot_data
from src.frontend.tabs import Tab


class Tab6(Tab):
    brake_pressure_cols = ['sensors_brake_pressure_L' for _ in range(4)]
    motor_torques_cols = [f'VSI_TrqFeedback_{wheel}' for wheel in VehicleParams.wheel_names]
    motor_speeds_cols = [f'VSI_Motor_Speed_{wheel}' for wheel in VehicleParams.wheel_names]
    slip_cols = [f'sensors_s_{wheel}_est' for wheel in VehicleParams.wheel_names]
    dpsi_col = 'sensors_dpsi_est'

    def __init__(self):
        super().__init__("tab6", "State Estimation tuning")
        if "data" not in self.memory:
            self.memory['data'] = pd.DataFrame()

    def build(self, session_creator: SessionCreator) -> bool:

        st.header(self.description)
        datetime_range = session_creator.r2d_session_selector(
            st.session_state.sessions, key=f"{self.name} session selector"
        )
        if st.button("Fetch this session", key=f"{self.name} fetch data button"):
            data = session_creator.fetch_data(datetime_range, verify_ssl=st.session_state.verify_ssl)
            data.index = (data.index - data.index[0]).total_seconds()
            data.index = np.array(data.index).round(2)

            # Drop all AMS data
            data = data.drop(columns=[col for col in data.columns if 'AMS' in col])

            # Add gyro data that is in  to deg/s
            gyro_cols = ['sensors_gyroX', 'sensors_gyroY', 'sensors_gyroZ']
            gyro_cols_deg = [col + '_deg' for col in gyro_cols]
            data[gyro_cols_deg] = data[gyro_cols].values * 180 / np.pi

            # Add wheel speeds in m/s
            wheel_speeds_cols_m_s = [col + '_m_s' for col in self.motor_speeds_cols]
            data[wheel_speeds_cols_m_s] = measure_wheel_speeds(data[self.motor_speeds_cols].values) * VehicleParams.Rw

            # Add wheel slips and dpsi if not present
            if 'sensors_s_FL_est' not in data.columns:
                data[self.slip_cols + [self.dpsi_col]] = 0
            self.memory['data'] = data.copy()

        if len(self.memory['data']) > 0:
            data = self.memory['data']

            # Convert yaw rate to deg/s
            data['sensors_dpsi_est_deg'] = data[self.dpsi_col] * 180 / np.pi

            # Multiply slip ratios bs 100
            slip_cols_100 = [col + '_100' for col in self.slip_cols]
            data[slip_cols_100] = data[self.slip_cols] * 100

            # Plot the data
            column_names, samples = plot_data(
                data, self.name, title='X-Estimation observation',
                default_columns=['sensors_vXEst', 'sensors_vYEst', 'sensors_aXEst', 'sensors_aYEst'],
            )

            cols = st.columns(2)
            cols[0].subheader("Data description")
            cols[0].dataframe(data[column_names].describe().T)

            # Compute state estimation
            independent_updates = st.checkbox("Independent updates", key=f"{self.name} independent updates",
                                              value=False)
            estimator_app = StateEstimatorApp(independent_updates=independent_updates)
            if st.button("Compute state estimation", key=f"{self.name} compute state estimation button"):
                with st.spinner("Computing state estimation..."):
                    sensors_list: list[Sensors] = get_sensors_from_data(data.loc[samples[0]:samples[1]])
                    estimator_app = StateEstimatorApp(independent_updates=independent_updates)
                    estimations = [estimator_app.run(sensors)[0] for sensors in stqdm(sensors_list)]

                    # Update the data
                    columns = SE_param.estimated_states_names
                    data.loc[samples[0]: samples[1], columns] = np.array(estimations)
                    self.memory['data'] = data.copy()
                    st.balloons()
            cols[1].subheader("Estimated - Measured error description")
            w_error_names = [f"w_speed {wheel}" for wheel in VehicleParams.wheel_names]
            fi_error_names = [f"Fi_{wheel}" for wheel in VehicleParams.wheel_names]
            percentiles = [0.01, 0.05, 0.1, 0.9, 0.95, 0.99]
            w_desc = pd.DataFrame(estimator_app.mkf.ukf.error_z_w, columns=w_error_names).describe(percentiles).T
            fi_desc = pd.DataFrame(estimator_app.mkf.ukf.error_z_fi, columns=fi_error_names).describe(percentiles).T
            cols[1].dataframe(pd.concat([w_desc, fi_desc], axis=0))

            tune_param_input(self.name)

            ###################################################################################################

            if st.checkbox("Show measurement transformations"):
                from src.backend.state_estimation.measurments.measurement_transformation import \
                    measure_wheel_acceleration, measure_delta_wheel_angle, \
                    measure_tire_longitudinal_forces

                from src.backend.state_estimation.kalman_filters.estimation_transformation import \
                    estimate_longitudinal_tire_forces, estimate_wheel_speeds, estimate_normal_forces, \
                    estimate_longitudinal_velocities
                new_data = data.copy()
                wheel_acc_cols = [f'wheel_acc_{wheel}' for wheel in VehicleParams.wheel_names]
                wheel_speeds_cols_m_s = [col + '_m_s' for col in self.motor_speeds_cols]

                # Add 0 line
                new_data['zero'] = 0

                # Reset wheel acceleration
                for i in range(30):
                    measure_wheel_acceleration(wheel_speeds=np.array([0, 0, 0, 0], dtype=float))
                new_data[wheel_acc_cols] = [
                    measure_wheel_acceleration(wheel_speeds=wheel_speeds)
                    for wheel_speeds in data[wheel_speeds_cols_m_s].values
                ]

                # Compute delta wheel angle
                new_data[['delta_FL', 'delta_FR']] = [
                    measure_delta_wheel_angle(steering_angle=steering_angle)[:2]
                    for steering_angle in data['sensors_steering_angle'].values
                ]
                new_data[['delta_FL_deg', 'delta_FR_deg']] = new_data[['delta_FL', 'delta_FR']].values * 180 / np.pi

                # Compute longitudinal tire forces
                long_tire_force_name = [f'long_tire_force_{wheel}' for wheel in VehicleParams.wheel_names]
                new_data[long_tire_force_name] = [
                    measure_tire_longitudinal_forces(torques=torques, bps=bps, wheel_speeds=wheel_speeds,
                                                     wheel_acc=wheel_acc)
                    for torques, bps, wheel_speeds, wheel_acc in
                    zip(data[self.motor_torques_cols].values, data[self.brake_pressure_cols].values,
                        data[wheel_speeds_cols_m_s].values, new_data[wheel_acc_cols].values)
                ]

                state_estimation_cols = SE_param.estimated_states_names

                # Compute estimated longitudinal tire forces
                long_tire_force_est_name_est = [name + '_est' for name in long_tire_force_name]
                new_data[long_tire_force_est_name_est] = [
                    estimate_longitudinal_tire_forces(state_estimation)
                    for state_estimation in data[state_estimation_cols].values
                ]

                # Compute estimated normal forces
                normal_force_name = [f'Fz_{wheel}' for wheel in VehicleParams.wheel_names]
                new_data[normal_force_name] = [
                    estimate_normal_forces(state_estimation)
                    for state_estimation in data[state_estimation_cols].values
                ]

                # Compute estimated longitudinal tire forces
                new_data[long_tire_force_name] = [
                    estimate_longitudinal_tire_forces(state_estimation)
                    for state_estimation in new_data[state_estimation_cols].values
                ]

                # Compute estimated wheel speeds
                wheel_speeds_cols_m_s_est = [col + '_m_s_est' for col in self.motor_speeds_cols]
                new_data[wheel_speeds_cols_m_s_est] = [
                    estimate_wheel_speeds(state_estimation, measure_delta_wheel_angle(steering_angle=steering_angle)) * VehicleParams.Rw
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

                # Plot the data
                with st.expander("Steering deltas"):
                    plot_data(
                        new_data, self.name + "_steering", title='Steering wheel and tires',
                        default_columns=['delta_FL_deg', 'delta_FR_deg', 'sensors_steering_angle'],
                    )
                with st.expander("Wheel accelerations"):
                    plot_data(
                        new_data, self.name + "_acceleration", title='Accelerations observation',
                        default_columns=['sensors_accX'] + wheel_acc_cols[2:],
                    )

                with st.expander("Longitudinal tire forces"):
                    plot_data(
                        new_data, self.name + "_long_tire_force", title='Longitudinal Tire Force',
                        default_columns=long_tire_force_name,
                    )
                with st.expander("Estimated longitudinal tire forces"):
                    plot_data(
                        new_data, self.name + "_long_tire_force_est", title='Longitudinal Tire Force Estimation',
                        default_columns=long_tire_force_name + long_tire_force_est_name_est,
                    )

                with st.expander("Normal forces"):
                    plot_data(
                        new_data, self.name + "_normal_force", title='Normal Force',
                        default_columns=normal_force_name + ['zero', 'sensors_steering_angle', 'sensors_gyroZ_deg'],
                    )

                with st.expander("Estimated Wheel speeds"):
                    plot_data(
                        new_data, self.name + "_wheel_speeds", title='Wheel speeds',
                        default_columns=wheel_speeds_cols_m_s + wheel_speeds_cols_m_s_est,
                    )
                with st.expander("Longitudinal velocities"):

                    plot_data(
                        new_data, self.name + "_long_velocities", title='Longitudinal velocities',
                        default_columns=[vl_cols[i] for i in [0, 2, 1, 3]] + ['zero', 'sensors_gyroZ'],
                    )

        return True
