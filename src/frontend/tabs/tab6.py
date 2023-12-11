import numpy as np
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from stqdm import stqdm

from src.backend.state_estimation.config.vehicle_params import VehicleParams
from src.backend.state_estimation.config.state_estimation_param import SE_param, tune_param_input
from src.backend.state_estimation.observe_measurments.create_new_features import create_new_features
from src.backend.state_estimation.observe_measurments.new_features_plots import plot_new_features
from src.backend.state_estimation.observe_measurments.wheel_analysis import plot_wheel_analysis
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
            slip_cols_1000 = [col + '_1000' for col in self.slip_cols]
            data[slip_cols_100] = data[self.slip_cols] * 100
            data[slip_cols_1000] = data[self.slip_cols] * 1000

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

            error_display = cols[1].radio("Raw or description", ["Raw", "Description"], key=f"{self.name} raw or description")
            if error_display == "Raw":
                w_error = pd.DataFrame(estimator_app.mkf.ukf.error_z_w, columns=w_error_names)
                fi_error = pd.DataFrame(estimator_app.mkf.ukf.error_z_fi, columns=fi_error_names)
                cols[1].dataframe(pd.concat([w_error, fi_error], axis=1))
            else:
                w_desc = pd.DataFrame(estimator_app.mkf.ukf.error_z_w, columns=w_error_names).describe(percentiles)
                fi_desc = pd.DataFrame(estimator_app.mkf.ukf.error_z_fi, columns=fi_error_names).describe(percentiles)
                cols[1].dataframe(pd.concat([w_desc, fi_desc], axis=1))

            # Allow to tune the state estimation parameters
            with st.expander("Tune the state estimation parameters"):
                tune_param_input(self.name)

            # Show some data from state estimation
            if st.checkbox("Show measurement transformations"):
                new_data = data.copy()
                with st.spinner("Computing new features..."):
                    new_cols = create_new_features(new_data, self.motor_torques_cols, self.brake_pressure_cols, self.motor_speeds_cols)
                    wheel_acc_cols, long_tire_force_cols, long_tire_force_est_cols_est, normal_force_cols, wheel_speeds_cols_m_s_est, vl_cols = new_cols

                # Plot new features
                wheel_speeds_cols_m_s = [col + '_m_s' for col in self.motor_speeds_cols]
                if st.checkbox("Plot new features"):
                    plot_new_features(
                        new_data, self.name, wheel_acc_cols, long_tire_force_cols, long_tire_force_est_cols_est,
                        normal_force_cols, wheel_speeds_cols_m_s, wheel_speeds_cols_m_s_est, vl_cols
                    )

                if st.checkbox("Plot wheel analysis"):
                    plot_wheel_analysis(
                        new_data, self.name, wheel_acc_cols, long_tire_force_cols, long_tire_force_est_cols_est,
                        normal_force_cols, wheel_speeds_cols_m_s, wheel_speeds_cols_m_s_est, vl_cols, slip_cols_100,
                        slip_cols_1000
                    )
        return True
