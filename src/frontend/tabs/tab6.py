import numpy as np
import streamlit as st
import pandas as pd
from stqdm import stqdm

from src.backend.state_estimation.config.vehicle_params import VehicleParams
from src.backend.state_estimation.config.state_estimation_param import SE_param
from src.backend.state_estimation.state_estimator_app import StateEstimatorApp
from src.backend.state_estimation.measurments.sensors import get_sensors_from_data, Sensors
from src.backend.sessions.create_sessions import SessionCreator
from src.frontend.plotting.plotting import plot_data
from src.frontend.tabs import Tab


class Tab6(Tab):

    def __init__(self):
        super().__init__("tab6", "State Estimation tuning")
        if "data" not in self.memory:
            self.memory['data'] = pd.DataFrame()

    def build(self, session_creator: SessionCreator) -> bool:

        st.header(self.description)
        datetime_range = session_creator.r2d_session_selector(st.session_state.sessions,
                                                              key=f"{self.name} session selector")
        if st.button("Fetch this session", key=f"{self.name} fetch data button"):
            data = session_creator.fetch_data(datetime_range, verify_ssl=st.session_state.verify_ssl)
            data.index = (data.index - data.index[0]).total_seconds()
            data.index = np.array(data.index).round(2)

            # Add gyro data that is in  to deg/s
            gyro_cols = ['sensors_gyroX', 'sensors_gyroY', 'sensors_gyroZ']
            gyro_cols_deg = [col + '_deg' for col in gyro_cols]
            data[gyro_cols_deg] = data[gyro_cols].values * 180 / np.pi

            # Add wheel speeds in m/s
            wheel_speeds_cols = ['VSI_Motor_Speed_FL', 'VSI_Motor_Speed_FR', 'VSI_Motor_Speed_RL', 'VSI_Motor_Speed_RR']
            wheel_speeds_cols_m_s = [col + '_m_s' for col in wheel_speeds_cols]
            data[wheel_speeds_cols_m_s] = data[wheel_speeds_cols] * np.pi * VehicleParams.Rw / (30 * VehicleParams.gear_ratio)
            self.memory['data'] = data.copy()

        if len(self.memory['data']) > 0:
            data = self.memory['data']

            # Convert yaw rate to deg/s
            data['sensors_dpsi_est_deg'] = data['sensors_dpsi_est'] * 180 / np.pi

            # Multiply slip ratios bs 100
            slip_cols = ['sensors_s_FL_est', 'sensors_s_FR_est', 'sensors_s_RL_est', 'sensors_s_RR_est']
            slip_cols_100 = [col + '_100' for col in slip_cols]
            data[slip_cols_100] = data[slip_cols] * 100

            # Plot the data
            column_names, samples = plot_data(
                data, self.name, title='X-Estimation observation',
                default_columns=['sensors_vXEst', 'sensors_vYEst', 'sensors_aXEst', 'sensors_aYEst'],
            )

            st.dataframe(data[column_names].describe().T)

            # Compute state estimation
            if st.button("Compute state estimation", key=f"{self.name} compute state estimation button"):
                with st.spinner("Computing state estimation..."):
                    sensors_list: list[Sensors] = get_sensors_from_data(data.loc[samples[0]:samples[1]])
                    estimator_app = StateEstimatorApp()

                    estimations: list = [None for _ in range(len(sensors_list))]
                    for i, sensors in stqdm(enumerate(sensors_list), total=len(sensors_list)):
                        state, cov = estimator_app.run(sensors)
                        estimations[i] = state

                        # Update the data
                    columns = SE_param.estimated_states_names
                    data.loc[samples[0]: samples[1], columns] = np.array(estimations)
                    self.memory['data'] = data.copy()
                    st.balloons()

            # Tune the state estimation parameters
            cols_ref = [1, 3]

            st.markdown("## Tune the state estimation parameters")
            cols = st.columns(cols_ref)
            cols[0].markdown("### LKF")
            sub_cols = cols[1].columns(3)
            values = SE_param.ins_measurement_noise.diagonal()
            ax = sub_cols[0].number_input("ax", value=values[0], key=f"{self.name} ax_lkf", format="%0.8f")
            ay = sub_cols[1].number_input("ay", value=values[1], key=f"{self.name} ay_lkf", format="%0.8f")
            yaw_rate = sub_cols[2].number_input("yaw rate", value=values[2], key=f"{self.name} yaw rate_lkf", format="%0.8f")
            SE_param.set_ins_measurement_noise(ax, ay, yaw_rate)
            st.divider()

            cols = st.columns(cols_ref)
            cols[0].markdown("### LKF vy reset")
            values = SE_param.vy_reset_noise.diagonal()
            vy_reset = cols[1].number_input("vy reset", value=values[0], key=f"{self.name} vy reset", format="%0.8f")
            SE_param.set_vy_reset_noise(vy_reset_noise=vy_reset)
            st.divider()

            cols = st.columns(cols_ref)
            cols[0].markdown("### EKF")
            sub_cols = cols[1].columns(2)
            values = SE_param.state_transition_noise.diagonal()
            vx = sub_cols[0].number_input("vx", value=values[0], key=f"{self.name} vx_ekf", format="%0.8f")
            vy = sub_cols[1].number_input("vy", value=values[1], key=f"{self.name} vy_ekf", format="%0.8f")
            ax = sub_cols[0].number_input("ax", value=values[2], key=f"{self.name} ax", format="%0.8f")
            ay = sub_cols[1].number_input("ay", value=values[3], key=f"{self.name} ay", format="%0.8f")
            w = sub_cols[0].number_input("yaw rate", value=values[4], key=f"{self.name} yaw rate_ekf", format="%0.8f")
            slip = sub_cols[1].number_input("slip ratio", value=values[5], key=f"{self.name} slip", format="%0.8f")
            SE_param.set_state_transition_noise(vx, vy, ax, ay, w, slip)
            st.divider()

            cols = st.columns(cols_ref)
            cols[0].markdown("### UKF")
            sub_cols = cols[1].columns(3)
            values = [SE_param.alpha, SE_param.beta, SE_param.kappa]
            alpha = sub_cols[0].number_input("alpha", value=values[0], key=f"{self.name} alpha", format="%0.8f")
            beta = sub_cols[1].number_input("beta", value=values[1], key=f"{self.name} beta", format="%0.8f")
            kappa = sub_cols[2].number_input("kappa", value=values[2], key=f"{self.name} kappa", format="%0.8f")
            SE_param.set_sigma_points_param(alpha, beta, kappa)

            sub_cols = cols[1].columns(2)
            values = [SE_param.wheel_speed_measurement_noise[0, 0], SE_param.longitudinal_force_measurement_noise[0, 0]]
            w = sub_cols[0].number_input("wheel speed", value=values[0], key=f"{self.name} wheel speed", format="%0.8f")
            lf = sub_cols[1].number_input("longitudinal force", value=values[1], key=f"{self.name} longitudinal force", format="%0.8f")
            SE_param.set_wheel_speed_measurement_noise(wheel_speed=w)
            SE_param.set_longitudinal_force_measurement_noise(longitudinal_force=lf)

        return True
