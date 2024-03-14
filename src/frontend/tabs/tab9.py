import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

from src.backend.sessions.create_sessions import SessionCreator
from src.backend.state_estimation.config.state_estimation_param import SE_param
from src.backend.state_estimation.config.vehicle_params import VehicleParams
from src.backend.state_estimation.mu_estimation.estimate_mu import MuEstimator
from src.frontend.plotting.plotting import plot_data, plot_data_comparaison
from src.frontend.tabs.base import Tab

from src.backend.torque_vectoring.tv_reference import tv_reference, tv_references

from src.backend.state_estimation.slip_angle_estimation.slip_angle_estimator import EKF_slip_angle


class Tab9(Tab):
    brake_pressure_cols = ['sensors_brake_pressure_L' for _ in range(4)]
    motor_torques_cols = [f'VSI_TrqFeedback_{wheel}' for wheel in VehicleParams.wheel_names]
    motor_speeds_cols = [f'VSI_Motor_Speed_{wheel}' for wheel in VehicleParams.wheel_names]
    steering_angle_cols = 'sensors_steering_angle'
    state_estimation_cols = ['']

    def __init__(self):
        super().__init__("tab9", "Slip Angle Estimation")
        if "data" not in self.memory:
            self.memory['data'] = pd.DataFrame()


    def build(self, session_creator: SessionCreator) -> bool:
        st.header(self.description)
        datetime_range = session_creator.r2d_session_selector(st.session_state.sessions,
                                                              key=f"{self.name} session selector")
        if st.button("Fetch this session", key=f"{self.name} fetch data button"):
            data = session_creator.fetch_data(datetime_range, verify_ssl=st.session_state.verify_ssl)
            data.index = (data.index - data.index[0]).total_seconds()
            self.memory['data'] = data


        if len(self.memory['data']) > 0:
            data = self.memory['data']

            plot_data(
                data=data,
                tab_name=self.name + "state_estimation",
                default_columns=SE_param.estimated_states_names,
                title="Slip Angle estimation",
            )

            ekf_slip_angle = EKF_slip_angle()

            steering = data[self.steering_angle_cols].values
            axs = data['sensors_aXEst'].values
            ays = data['sensors_aYEst'].values

            all_states = np.zeros((len(data), 3))
            for i in range(len(data)):
                state, cov = ekf_slip_angle.predict_update(steering[i], axs[i], ays[i])
                all_states[i] = state.reshape(3)

            data['beta_est'] = all_states[:, 1]
            data['gyroZ_est'] = all_states[:, 0]
            data['vX_est'] = all_states[:, 2]

            st.dataframe(data[['beta_est', 'gyroZ_est', 'vX_est']])

            plot_data_comparaison(
                data=data,
                tab_name=self.name + "yaw rate estimation",
                default_columns=['sensors_gyroZ', 'gyroZ_est'],
                title="yaw rate estimation",
            )

            plot_data(
                data=data,
                tab_name=self.name + "beta estimation",
                default_columns=["beta_est"],
                title="beta estimation",
            )












