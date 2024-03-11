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


class Tab8(Tab):
    brake_pressure_cols = ['sensors_brake_pressure_L' for _ in range(4)]
    motor_torques_cols = [f'VSI_TrqFeedback_{wheel}' for wheel in VehicleParams.wheel_names]
    motor_speeds_cols = [f'VSI_Motor_Speed_{wheel}' for wheel in VehicleParams.wheel_names]
    steering_angle_cols = 'sensors_steering_angle'

    def __init__(self):
        super().__init__("tab8", "Mu Estimation")
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
                tab_name=self.name + "_mu_estimation",
                default_columns=["sensors_gyroZ"],
                title="Mu estimation",
            )

            mu_estimator = MuEstimator(mu_init=1.2, init_cov=0.02, measurement_noise=1, steering_noise=0.1)

            torques = data[self.motor_torques_cols].values
            brakes = data[self.brake_pressure_cols].values
            wheel_speeds = data[self.motor_speeds_cols].values
            steering = data[self.steering_angle_cols].values

            states = data[SE_param.estimated_states_names].values

            mu = np.zeros(len(data))
            mu_var = np.zeros(len(data))
            for i in range(len(data)):
                mu[i], mu_var[i] = mu_estimator.update_mu(x=states[i], torques=torques[i], brakes=brakes[i], wheel_speeds=wheel_speeds[i], steering=steering[i])


            data['mu'] = mu
            data['mu_var'] = mu_var

            plot_data(
                data=data,
                tab_name=self.name + "_mu_estimation2",
                default_columns=["mu", "mu_var"],
                title="Mu estimation2",
            )




