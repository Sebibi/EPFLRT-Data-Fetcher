import numpy as np
import streamlit as st
import pandas as pd

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
            self.memory['data'] = data.copy()

        if len(self.memory['data']) > 0:
            data = self.memory['data']

            # Plot the data
            column_names, samples = plot_data(
                data, self.name, title='X-Estimation observation',
                default_columns=['sensors_vXEst', 'sensors_vYEst', 'sensors_aXEst', 'sensors_aYEst'],
            )

            sampled_data = data.loc[samples[0]:samples[1]]

            # Compute state estimation
            if st.button("Compute state estimation", key=f"{self.name} compute state estimation button"):
                with st.spinner("Computing state estimation..."):
                    sensors_list: list[Sensors] = get_sensors_from_data(sampled_data)
                    estimator_app = StateEstimatorApp()

                    estimations: list = [None for _ in range(len(sensors_list))]
                    progress_bar = st.progress(value=0.0, text=f"Computing state estimation...")
                    for i, sensors in enumerate(sensors_list):
                        state, cov = estimator_app.run(sensors)
                        estimations[i] = state
                        progress_bar.progress((i + 1) / len(sensors_list), text=f"Computing state estimation...")

                        # Update the data
                    columns = SE_param.estimated_states_names
                    data.loc[samples[0]: samples[1], columns] = np.array(estimations)
                    self.memory['data'] = data.copy()

            # Tune the state estimation parameters
            cols_ref = [1, 3]

            st.markdown("## Tune the state estimation parameters")
            cols = st.columns(cols_ref)
            cols[0].markdown("### LKF")
            sub_cols = cols[1].columns(3)
            vx = sub_cols[0].number_input("vx", value=0.09, step=0.01, key=f"{self.name} vx_lkf")
            vy = sub_cols[1].number_input("vy", value=0.09, step=0.01, key=f"{self.name} vy_lkf")
            yaw_rate = sub_cols[2].number_input("yaw rate", value=2.5e-7, step=1e-7, key=f"{self.name} yaw rate_lkf")
            SE_param.set_ins_measurement_noise(vx, vy, yaw_rate)
            st.divider()

            cols = st.columns(cols_ref)
            cols[0].markdown("### LKF vy reset")
            vy_reset = cols[1].number_input("vy reset", value=0.1, step=0.01, key=f"{self.name} vy reset")
            SE_param.set_vy_reset_noise(vy_reset_noise=vy_reset)
            st.divider()

            cols = st.columns(cols_ref)
            cols[0].markdown("### EKF")
            sub_cols = cols[1].columns(2)
            vx = sub_cols[0].number_input("vx", value=0.009, step=0.001, key=f"{self.name} vx_ekf")
            vy = sub_cols[1].number_input("vy", value=0.01, step=0.001, key=f"{self.name} vy_ekf")
            ax = sub_cols[0].number_input("ax", value=0.0004, step=0.0001, key=f"{self.name} ax")
            ay = sub_cols[1].number_input("ay", value=0.00001, step=1e-7, key=f"{self.name} ay")
            w = sub_cols[0].number_input("yaw rate", value=0.00001, step=1e-7, key=f"{self.name} yaw rate_ekf")
            slip = sub_cols[1].number_input("slip ratio", value=0.00001, step=1e-7, key=f"{self.name} slip")
            SE_param.set_state_transition_noise(vx, vy, ax, ay, w, slip)
            st.divider()

            # cols = st.columns(2)
            # cols[0].markdown("UKF")
            # cols[1].number_input("alpha", value=0.001, step=0.001, key=f"{self.name} alpha")
            # cols[1].number_input("beta", value=2., step=0.1, key=f"{self.name} beta")
            # cols[1].number_input("kappa", value=0, step=1, key=f"{self.name} kappa")
            # cols[1].number_input("wheel speed", value=0.1, step=0.01, key=f"{self.name} wheel speed")
            # cols[1].number_input("longitudinal force", value=0.1, step=0.01,
            #                      key=f"{self.name} longitudinal force")


        return True
