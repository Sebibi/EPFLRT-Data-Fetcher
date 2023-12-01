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
                        st.info(np.diag(cov).round(4))

                    # Update the data
                    columns = SE_param.estimated_states_names
                    data.loc[samples[0]: samples[1], columns] = np.array(estimations)
                    self.memory['data'] = data.copy()







