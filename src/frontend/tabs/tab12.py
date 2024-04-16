import streamlit as st
import pandas as pd
from src.backend.sessions.create_sessions import SessionCreator
from src.frontend.tabs.base import Tab
from src.frontend.plotting.plotting import plot_data_comparaison, plot_data, plot_multiple_data
from src.backend.state_estimation.config.vehicle_params import VehicleParams


class Tab12(Tab):
    motor_torques_cols = [f'VSI_TrqFeedback_{wheel}' for wheel in VehicleParams.wheel_names]
    max_torques_cols = [f'sensors_TC_Tmax_{wheel}' for wheel in VehicleParams.wheel_names]
    min_torques_cols = [f'sensors_TC_Tmin_{wheel}' for wheel in VehicleParams.wheel_names]

    def __init__(self):
        super().__init__(name="tab12", description="Traction Control")

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

            wheels = st.multiselect(
                label="Select the wheel to plot",
                key=f"{self.name} wheel selector",
                options=VehicleParams.wheel_names,
                default=VehicleParams.wheel_names[-1:]
            )
            wheel_ids = [VehicleParams.wheel_names.index(wheel) for wheel in wheels]

            torque_cols = [self.motor_torques_cols[wheel_id] for wheel_id in wheel_ids]
            max_torque_cols = [self.max_torques_cols[wheel_id] for wheel_id in wheel_ids]
            min_torque_cols = [self.min_torques_cols[wheel_id] for wheel_id in wheel_ids]

            cols = torque_cols + max_torque_cols + min_torque_cols
            plot_data(data=data, tab_name=self.name + "TCC", title="Traction Control", default_columns=cols)
