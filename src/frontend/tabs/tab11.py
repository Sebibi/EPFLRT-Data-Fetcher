import streamlit as st
import pandas as pd
from src.backend.sessions.create_sessions import SessionCreator
from src.frontend.tabs.base import Tab
from src.frontend.plotting.plotting import plot_data_comparaison, plot_data, plot_multiple_data
from src.backend.state_estimation.config.vehicle_params import VehicleParams

class Tab11(Tab):
    
    motor_torques_cols = [f'VSI_TrqFeedback_{wheel}' for wheel in VehicleParams.wheel_names]


    def __init__(self):
        super().__init__(name="tab11", description="Torque Allocator")

        if "data" not in self.memory:
            self.memory['data'] = pd.DataFrame()


    def build(self, session_creator: SessionCreator) -> bool:
        st.header(self.description)
        datetime_range = session_creator.r2d_session_selector(st.session_state.sessions, key=f"{self.name} session selector")
        if st.button("Fetch this session", key=f"{self.name} fetch data button"):
            data = session_creator.fetch_data(datetime_range, verify_ssl=st.session_state.verify_ssl)
            data.index = (data.index - data.index[0]).total_seconds()
            self.memory['data'] = data

        if len(self.memory['data']) > 0:
            data = self.memory['data']
            torques = data[self.motor_torques_cols]

            left_torques = torques[[self.motor_torques_cols[0], self.motor_torques_cols[2]]]
            right_torques = torques[[self.motor_torques_cols[1], self.motor_torques_cols[3]]]

            data['VSI_TrqFeedback_sum'] = torques.sum(axis=1)
            data['VSI_TrqFeedback_delta'] = left_torques.sum(axis=1) - right_torques.sum(axis=1)

            # Plot the data
            plot_data(data=data, tab_name=self.name + "TA", title="Torque Allocator", default_columns=['VSI_TrqFeedback_sum', 'VSI_TrqFeedback_delta'])

            # Plot data comparaison
            plot_data_comparaison(data=data, tab_name=self.name + "TC", title="Torque Command", default_columns=['VSI_TrqFeedback_sum', 'sensors_Torque_cmd'])
            plot_data_comparaison(data=data, tab_name=self.name + "TD", title="Torque Delta", default_columns=['VSI_TrqFeedback_delta', 'sensors_TV_delta_torque'])



            
            

            

       