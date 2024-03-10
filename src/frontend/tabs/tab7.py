import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

from src.backend.sessions.create_sessions import SessionCreator
from src.backend.state_estimation.measurments.measurement_transformation import measure_delta_wheel_angle
from src.backend.torque_vectoring.config_tv import TVParams
from src.frontend.plotting.plotting import plot_data, plot_data_comparaison
from src.frontend.tabs.base import Tab

from src.backend.torque_vectoring.tv_reference import tv_reference, tv_references


class Tab7(Tab):

    def __init__(self):
        super().__init__("tab7", "Torque Vectoring")
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
            steer_angle = self.memory['data']['sensors_steering_angle']

            v_type = ['sensors_vXEst', 'sensors_Best_Vx']

            cols = st.columns(2)
            v_selected = cols[0].radio("Select the velocity type", v_type, index=0)
            k = cols[1].number_input(
                label="K understeer",
                value=0.0, step=0.001, format="%.3f"
            )

            TVParams.set_K_understeer(k)

            v = self.memory['data'][v_selected]
            tv_ref = tv_references(v, steer_angle)
            self.memory['data']['TV_ref'] = tv_ref

            delta_wheels = np.array([measure_delta_wheel_angle(s) for s in steer_angle])
            names = ["delta_wheel_FL", "delta_wheel_FR", "delta_wheel_RL", "delta_wheel_RR"]
            self.memory['data'][names] = delta_wheels

            plot_data(
                data=self.memory['data'],
                tab_name=self.name + "_tv_reference",
                default_columns=["sensors_gyroZ", "TV_ref"] + names[:2],
                title="TV reference",
            )

            plot_data_comparaison(
                data=self.memory['data'],
                tab_name=self.name + "_tv_reference_comparison",
                default_columns=["sensors_gyroZ", "TV_ref"],
                title="TV reference_comparison",
                comparison_names=["Oversteer", "Understeer"]
            )



            # st.dataframe(self.memory['data'][['sensors_vXEst', 'sensors_steering_angle', 'sensors_gyroZ', 'sensors_Best_Vx'] + names[:2]])


