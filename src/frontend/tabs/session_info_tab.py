import json

import pandas as pd
import streamlit as st
from src.backend.sessions.create_sessions import SessionCreator
from src.backend.data_crud.json_session_info import SessionInfoJsonCRUD, SessionInfo
from src.backend.state_estimation.config.vehicle_params import VehicleParams
from src.frontend.tabs.base import Tab
from config.config import drivers


class SessionInfoTab(Tab):
    def __init__(self):
        super().__init__("Session Info Modification", "Session Info Modification")
        self.crud = st.session_state.session_info_crud

        if "data" not in self.memory:
            self.memory['data'] = pd.DataFrame()

        if "session_info_data" not in self.memory:
            self.memory['session_info_data'] = pd.DataFrame()

    def build(self, session_creator: SessionCreator):

        st.header(self.description)
        datetime_ranges = session_creator.r2d_multi_session_selector(st.session_state.sessions, key=f"{self.name} session selector session info")

        if st.button("Fetch these sessions", key=f"{self.name} fetch data button"):
            session_infos = {key: self.crud.read(key) for key in datetime_ranges}
            self.memory['session_info_data'] = pd.DataFrame(session_infos).T

        if len(self.memory['session_info_data']) > 0:
            session_infos = self.memory['session_info_data']
            df = pd.DataFrame(session_infos)
            new_df = st.data_editor(
                df,
                column_config={
                    "driver": st.column_config.SelectboxColumn(options=list(drivers.keys()), default='Unknown'),
                    "weather_condition": st.column_config.SelectboxColumn(options=['Wet', 'Dry', 'Humid'], default="None"),
                    "control_mode": st.column_config.SelectboxColumn(options=list(VehicleParams.ControlMode.values()), default="None"),
                    "description": st.column_config.TextColumn(help="Enter a description"),
                },
                column_order=["control_mode", "driver", "weather_condition", "description"],
                use_container_width=True,
            )
            if st.button("Save", key=f"{self.name} save button"):
                for key, row in new_df.iterrows():
                    self.crud.create(key, **row.to_dict())






