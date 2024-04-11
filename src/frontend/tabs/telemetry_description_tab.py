import json

import pandas as pd
import streamlit as st
from src.backend.sessions.create_sessions import SessionCreator
from src.backend.data_crud.json_crud import JsonCRUD
from src.frontend.tabs.base import Tab


class TelemetryDescriptionTab(Tab):
    telemetry_description_file_path = "data/telemetry_description.json"
    bucket_data_names = ["AMS", "MISC", "VSI", "sensors"]
    units = list(sorted(["m/s", "rad", "°", "rad/s", "m", "m/s²", "A", "mA", "rpm", "I", "U", "None"]))

    def __init__(self):
        super().__init__("telemetry_description_tab", "Telemetry Description")

        if "data" not in self.memory:
            self.memory['data'] = pd.DataFrame()

        if "telemetry_description" not in self.memory:
            self.memory['telemetry_description_crud'] = JsonCRUD(self.telemetry_description_file_path)

    def build(self, session_creator: SessionCreator):

        st.header(self.description)
        datetime_range = session_creator.r2d_session_selector(st.session_state.sessions,
                                                              key=f"{self.name} session selector")
        if st.button("Fetch this session", key=f"{self.name} fetch data button"):
            data = session_creator.fetch_data(datetime_range, verify_ssl=st.session_state.verify_ssl)
            data.index = (data.index - data.index[0]).total_seconds()
            self.memory['data'] = data

        if len(self.memory['data']) > 0:
            data_bucket = st.selectbox("Select the data bucket", self.bucket_data_names)
            # data_bucket_len = len(data_bucket) + 1
            data_fields = [field for field in self.memory['data'].columns if data_bucket in field]

            crud = self.memory['telemetry_description_crud']
            data = pd.DataFrame([crud.read(data_bucket, field) for field in data_fields], index=data_fields)

            text_mode = st.checkbox("Text mode", key=f"{self.name} text mode")
            text_config = st.column_config.TextColumn(width='small')
            select_box_config = st.column_config.SelectboxColumn(options=self.units, default="None", width='small')
            unit_config = text_config if text_mode else select_box_config
            edited_data = st.data_editor(
                data, use_container_width=True,
                key=f"{self.name} data editor",
                column_config={
                    "unit": unit_config,
                    "description": st.column_config.TextColumn(help="Enter a description", width='large'),
                },
                column_order=["unit", "description"]
            )

            if st.button("Save", key=f"{self.name} save button"):
                for field, row in edited_data.iterrows():
                    crud.create(field, row['unit'], row['description'])
                st.success("Data saved")


            # Download the telemetry description
            st.download_button(
                label="Download telemetry description",
                data=json.dumps(crud.get_raw_data(), indent=4, sort_keys=True),
                file_name="telemetry_description.json",
                mime="application/json"
            )





