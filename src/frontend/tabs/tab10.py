import streamlit as st
import pandas as pd
from src.backend.sessions.create_sessions import SessionCreator
from src.frontend.tabs.base import Tab
from src.frontend.plotting.plotting import plot_data_comparaison, plot_data, plot_multiple_data

class Tab10(Tab):
    def __init__(self):
        super().__init__(name="tab10", description="Run comparaison between two sessions")

        if "data" not in self.memory:
            self.memory['data'] = pd.DataFrame()

        if "data2" not in self.memory:
            self.memory['data2'] = pd.DataFrame()

    def build(self, session_creator: SessionCreator) -> bool:
        st.header(self.description)
        cols = st.columns(2)
        with cols[0]:
            datetime_range = session_creator.r2d_session_selector(st.session_state.sessions, key=f"{self.name} session selector")
            if st.button("Fetch this session", key=f"{self.name} fetch data button"):
                data = session_creator.fetch_data(datetime_range, verify_ssl=st.session_state.verify_ssl)
                data.index = (data.index - data.index[0]).total_seconds()
                self.memory['data'] = data

        with cols[1]:
            datetime_range2 = session_creator.r2d_session_selector(st.session_state.sessions, key=f"{self.name} session selector2")
            if st.button("Fetch this session", key=f"{self.name} fetch data button2"):
                data2 = session_creator.fetch_data(datetime_range2, verify_ssl=st.session_state.verify_ssl)
                data2.index = (data2.index - data2.index[0]).total_seconds()
                self.memory['data2'] = data2

        if len(self.memory['data']) > 0 and len(self.memory['data2']) > 0:
            data = self.memory['data']
            data2 = self.memory['data2']

            with cols[0]:
                columns_to_plot, samples_to_plot = plot_data(data=data, tab_name=self.name + "data1", title="Session 1")
            with cols[1]:
                columns_to_plot2, samples_to_plot2 = plot_data(
                    data=data2, tab_name=self.name + "data2", title="Session 2", default_columns=columns_to_plot
                )

            # Get the sub dataframes
            sub_data = data[columns_to_plot].loc[samples_to_plot[0]:samples_to_plot[1]]
            sub_data2 = data2[columns_to_plot2].loc[samples_to_plot2[0]:samples_to_plot2[1]]

            # Re_index the sub dataframes to start from 0
            sub_data.index = sub_data.index - sub_data.index[0]
            sub_data2.index = sub_data2.index - sub_data2.index[0]

            plot_multiple_data(datas=[sub_data, sub_data2], tab_name=self.name + "comparaison", title="Comparaison")
            return True