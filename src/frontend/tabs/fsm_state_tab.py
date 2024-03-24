import pandas as pd
import streamlit as st
from src.frontend.tabs import Tab
from stqdm import stqdm

from src.frontend.plotting.plotting import plot_data


class FSMStateTab(Tab):

    def __init__(self):
        super().__init__("fsm_state_tab", "FSM State")

        if "fsm_state_sessions" not in self.memory:
            self.memory['fsm_state_sessions'] = []

    def build(self, session_creator):
        if len(st.session_state.fsm_states) > 0:
            fsm_states = st.session_state.fsm_states.copy()
            fsm_states["select"] = [False] * len(fsm_states)
            fsm_states_edit = st.data_editor(fsm_states, use_container_width=True)

            selected_states = fsm_states_edit[fsm_states_edit["select"] == True]
            if len(selected_states) > 0:
                fetch_sessions = st.button("Fetch sessions for selected FSM states")
                st.divider()
                if fetch_sessions:
                    # Fetch data from InfluxDB
                    res = [session_creator.fetch_data2(state['start'], state['end'],
                                                       verify_ssl=st.session_state.verify_ssl)
                           for _, state in stqdm(selected_states.iterrows(), total=len(selected_states))]
                    self.memory['fsm_state_sessions'] = res

                if len(self.memory['fsm_state_sessions']) > 0:
                    max_len = len(self.memory['fsm_state_sessions'])
                    tabs = st.tabs(tabs=[f"{i} - {s['FSM']}" for i, s in selected_states.iloc[:max_len].iterrows()])
                    for tab, data in zip(tabs, self.memory['fsm_state_sessions']):
                        with tab:
                            st.subheader("Plot data")
                            columns, samples = plot_data(
                                data=data,
                                title="FSM State",
                                tab_name=self.name + str(tab),
                                default_columns=['AMS_VBat', 'MISC_VDC_Bus']
                            )

                            st.divider()
                            st.subheader("Raw data")
                            cols = st.columns(2)
                            cols[0].dataframe(data[columns].loc[samples[0]:samples[1]])
                            cols[1].dataframe(data[columns].loc[samples[0]:samples[1]].describe())
