from datetime import datetime, timedelta
from typing import List

import pandas as pd
import streamlit as st

from config.config import ConfigLogging, ConfigLive, FSM
from src.backend.api_call.influxdb_api import InfluxDbFetcher
from src.backend.sessions.create_sessions import SessionCreator
from src.frontend.tabs import create_tabs, Tab, FSMStateTab
from stqdm import stqdm


def init_sessions_state():
    if "sessions" not in st.session_state:
        st.session_state.sessions = []

    if "fetcher" not in st.session_state:
        st.session_state.fetcher = InfluxDbFetcher(config=ConfigLogging)

    if "session_creator" not in st.session_state:
        st.session_state.session_creator = SessionCreator(fetcher=st.session_state.fetcher)

    if "verify_ssl" not in st.session_state:
        st.session_state.verify_ssl = True

    if "fsm_states" not in st.session_state:
        st.session_state.fsm_states = pd.DataFrame()


if __name__ == '__main__':

    # Initialize the application
    st.set_page_config(layout="wide")
    init_sessions_state()

    cols = st.columns([5, 1])
    cols[0].title("InfluxDB data Fetcher")
    cols[0].markdown("This application allows you to fetch data from the InfluxDB database of the EPFL Racing Team")
    cols[1].image("data/img/epflrt_logo.png", width=200)

    with st.sidebar:
        # Choose date range
        st.header("Select a date range")
        date_cols = st.columns(2)
        start_date = date_cols[0].date_input("Start date", value=pd.to_datetime("2023-11-18"),
                                             max_value=pd.to_datetime(datetime.now().strftime("%Y-%m-%d")))
        end_date = date_cols[1].date_input("End date", value=pd.to_datetime("2023-11-19"),
                                           max_value=pd.to_datetime(
                                               (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")))

        # Enable / Disable SSL verification
        st.session_state.verify_ssl = st.checkbox("Fetch with SSL", value=True)
        if st.checkbox("Fetch Live Data", value=False):
            st.session_state.fetcher = InfluxDbFetcher(config=ConfigLive)
            st.session_state.session_creator = SessionCreator(fetcher=st.session_state.fetcher)
        else:
            st.session_state.fetcher = InfluxDbFetcher(config=ConfigLogging)
            st.session_state.session_creator = SessionCreator(fetcher=st.session_state.fetcher)

        # Choose FSM value to fetch
        st.divider()
        fsm_values = FSM.all_states
        fsm_value = st.selectbox("FSM value", fsm_values, index=fsm_values.index(FSM.r2d))

        # Fetch R2D sessions
        fetch = st.button(f"Fetch '{fsm_value}' sessions")
        session_creator: SessionCreator = st.session_state.session_creator

        if fetch:
            st.session_state.fsm_states = pd.DataFrame()
            dfs = session_creator.fetch_r2d_session(start_date, end_date, verify_ssl=st.session_state.verify_ssl,
                                                    fsm_value=fsm_value)
            st.session_state.sessions = dfs
            if len(dfs) == 0:
                st.error(
                    "No R2D session found in the selected date range (if the requested data is recent, it might not have been uploaded yet)")
            else:
                st.success(f"Fetched {len(dfs)} sessions, select one in the dropdown menu")

        st.divider()
        fetch_fsm = st.button("Fetch FSM states")
        if fetch_fsm:
            st.session_state.sessions = []
            dfs = session_creator.fetch_fsm(start_date, end_date, verify_ssl=st.session_state.verify_ssl)
            st.session_state.fsm_states = dfs
            if len(dfs) == 0:
                st.error(
                    "No FSM states found in the selected date range (if the requested data is recent, it might not "
                    "have been uploaded yet)")
            else:
                st.success(f"Fetched {len(dfs)} states, select one or multiple in the data editor")

    # Build the tabs
    if len(st.session_state.sessions) > 0:
        tabs: List[Tab] = create_tabs()
        st_tabs = st.tabs(tabs=[tab.description for tab in tabs])

        for tab, st_tab in zip(tabs, st_tabs):
            with st_tab:
                tab.build(session_creator=session_creator)

    if len(st.session_state.fsm_states) > 0:
        fsm_state_tab = FSMStateTab()
        fsm_state_tab.build(session_creator=session_creator)
