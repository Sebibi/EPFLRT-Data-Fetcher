from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from config.config import Config
from src.api_call.influxdb_api import InfluxDbFetcher
from src.functionnal.create_sessions import SessionCreator

from src.tabs import build_tab1, build_tab2


def init_sessions_state():
    if "sessions" not in st.session_state:
        st.session_state.sessions = []

    if "fetcher" not in st.session_state:
        st.session_state.fetcher = InfluxDbFetcher(token=Config.token, org=Config.org, url=Config.url)

    if "session_creator" not in st.session_state:
        st.session_state.session_creator = SessionCreator(fetcher=st.session_state.fetcher)

    if "verify_ssl" not in st.session_state:
        st.session_state.verify_ssl = False


if __name__ == '__main__':

    # Initialize the application
    st.set_page_config(layout="wide")
    init_sessions_state()

    cols = st.columns([5, 1])
    cols[0].title("InfluxDB data Fetcher")
    cols[0].markdown("This application allows you to fetch data from the InfluxDB database of the EPFL Racing Team")
    cols[1].image("data/img/epflrt_logo.png", width=200)
    st.session_state.verify_ssl = st.checkbox("Fetch with SSL", value=False)

    # API information and Fetcher initialization
    st.markdown(f"**Organisation**: {Config.org}")
    st.markdown(f"**Api token**: {Config.token}")

    # Choose date range
    st.subheader("Select a date range")
    date_cols = st.columns(2)
    start_date = date_cols[0].date_input("Start date", value=pd.to_datetime("2023-10-05"),
                                         max_value=pd.to_datetime(datetime.now().strftime("%Y-%m-%d")))
    end_date = date_cols[1].date_input("End date", value=pd.to_datetime("2023-10-06"),
                                       max_value=pd.to_datetime(
                                           (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")))

    session_creator: SessionCreator = st.session_state.session_creator

    # Fetch R2D sessions
    fetch = st.button("Fetch R2D sessions")
    if fetch:
        dfs = session_creator.fetch_r2d_session(start_date, end_date, verify_ssl=st.session_state.verify_ssl)
        st.session_state.sessions = dfs
        if len(dfs) == 0:
            st.error(
                "No R2D session found in the selected date range (if the requested data is recent, it might not have been uploaded yet)")
        else:
            st.success(f"Fetched {len(dfs)} sessions, select one in the dropdown below")


        # Build the tabs
    if len(st.session_state.sessions) > 0:
        st.subheader("Select a tab")
        tabs = st.tabs(["Session Analysis", "Maximum features extraction"])
        with tabs[0]:
            build_tab1(session_creator)

        with tabs[1]:
            build_tab2(session_creator)
