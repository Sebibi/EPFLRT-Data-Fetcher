from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from influxdb_client import InfluxDBClient
import streamlit as st
import matplotlib.pyplot as plt
from typing import List
from sklearn.preprocessing import MinMaxScaler

from utils import fetch_r2d_session, choose_r2d_session, fetch_data


def init_sessions_state():
    if "sessions" not in st.session_state:
        st.session_state.sessions = []

    if "data" not in st.session_state:
        st.session_state.data = pd.DataFrame()


if __name__ == '__main__':

    # Initialize the application
    st.set_page_config(layout="wide")
    init_sessions_state()
    st.title("Data Fetcher")
    verify_sll = st.checkbox("Fetch with SSL", value=False)

    # API information
    st.subheader("Fetch data from InfluxDB")
    token = "cCtyspyt-jeehwf5Ayz5OmaOXnvkMj46z3C6UQlud4s8MiPZLFaFuM7z1Y_qqpmVyI5cvF4h9k-kl5dCiYmWFw=="
    org = "racingteam"
    st.markdown(f"**Organisation**: {org}")
    st.markdown(f"**Api token**: {token}")

    # Choose date range
    date_cols = st.columns(2)
    start_date = date_cols[0].date_input("Start date", value=pd.to_datetime("2023-10-05"),
                                         max_value=pd.to_datetime(datetime.now().strftime("%Y-%m-%d")))
    end_date = date_cols[1].date_input("End date", value=pd.to_datetime("2023-10-06"),
                                       max_value=pd.to_datetime(
                                           (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")))

    # Fetch R2D sessions
    fetch = st.button("Fetch R2D sessions")
    if fetch:
        dfs = fetch_r2d_session(start_date, end_date, verify_ssl=verify_sll)
        st.session_state.sessions = dfs
        if len(dfs) == 0:
            st.error(
                "No R2D session found in the selected date range (if the requested data is recent, it might not have been uploaded yet)")
        else:
            st.success(f"Fetched {len(dfs)} sessions, select one in the dropdown below")

    # Choose R2D session and fetch data
    if len(st.session_state.sessions) > 0:
        st.header("Choose a session !")
        datetime_range = choose_r2d_session(st.session_state.sessions)
        if st.button("Fetch data"):
            data = fetch_data(datetime_range, verify_ssl=verify_sll)
            data.index = (data.index - data.index[0]).total_seconds()
            st.session_state.data = data

    if len(st.session_state.data) > 0:
        # measurements = ["MISC", "AMS", "VSI", "sensors"]
        data = st.session_state.data
        # Select the Data
        selected_columns = st.multiselect(label="Select the fields you want to download", options=data.columns,
                                          default=list(data.columns[:2]))
        samples_to_select = st.select_slider(label="Number of samples to select", options=data.index,
                                             value=[data.index[0], data.index[-1]], format_func=lambda x: f"{x:.2f}")
        output_data = data[selected_columns].loc[samples_to_select[0]:samples_to_select[1]]

        cols = st.columns(2)
        with cols[0]:
            st.subheader("Data to Download")
            st.dataframe(output_data)
        with cols[1]:
            st.subheader("Data Description")
            st.dataframe(output_data.describe())

        # Download data
        file_name = st.text_input("File name", value="output_data.csv")
        st.download_button(
            label="Download data as CSV",
            data=output_data.to_csv().encode("utf-8"),
            file_name=file_name,
        )

        # Plot data
        st.subheader("Plot some data")

        columns_to_plot = st.multiselect(label="Select the labels to plot", options=data.columns,
                                         default=list(data.columns[:2]))
        samples_to_plot = st.select_slider(label="Number of samples to plot", options=data.index,
                                           value=[data.index[0], data.index[-1]], format_func=lambda x: f"{x:.2f}")
        plot_data = data[columns_to_plot].loc[samples_to_plot[0]:samples_to_plot[1]]
        if st.checkbox("Smooth data"):
            smooth_cols = st.multiselect(label="Select the labels to smooth", options=plot_data.columns,
                                         default=list(plot_data.columns))
            plot_data[smooth_cols] = plot_data[smooth_cols].rolling(window=7).mean()

        if st.checkbox("Normalize data"):
            scaler = MinMaxScaler()
            data['sensors_vX'] = abs(data['sensors_vX'])
            scale_cols = st.multiselect(label="Select the labels to scale", options=plot_data.columns,
                                        default=list(plot_data.columns))
            plot_data[scale_cols] = pd.DataFrame(scaler.fit_transform(plot_data[scale_cols]), columns=scale_cols,
                                                 index=plot_data.index)
        fig, ax = plt.subplots(figsize=(16, 9))
        # ax.axhline(y=25, color='r', linestyle='-', label='Horizontal Line at y=25')
        plot_data.plot(ax=ax)
        st.pyplot(fig, use_container_width=False)
