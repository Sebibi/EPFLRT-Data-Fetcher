import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from src.functionnal.create_sessions import SessionCreator
from src.tabs.base import Tab


class Tab1(Tab):

    def __init__(self):
        super().__init__("tab1", "Session analysis")
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
            # Select the Data
            selected_columns = st.multiselect(
                label="Select the fields you want to download", options=data.columns, default=list(data.columns[:2]))
            samples_to_select = st.select_slider(
                label="Number of samples to select", options=data.index,
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
            columns_to_plot = st.multiselect(
                label="Select the labels to plot",
                options=data.columns,
                default=list(data.columns[:2])
            )
            samples_to_plot = st.select_slider(
                label="Number of samples to plot", options=data.index,
                value=[data.index[0], data.index[-1]], format_func=lambda x: f"{x:.2f}"
            )
            plot_data = data[columns_to_plot].loc[samples_to_plot[0]:samples_to_plot[1]]

            # Allow for data smoothing
            if st.checkbox("Smooth data"):
                smooth_cols = st.multiselect(label="Select the labels to smooth", options=plot_data.columns,
                                             default=list(plot_data.columns))
                plot_data[smooth_cols] = plot_data[smooth_cols].rolling(window=7).mean()

            # Allow for data normalization
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
            return True
