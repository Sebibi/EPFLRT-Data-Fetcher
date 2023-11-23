import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from src.functionnal.create_sessions import SessionCreator
from src.tabs.base import Tab


class Tab5(Tab):

    def __init__(self):
        super().__init__("tab5", "State Estimation analysis")
        if "data" not in self.memory:
            self.memory['data'] = pd.DataFrame()

        self.state_estimation_df_cols: dict[str, list[str]] = dict(
            No_slips=['_time', 'sensors_vXEst', 'sensors_vYEst', 'sensors_aXEst', 'sensors_aYEst'],
        )
        self.state_estimation_df_cols['Full'] = self.state_estimation_df_cols['No_slips'] + [
            'sensors_dpsi_est', 'sensors_FL_est', 'sensors_FR_est', 'sensors_RL_est', 'sensors_RR_est']

    def build(self, session_creator: SessionCreator) -> bool:

        st.header(self.description)
        datetime_range = session_creator.r2d_session_selector(st.session_state.sessions,
                                                              key=f"{self.name} session selector")
        if st.button("Fetch this session", key=f"{self.name} fetch data button"):
            data = session_creator.fetch_data(datetime_range, verify_ssl=st.session_state.verify_ssl)
            data.index = (data.index - data.index[0]).total_seconds()
            self.memory['data'] = data.copy()

        if len(self.memory['data']) > 0:
            data = self.memory['data']

            # Import the new estimated data
            cols = st.columns([1, 1])
            estimation_type = cols[1].radio("Choose the data to import", options=['Full', 'No_slips'], index=1)
            uploaded_file = cols[0].file_uploader(
                "Choose a file", type="csv", key=f"{self.name} file uploader",
                label_visibility='collapsed',
            )

            # Load the state estimation data and Replace the new data with the old one
            if uploaded_file is not None:
                if cols[1].button("Update the state estimation data", key=f"{self.name} save estimation data button"):
                    state_estimation_df = pd.read_csv(uploaded_file)
                    state_estimation_df.columns = self.state_estimation_df_cols[estimation_type]
                    state_estimation_df.set_index('_time', inplace=True)
                    data.loc[state_estimation_df.index, state_estimation_df.columns] = state_estimation_df.values
                    self.memory['data'] = data

            # Send data to Other Tabs
            with st.expander("Send data to another TAB"):
                other_tabs = ['tab1', 'tab2', 'tab3', 'tab4']
                for i, other_tab in enumerate(other_tabs):
                    cols = st.columns([1, 3])
                    if cols[0].button(f"Send data to {other_tab}", key=f"{self.name} send data to {other_tab} button"):
                        st.session_state[other_tab]['data'] = self.memory['data'].copy()
                        cols[1].success(f"Data sent to {other_tab}")

            # Plot data
            st.subheader("Plot some data")
            columns_to_plot = st.multiselect(
                label="Select the labels to plot",
                options=data.columns,
                default=self.state_estimation_df_cols['No_slips'][1:],
                key=f"{self.name} columns to plot",
            )
            samples_to_plot = st.select_slider(
                label="Number of samples to plot", options=data.index,
                value=[data.index[0], data.index[-1]], format_func=lambda x: f"{x:.2f}",
                key=f"{self.name} samples to plot",
            )
            plot_data = data[columns_to_plot].loc[samples_to_plot[0]:samples_to_plot[1]]

            st.subheader("Plot state estimation data")
            fig, ax = plt.subplots()
            plot_data.plot(ax=ax)
            ax.legend()
            ax.set_title('State Estimation observation')
            st.pyplot(fig)
        return True
