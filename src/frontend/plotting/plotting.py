from typing import Tuple, List

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def plot_data(
        data: pd.DataFrame, tab_name: str, title: str = "Sensors",
        default_columns: list = None, fig_ax: Tuple[plt.Figure, plt.Axes] = None) -> Tuple[List[str], List[str]]:

    columns_to_plot = st.multiselect(
        label="Select the labels to plot",
        options=data.columns,
        default=["sensors_aXEst", "sensors_vXEst"] if default_columns is None else default_columns,
        key=f"{tab_name} columns to plot",
    )
    samples_to_plot = st.select_slider(
        label="Number of samples to plot", options=data.index,
        value=[data.index[0], data.index[-1]], format_func=lambda x: f"{x:.2f}",
        key=f"{tab_name} samples to plot",
    )
    plot_data = data[columns_to_plot].loc[samples_to_plot[0]:samples_to_plot[1]]

    cols = st.columns(2)
    fig, ax = plt.subplots(figsize=(10, 5)) if fig_ax is None else fig_ax
    window_size = cols[1].number_input(f"Moving average to be applied to data", value=1, step=1, min_value=1, key=f"{tab_name} window size")
    rolled_plot_data = plot_data.rolling(window=window_size).mean()
    rolled_plot_data.plot(ax=ax, subplots=cols[0].checkbox("Subplots", value=False, key=f"{tab_name} subplots"))
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('Time [s]')
    st.pyplot(fig)
    return columns_to_plot, samples_to_plot
