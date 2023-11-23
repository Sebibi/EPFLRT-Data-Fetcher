import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from typing import Tuple, List

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

    fig, ax = plt.subplots(figsize=(10, 5)) if fig_ax is None else fig_ax
    plot_data.plot(ax=ax)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('Time [s]')
    st.pyplot(fig)
    return columns_to_plot, samples_to_plot
