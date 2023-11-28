from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


def plot_correlation_log(data: pd.DataFrame, col1: str, col2: str, lags: Sequence[int], samples_to_plot) -> int:
    if st.checkbox("Differentiate the signals"):
        signals = [data[col1].diff(), data[col2].shift(1).diff()]
    else:
        signals = [data[col1], data[col2]]

    signals[0] = signals[0].loc[samples_to_plot[0]:samples_to_plot[1]]
    signals[1] = signals[1].loc[samples_to_plot[0]:samples_to_plot[1]]

    fig, ax = plt.subplots(figsize=(10, 5))
    correlations = np.array([signals[0].corr(signals[1].shift(lag)) for lag in lags])

    if st.checkbox("High frequency pass filter"):
        correlations = correlations[:-1] - 0.9 * correlations[1:]
        lags = lags[1:]

    max_corr = np.max(correlations)
    max_lag = lags[np.argmax(correlations)]

    # plot the max correlation as cross marker
    ax.vlines(x=max_lag, ymin=np.min(correlations), ymax=max_corr, color='r', linestyle='--')
    ax.plot(lags, correlations)
    ax.set_xlabel('Lag [10 ms]')
    ax.set_ylabel('Correlation')
    ax.set_title(f"Correlation between {col1} and {col2}")
    st.pyplot(fig)
    return max_lag