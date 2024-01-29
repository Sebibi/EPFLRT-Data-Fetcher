import pandas as pd
import streamlit as st


def upload_estimated_states(tab_name: str, data: pd.DataFrame, columns: list[str], cols: st.columns = None, show: bool = False) -> pd.DataFrame:
    if cols is None:
        cols = st.columns(2)
    uploaded_file = cols[0].file_uploader(
        "Choose a file", type="csv", key=f"{tab_name} file uploader",
        label_visibility='collapsed',
    )
    # Load the state estimation data and Replace the new data with the old one
    if uploaded_file is not None:
        if cols[1].button("Update the state estimation data", key=f"{tab_name} save estimation data button"):
            state_estimation_df = pd.read_csv(uploaded_file, header=None)
            state_estimation_df = state_estimation_df.iloc[:, :len(columns)]
            state_estimation_df.columns = columns
            state_estimation_df.set_index('_time', inplace=True)
            if show:
                cols[0].dataframe(data[state_estimation_df.columns], use_container_width=True)
                cols[0].info(data[state_estimation_df.columns].shape)
                cols[1].dataframe(state_estimation_df, use_container_width=True)
                cols[1].info(state_estimation_df.shape)
            data.loc[state_estimation_df.index[0]:state_estimation_df.index[-1], state_estimation_df.columns] = state_estimation_df.values
            data.drop_duplicates(inplace=True)
    return data
