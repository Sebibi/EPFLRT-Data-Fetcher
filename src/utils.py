from typing import List

import numpy as np
import streamlit as st
import pandas as pd
from influxdb_client import InfluxDBClient

from config.config import Config
from src.api_call.influxdb_api import InfluxDbFetcher


def date_to_influx(date: pd.Timestamp) -> str:
    return date.strftime("%Y-%m-%dT%H:%M:%SZ")


def timestamp_to_datetime_range(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"range(start: {date_to_influx(start)}, stop: {date_to_influx(end)})"
