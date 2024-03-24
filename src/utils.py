import pandas as pd


def date_to_influx(date: pd.Timestamp) -> str:
    return date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def timestamp_to_datetime_range(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"range(start: {date_to_influx(start)}, stop: {date_to_influx(end)})"
