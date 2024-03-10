import pandas as pd
from influxdb_client import InfluxDBClient

from config.config import Config
from src.backend.api_call.base import Fetcher


class InfluxDbFetcher(Fetcher):

    def __init__(self, config: Config):
        self.token = config.token
        self.org = config.org
        self.url = config.url
        self.bucket_name = config.bucket_name

    def fetch_data(self, query: str, verify_sll: bool) -> pd.DataFrame:
        with InfluxDBClient(url=self.url, token=self.token, org=self.org, verify_ssl=verify_sll) as client:
            df = client.query_api().query_data_frame(query=query, org=self.org)
            if len(df) == 0:
                return pd.DataFrame()
            else:
                df.drop(columns=["result", "table"], inplace=True)
                df["_time"] = pd.to_datetime(df["_time"])
                df.set_index("_time", inplace=True)
            return df
