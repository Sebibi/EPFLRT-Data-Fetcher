from influxdb_client import InfluxDBClient
import pandas as pd

from src.backend.api_call.base import Fetcher


class InfluxDbFetcher(Fetcher):

    def __init__(self, token, org, url):
        self.token = token
        self.org = org
        self.url = url

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
