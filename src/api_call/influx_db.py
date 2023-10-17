from datetime import datetime

import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS


class DataFetcher:
    token = "cCtyspyt-jeehwf5Ayz5OmaOXnvkMj46z3C6UQlud4s8MiPZLFaFuM7z1Y_qqpmVyI5cvF4h9k-kl5dCiYmWFw=="
    org = "racingteam"

    query: str = """from(bucket: "ariane")
|> range(start: 2023-09-05T00:00:00Z, stop: 2023-10-03T00:00:00Z)
|> filter(fn: (r) => r["_measurement"] == "sensors" or r["_measurement"] == "MISC" and r["_field"] == "FSM")
|> pivot(rowKey:["_time"], columnKey: ["_field", "_measurement"], valueColumn: "_value")
|> yield(name: "mean")"""

    def fetch_data(self) -> pd.DataFrame:
        with InfluxDBClient(url="https://epfl-rt-data-logging.epfl.ch:8443", token=self.token, org=self.org) as client:
            tables = client.query_api().query_data_frame(query=self.query, org=self.org)
        return tables

    def parse_data(self, data: pd.DataFrame) -> pd.DataFrame:
        fields = data["_field"].unique()
        data = data.pivot_table(index=["_time"], columns=["_field"], values=["_value"])
        data.columns = data.columns.droplevel(0)
        print(data)



if __name__ == '__main__':
    fetcher = DataFetcher()
    tables = fetcher.fetch_data()
