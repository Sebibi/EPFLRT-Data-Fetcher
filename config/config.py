from typing import TypedDict


class Config(TypedDict):
    name: str
    bucket_name: str
    token: str
    org: str
    url: str

class ConfigLogging:
    name = "logging"
    bucket_name = "ariane"
    token = "cCtyspyt-jeehwf5Ayz5OmaOXnvkMj46z3C6UQlud4s8MiPZLFaFuM7z1Y_qqpmVyI5cvF4h9k-kl5dCiYmWFw=="
    org = "racingteam"
    url = "https://epfl-rt-data-logging.epfl.ch:8443"


class ConfigLive:
    name = "live"
    bucket_name = "Ariane"
    token = "O8lM3ToG3BmtgYqK8TZF_Bu6XxYvImWX2R_nBOAXvE-u0Gpgs2fkViKviIG5hRZmjgHqX4IJhNNb399bbEE5jg=="
    org = "racingteam"
    url = "http://192.168.1.10:8086"



