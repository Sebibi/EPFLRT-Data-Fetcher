from abc import ABC
from typing import TypedDict


class Config(ABC):
    name: str
    bucket_name: str
    token: str
    org: str
    url: str


class ConfigLogging(Config):
    name = "logging"
    bucket_name = "ariane"
    token = "cCtyspyt-jeehwf5Ayz5OmaOXnvkMj46z3C6UQlud4s8MiPZLFaFuM7z1Y_qqpmVyI5cvF4h9k-kl5dCiYmWFw=="
    org = "racingteam"
    url = "https://epfl-rt-data-logging.epfl.ch:8443"


class ConfigLive(Config):
    name = "live"
    bucket_name = "Ariane"
    token = "uC8xUsjnPK3D2Gd-RJbYLkWYHID3k527OcBHvZQe16SfRTu-LzKSGVaZ9ckLwXDsk9UqlfNCXJhFht9FGLcXiw=="
    org = "racingteam"
    url = "http://192.168.1.10:8086"


class FSM:
    error = "Error"
    init = "Init"
    ts_off = "TS off"
    pre_charge = "PreCharge"
    ts_on = "TS On"
    amk_start = "AMK Start"
    r2d = "R2D"
    all_states = [error, init, ts_off, pre_charge, ts_on, amk_start, r2d]
