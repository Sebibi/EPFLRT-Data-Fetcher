from typing import TypedDict
import json
from src.backend.data_crud.base import CRUD
from copy import deepcopy


class SessionInfo(TypedDict):
    driver: str
    weather_condition: str
    control_mode: str
    description: str


class SessionInfoJsonCRUD(CRUD):
    file_path_name: str
    data: dict[str, dict[str]]

    def __init__(self, file_path_name: str):
        assert file_path_name.endswith('.json'), "File must be of type 'json'"
        self.file_path_name = file_path_name
        with open(file_path_name, 'r') as f:
            if f.read() == "":
                self.data = {}
            else:
                f.seek(0)
                self.data = json.load(f)


    def _get_data(self, key: str) -> SessionInfo | None:
        return self.data.get(key, None)

    def _set_data(self, key: str, data: SessionInfo):
        self.data[key] = data

    def create(self, time_str: str, driver: str, weather_condition: str, control_mode: str, description: str) -> bool:
        new_data = dict(driver=driver, weather_condition=weather_condition, control_mode=control_mode, description=description)
        self._set_data(time_str, new_data)
        with open(self.file_path_name, 'w') as f:
            f.write(json.dumps(self.data, sort_keys=True, indent=4, ensure_ascii=False))
        return True

    def update(self, time_str: str, driver: str, weather_condition: str, control_mode: str, description: str) -> bool:
        return self.create(time_str, driver, weather_condition, control_mode, description)

    def read(self, time_str: str) -> SessionInfo:
        res = self._get_data(time_str)
        return res if res else dict(driver=None, weather_condition=None, control_mode=None, description=None)

    def delete(self, time_str: str) -> bool:
        if time_str in self.data:
            del self.data[time_str]
            with open(self.file_path_name, 'w') as f:
                f.write(json.dumps(self.data, sort_keys=True, indent=4, ensure_ascii=False))
            return True
        return False

    def get_raw_data(self) -> dict[str, SessionInfo]:
        return deepcopy(self.data)
