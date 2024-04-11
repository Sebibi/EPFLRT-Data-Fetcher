from typing import TypedDict
import json
from src.backend.data_crud.base import CRUD
from copy import deepcopy


class TelemetryDescription(TypedDict):
    unit: str
    description: str


class JsonCRUD(CRUD):
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

    @classmethod
    def _get_keys(cls, field_name: str) -> tuple[str, str]:
        splits = field_name.split('_')
        return splits[0], '_'.join(splits[1:])

    def _get_data(self, keys: tuple[str, str]) -> TelemetryDescription | None:
        assert len(keys) == 2
        res = self.data.get(keys[0], None)
        if res:
            return res.get(keys[1], None)
        else:
            return None

    def _set_data(self, keys: tuple[str, str], unit: str, description: str):
        new_data = dict(unit=unit, description=description)
        if keys[0] not in self.data:
            self.data[keys[0]] = {}
        self.data[keys[0]][keys[1]] = new_data

    def create(self, field_name: str, unit: str, description: str) -> bool:
        self._set_data(self._get_keys(field_name), unit, description)
        with open(self.file_path_name, 'w') as f:
            f.write(json.dumps(self.data, sort_keys=True, indent=4, ensure_ascii=False))
        return True

    def update(self, field_name: str, unit: str, description: str) -> bool:
        return self.create(field_name, unit, description)

    def read(self, bucket_name: str, field_name: str) -> TelemetryDescription:
        data_read = self._get_data(self._get_keys(field_name))
        if data_read:
            return data_read
        else:
            return dict(unit=None, description=None)

    def delete(self, field_name: str) -> bool:
        keys = self._get_keys(field_name)
        if keys[0] in self.data:
            if keys[1] in self.data[keys[0]]:
                del self.data[keys[0]][keys[1]]
                with open(self.file_path_name, 'w') as f:
                    f.write(json.dumps(self.data, sort_keys=True, indent=4, ensure_ascii=False))
                return True
        return False

    def get_raw_data(self) -> dict[str, dict[str]]:
        return deepcopy(self.data)
