from typing import TypedDict
import json
from src.backend.data_crud.base import CRUD


class TelemetryDescription(TypedDict):
    unit: str
    description: str


class JsonCRUD(CRUD):
    file_path_name: str
    data: dict[str, dict[str]]

    def __init__(self, file_path_name: str):
        assert file_path_name[:5] == '.json', "File must be of type 'json'"
        self.file_path_name = file_path_name
        with open(file_path_name, 'r') as f:
            self.data = json.load(f)

    def _get_data(self, bucket_name: str, field_name: str) -> TelemetryDescription | None:
        bucket_res = self.data.get(bucket_name, None)
        if bucket_res:
            return bucket_res.get(field_name, None)
        else:
            return None

    def _set_data(self, bucket_name: str, field_name: str, unit: str, description: str):
        new_data: TelemetryDescription = dict(unit=unit, description=description)
        if bucket_name not in self.data:
            self.data[bucket_name] = {}
        self.data[bucket_name][field_name] = new_data

    def create(self, bucket_name: str, field_name: str, unit: str, description: str) -> bool:
        self._set_data(bucket_name, field_name, unit, description)
        with open(self.file_path_name, 'w') as f:
            json.dumps(f, sort_keys=True, indent=4)
        return True

    def update(self, bucket_name: str, field_name: str, unit: str, description: str) -> bool:
        return self.create(bucket_name, field_name, unit, description)

    def read(self, bucket_name: str, field_name: str) -> TelemetryDescription:
        data_read = self._get_data(bucket_name, field_name)
        if data_read:
            return data_read
        else:
            return dict(unit="None", description="None")

    def delete(self, bucket_name: str, field_name: str = None) -> bool:
        assert bucket_name in self.data
        del self.data[bucket_name][field_name]
        if field_name == None:
            del self.data[bucket_name]
        return True
