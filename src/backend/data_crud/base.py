from abc import ABC, abstractmethod


class CRUD(ABC):
    file_path_name: str
    data: dict

    @abstractmethod
    def create(self, *args, **kwargs):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def read(self, *args, **kwargs):
        pass

    @abstractmethod
    def delete(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_raw_data(self) -> dict:
        pass