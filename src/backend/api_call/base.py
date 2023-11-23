from abc import ABC, abstractmethod

import pandas as pd


class Fetcher(ABC):

    @abstractmethod
    def fetch_data(self, query: str, verify_sll: bool) -> pd.DataFrame:
        pass
