from typing import List

from src.tabs.base import Tab
from src.tabs.tab1 import Tab1
from src.tabs.tab2 import Tab2
from src.tabs.tab3 import Tab3


def create_tabs() -> List[Tab]:
    tabs = [Tab1(), Tab2(), Tab3()]
    return tabs
