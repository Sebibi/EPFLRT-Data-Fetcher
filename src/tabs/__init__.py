from typing import List

from src.tabs.base import Tab
from src.tabs.tab1 import Tab1
from src.tabs.tab2 import Tab2
from src.tabs.tab3 import Tab3
from src.tabs.tab4 import Tab4


def create_tabs() -> List[Tab]:
    tabs = [Tab1(), Tab2(), Tab3(), Tab4()]
    tab_names = [tab.name for tab in tabs]
    assert len(tab_names) == len(set(tab_names)), "There must not be any duplicate tab names"
    return tabs
