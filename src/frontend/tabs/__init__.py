from typing import List

from src.frontend.tabs.base import Tab
from src.frontend.tabs.tab1 import Tab1
from src.frontend.tabs.tab2 import Tab2
from src.frontend.tabs.tab3 import Tab3
from src.frontend.tabs.tab4 import Tab4
from src.frontend.tabs.tab5 import Tab5
from src.frontend.tabs.tab6 import Tab6
from src.frontend.tabs.tab7 import Tab7
from src.frontend.tabs.tab8 import Tab8
from src.frontend.tabs.tab9 import Tab9
from src.frontend.tabs.tab10 import Tab10
from src.frontend.tabs.tab11 import Tab11
from src.frontend.tabs.tab12 import Tab12
from src.frontend.tabs.tab13 import Tab13

from src.frontend.tabs.fsm_state_tab import FSMStateTab
from src.frontend.tabs.telemetry_description_tab import TelemetryDescriptionTab


def create_tabs() -> List[Tab]:
    tabs = [Tab1(), Tab2(), Tab3(), Tab4(), Tab5(), Tab6(), Tab7(), Tab8(), Tab9(), Tab10(), Tab11(), Tab12(), Tab13()]
    tabs = list(reversed(tabs))
    tab_names = [tab.name for tab in tabs]
    assert len(tab_names) == len(set(tab_names)), "There must not be any duplicate tab names"
    return tabs
