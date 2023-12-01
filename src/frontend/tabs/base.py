from abc import ABC, abstractmethod

import streamlit as st

from src.backend.sessions.create_sessions import SessionCreator


class Tab(ABC):
    name: str
    description: str
    memory: dict

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        if self.name not in st.session_state:
            st.session_state[self.name] = {}
        self.memory = st.session_state[self.name]


    @abstractmethod
    def build(self, session_creator: SessionCreator) -> bool:
        pass
