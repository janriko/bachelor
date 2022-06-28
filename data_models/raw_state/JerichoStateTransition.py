from typing import List
from data_models.raw_state.JerichoState import JerichoState


class JerichoStateTransition:
    def __init__(self, rom, state, next_state, graph_diff, action, reward):
        self.rom: str = rom
        self.state: JerichoState = JerichoState(**state)
        self.next_state: JerichoState = JerichoState(**next_state)
        self.graph_diff: List[List[str]] = graph_diff
        self.action: str = action
        self.reward: int = reward
