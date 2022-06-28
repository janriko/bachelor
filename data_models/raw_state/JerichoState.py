from typing import Dict, List
from data_models.raw_state.JerichoLocation import JerichoLocation


class JerichoState:
    def __init__(self, walkthrough_act, walkthrough_diff, obs, loc_desc, inv_desc, inv_objs, inv_attrs, location,
                 surrounding_objs, surrounding_attrs, graph, valid_acts, score):
        self.walkthrough_act: str = walkthrough_act
        self.walkthrough_diff: str = walkthrough_diff
        self.obs: str = obs
        self.loc_desc: str = loc_desc
        self.inv_desc: str = inv_desc
        self.inv_objs: Dict[str, List[str]] = inv_objs
        self.inv_attrs: Dict[str, List[str]] = inv_attrs
        self.location: JerichoLocation = JerichoLocation(**location)
        self.surrounding_objs: Dict[str, List[str]] = surrounding_objs
        self.surrounding_attrs: Dict[str, List[str]] = surrounding_attrs
        self.graph: List[List[str]] = graph
        self.valid_acts: Dict[str, str] = valid_acts
        self.score: int = score
