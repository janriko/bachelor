from typing import List
from data_models.raw_state.JerichoStateTransition import JerichoStateTransition


class JerichoTransitionList:
    def __init__(self, transitionList):
        self.transitionList: List[JerichoStateTransition] = transitionList

    def __len__(self):
        return len(self.transitionList)

    def __getitem__(self, index):
        return self.transitionList[index]

    def append(self, nee_item: JerichoStateTransition):
        return self.transitionList.append(nee_item)
