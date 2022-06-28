from typing import List
from torch.utils.data import Dataset
from data_models.raw_state.JerichoTransitionList import JerichoTransitionList


class JerichoDataset(Dataset):
    def __init__(self, worldList):
        self.worldList: List[JerichoTransitionList] = worldList

    def __len__(self):
        return len(self.worldList)

    def __getitem__(self, index):
        return self.worldList[index]

    def append(self, nee_item: JerichoTransitionList):
        return self.worldList.append(nee_item)
