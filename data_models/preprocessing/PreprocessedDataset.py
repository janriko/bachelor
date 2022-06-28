from typing import List
from torch.utils.data import Dataset
from data_models.preprocessing.PreprocessedState import PreprocessedState


class PreprocessedDataset(Dataset):
    def __init__(self, preprocessed_state):
        self.preprocessed_states: List[PreprocessedState] = preprocessed_state

    def __len__(self):
        return len(self.preprocessed_states)

    def __getitem__(self, index):
        return self.preprocessed_states[index]

    def append(self, nee_item: PreprocessedState):
        return self.preprocessed_states.append(nee_item)
