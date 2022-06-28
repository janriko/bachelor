from typing import List

from torch.utils.data import Dataset

# from data_models.tokenized.TokenizedState import TokenizedState


class TokenizedDataset(Dataset):
    def __init__(self, tokenized_states):
#        super(TokenizedDataset1).__init__(tokenized_states)
        self.tokenized_states: List[List[List[int]]] = tokenized_states

    def __len__(self):
        return len(self.tokenized_states)

    def __getitem__(self, index):
        return self.tokenized_states[index]

    def append(self, tokenized_state: List[List[int]]):
        self.tokenized_states.append(tokenized_state)
