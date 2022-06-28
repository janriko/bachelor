from typing import List
import ast

import numpy as np
import torch
from datasets.arrow_dataset import Dataset


class GraphTokenizer:
    tokens = ast.literal_eval(open("create_vocab/graph_encoder_vocabulary_training.txt", "r").read())

    def __init__(self):
        pass

    def convert_triple_to_ids(self, triple: str) -> List[int]:
        special_token, rest = triple.split("]")
        first_token, second_token, third_token = rest.split("§")

        special_token_id = self.tokens[special_token + "]"]
        first_token_id = self.tokens[first_token]
        second_token_id = self.tokens[second_token]
        third_token_id = self.tokens[third_token]
        return [special_token_id, first_token_id, second_token_id, third_token_id]

    def encode(self, text: Dataset, column_name: str, shouldMask: bool) -> Dataset:
        text_list = text[column_name]
        input_ids: List[List[id]] = []
        attention_mask: List[List[id]] = []
        token_type_ids: List[List[id]] = []

        for state_index in range(len(text_list)):
            input_ids.append([])
            attention_mask.append([])
            token_type_ids.append([])
            attention_mask_counter = 0
            current_text = text_list[state_index]
            while len(current_text) > 0:
                attention_mask_counter += 4

                index_of_next_triple = current_text.find("[", 1)
                # if state doesn't contain any more triples, take last triple
                if index_of_next_triple == -1:
                    index_of_next_triple = len(current_text)
                # convert text to id's
                four_ids = self.convert_triple_to_ids(triple=current_text[:index_of_next_triple])
                # add tokenized triple to list
                input_ids[state_index].extend(four_ids)
                # remove just tokenized triple
                current_text = current_text[index_of_next_triple:]
            if len(input_ids[state_index]) < 1024:
                input_ids[state_index].extend([0] * (1024 - len(input_ids[state_index])))
            elif len(input_ids[state_index]) > 1024:
                input_ids[state_index] = input_ids[state_index][:1024]

            state_attention_mask = ([1]*attention_mask_counter)
            state_attention_mask.extend([0]*(1024-attention_mask_counter))
            attention_mask[state_index] = state_attention_mask

        new_dataset = text.add_column("input_ids", input_ids)
        new_dataset = new_dataset.add_column("attention_mask", attention_mask)
        new_dataset = new_dataset.add_column("token_type_ids", [[0]*1024] * len(input_ids))
        new_dataset = new_dataset.remove_columns(column_name)
        if shouldMask:
            new_dataset = self.mask_dataset(new_dataset, 0.2)
        new_dataset.set_format("torch")
        return new_dataset

    def decode(self, words: List[int]):
        pass

    def mask_dataset(self, dataset: Dataset, probability: float) -> Dataset:
        probability = probability / 3
        dataset = dataset.remove_columns("labels")
        dataset = dataset.add_column("labels", dataset["input_ids"])

        input_ids = np.array(dataset["input_ids"])
        attention_mask = np.array(dataset["attention_mask"])

        rand = torch.rand(input_ids.shape)
        mask_arr = (rand < probability) * (input_ids != self.tokens["[GRAPH]"]) * (input_ids != self.tokens["[TRIPLE]"]) * attention_mask
        for index in range(len(mask_arr)):
            input_ids = self.mask_one_list(input_ids, mask_arr, index)
        dataset = dataset.remove_columns("input_ids")
        dataset = dataset.add_column("input_ids", input_ids.tolist())
        return dataset

    def mask_one_list(self, input_ids, mask_arr, index):
        selection: list = torch.flatten((mask_arr[index]).nonzero()).tolist()
        new_selection: set = set()
        for m in selection:
            new_selection.add(m)
            # check if previous token is special token -> use next 2 tokens
            if self.check_if_special_token(input_ids[index, m - 1]):
                new_selection.add(m + 1)
                new_selection.add(m + 2)
            # check if next token is special token -> use 2 tokens before
            elif self.check_if_special_token(input_ids[index, m + 1]):
                new_selection.add(m - 1)
                new_selection.add(m - 2)
            # else -> currently the middle token -> use 1 before and one after
            else:
                new_selection.add(m - 1)
                new_selection.add(m + 1)
        input_ids[index, list(new_selection)] = self.tokens["[MASK]"]
        return input_ids

    def check_if_special_token(self, token_id: int) -> bool:
        return (token_id == self.tokens["[GRAPH]"]) or (token_id == self.tokens["[TRIPLE]"])

