from typing import List, Union
import ast

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
                    index_of_next_triple = len(text)
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
            attention_mask[state_index].append(([1]*attention_mask_counter).extend([0]*(1024-attention_mask_counter)))
        new_dataset = text.add_column("input_ids", input_ids)
        new_dataset = new_dataset.add_column("attention_mask", attention_mask)
        new_dataset = new_dataset.add_column("token_type_ids", [[0]*1024] * len(input_ids))
        new_dataset = new_dataset.remove_columns(column_name)
        # if shouldMask:
        #     new_dataset = self.mask_dataset(new_dataset, 0.2)
        new_dataset.set_format("torch")
        return new_dataset

    def decode(self, words: List[int]):
        pass
