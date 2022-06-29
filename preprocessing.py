from typing import List, Dict

from datasets import Dataset, DatasetDict
from datasets.features import features
from transformers.file_utils import ExplicitEnum

import file_loader
from data_models import *
from data_models.preprocessing import PreprocessedState
from data_models.preprocessing.PreprocessedDataset import PreprocessedDataset
from data_models.raw_state.JerichoDataset import JerichoDataset
from data_models.raw_state.JerichoState import JerichoState
from data_models.raw_state.JerichoStateTransition import JerichoStateTransition
from data_models.raw_state.JerichoTransitionList import JerichoTransitionList
from tokenizing import PipelineType


def map_json_to_python_obj(pythonObj: dict) -> JerichoDataset:
    all_state_transitions: JerichoDataset = JerichoDataset(worldList=JerichoTransitionList(list()))
    for liste_index in range(len(pythonObj)):
        all_state_transitions.append(JerichoTransitionList(transitionList=list()))
        for state_trans_index in range(len(pythonObj[liste_index])):
            all_state_transitions[liste_index].append(
                JerichoStateTransition(**(pythonObj[liste_index][state_trans_index])))
    return all_state_transitions


def map_all_state_transitions_to_preprocessed_state(dataset: JerichoDataset) -> Dataset:
    # pre_processed_list: PreprocessedDataset = PreprocessedDataset(preprocessed_state=list())
    pre_processed_dict = {
        "text": [],
        "text_plus_1": [],
        "graph": [],
        "graph_diff": []
    }
    all_obs_act_graph_triple = ""
    for listIndex in range(dataset.__len__()):
        for stateTransIndex in range((dataset[listIndex]).__len__()):
            # pre_processed_list.append(
            #     PreprocessedState(
            graph = concatenate_state_to_graph_encoder_string(dataset[listIndex][stateTransIndex].state.graph)
            text = concatenate_state_to_text_encoder_string(dataset[listIndex][stateTransIndex].state)
            graph_plus_text = graph + text

            pre_processed_dict["text"].append(graph_plus_text)
            pre_processed_dict["text_plus_1"].append(concatenate_state_to_text_plus_1_encoder_string(dataset[listIndex][stateTransIndex].next_state))
            pre_processed_dict["graph"].append(graph_plus_text)
            pre_processed_dict["graph_diff"].append(concatenate_state_to_graph_encoder_string(dataset[listIndex][stateTransIndex].next_state.graph))


    #         all_obs_act_graph_triple = all_obs_act_graph_triple + add_graph_to_set(dataset[listIndex][stateTransIndex].state.graph)
    #         all_obs_act_graph_triple = all_obs_act_graph_triple + add_state_to_set(dataset[listIndex][stateTransIndex].state)
    #         all_obs_act_graph_triple = all_obs_act_graph_triple + add_state_to_set(dataset[listIndex][stateTransIndex].next_state)
    #         all_obs_act_graph_triple = all_obs_act_graph_triple + add_graph_to_set(dataset[listIndex][stateTransIndex].next_state.graph)
    #
    # with open("all_obs_act_graph_triple", "a") as outfile:
    #     outfile.write(all_obs_act_graph_triple)
    # #     )
            # )
    return Dataset.from_dict(pre_processed_dict)


def add_graph_to_set(graph: List[List[str]]) -> str:
    return_str = ""

    for graph_entry in graph[0:]:
        return_str = return_str + graph_entry[0] + graph_entry[1] + graph_entry[2] + " "
    else:
        return_str = ""
    return return_str


def add_state_to_set(state: JerichoState) -> str:
    return_str = state.obs.replace("\n", " ")

    if state.valid_acts.__len__() > 0:
        for valid_act in state.valid_acts.values():
            return_str = return_str + " " + valid_act + " "
    else:
        return_str = ""

    return return_str


def concatenate_state_to_text_encoder_string(state: JerichoState) -> PreprocessedState:
    tags: Dict[str] = {
        "observableText": "[OBS]",
        "validActs": "[ACT]"
    }
    return_str = tags["observableText"] + state.obs.replace("\n", "")

    if state.valid_acts.__len__() > 0:
        for valid_act in state.valid_acts.values():
            return_str = return_str + tags["validActs"] + valid_act
    else:
        return_str = ""

    return return_str


def concatenate_state_to_text_plus_1_encoder_string(next_state: JerichoState) -> PreprocessedState:
    valid_acts = "[ACT]"

    return_str = ""

    if next_state.valid_acts.__len__() > 0:
        for valid_act in next_state.valid_acts.values():
            return_str = return_str + valid_acts + valid_act
    else:
        return_str = ""

    return return_str


def concatenate_graph_triple(tag: str, graph_entry: List[str]):
    return tag + graph_entry[0] + "§" + graph_entry[1] + "§" + graph_entry[2]


def concatenate_state_to_graph_encoder_string(graph: List[List[str]]):
    tags: Dict[str] = {
        "startingTag": "[GRAPH]",
        "inbetweenTag": "[TRIPLE]"
    }

    if graph.__len__() > 0:
        return_str = concatenate_graph_triple(tags["startingTag"], graph[0])

        for graph_entry in graph[1:]:
            return_str = return_str + concatenate_graph_triple(tags["inbetweenTag"], graph_entry)
    else:
        return_str = ""

    return return_str


class GameType(ExplicitEnum):
    ALL = ""
    ZORK = "ZORK"
    LIBRARY = "LIBRARY"
    DETECTIVE = "DETECTIVE"
    BALANCE = "BALANCE"
    PENTARI = "PENTARI"
    ZTUU = "ZTUU"
    LUDICORP = "LUDICORP"
    DEEPHOME = "DEEPHOME"
    TEMPLE = "TEMPLE"


def preprocessing(pipeline_type: PipelineType, game: GameType = GameType.ALL, remove_labels: bool = False) -> (DatasetDict, DatasetDict):
    """
    either get preprocessed dataset from cache (give file name as parameter)
    or recompile dataset from file (give filename with jericho dataset)
    """
    test_set = "JerichoWorld-main/data/small_test_set" + game.value + ".json"
    preprocessed_training_data_set: Dataset = file_loader.FileLoader.get_preprocessed_and_tokenized_text_and_graph(isTraining=True, file_name="JerichoWorld-main/data/small_training_set.json")
    preprocessed_testing_data_set: Dataset = file_loader.FileLoader.get_preprocessed_and_tokenized_text_and_graph(isTraining=False, file_name=test_set)

    dataset_dict = DatasetDict({"train": preprocessed_training_data_set, "test": preprocessed_testing_data_set})

    """
    create tokenized ids with dataset (give dataset)
    or load from cache (don't give dataset as parameter)
    """
    tokenized_datasets: DatasetDict = file_loader.get_tokenized_text_and_graph_ids(pipeline_type, dataset_dict, remove_labels)

    # print(token_ids.get("train")["t_input_ids"])
    # print(token_ids.get("train")["t_attention_mask"])

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]

    return train_dataset, eval_dataset
