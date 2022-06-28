from io import StringIO
from typing import Generic, TypeVar, List

import json
import pickle
<<<<<<< HEAD
from typing import List, Union

from datasets import Dataset, DatasetDict
import preprocessing
from data_models.raw_state.JerichoDataset import JerichoDataset
from tokenizing import tokenize_text_and_graph, PipelineType

=======

from datasets import Dataset, DatasetDict

import preprocessing
from data_models.preprocessing.PreprocessedState import PreprocessedState
from data_models.preprocessing.PreprocessedDataset import PreprocessedDataset
from data_models.raw_state.JerichoDataset import JerichoDataset
from data_models.tokenized.TokenizedDataset import TokenizedDataset

# preprocessed_dataset_text_file = 'cached_data/PreprocessedDataset/preprocessed_states/concatenatedText.txt'
# preprocessed_dataset_graph_file = 'cached_data/PreprocessedDataset/preprocessed_states/graphWithTriples.txt'
from tokenizing import tokenize_text_and_graph

>>>>>>> parent of fade205... both models completely trained
preprocessed_training_dataset_file = 'cached_data/preprocessed_training_dataset'
preprocessed_testing_dataset_file = 'cached_data/preprocessed_testing_dataset'
tokenized_dataset_file = 'cached_data/tokenized_dataset'


# gets tokenized text [O_t + V_t] and tokenized graph [G_t]
# If a filename is given -> re preprocess it from file and save to cache file
#                   else -> use cached data
# O_t:
#       [OBS] West of House You are standing in an open field west of a white house...
# V_t:
#       [ACT] go north [ACT] open mailbox ...
# G_t:
#       [GRAPH] you, in, West of House [TRIPLE] West of house, has, mailbox [TRIPLE] mailbox, is, openable ...
def get_preprocessed_and_tokenized_text_and_graph(isTraining: bool, file_name: str = None) -> Dataset:
    if file_name is None:
        # load from cached file
        if isTraining:
            preprocessed_dataset: Dataset = pickle.load(open(preprocessed_training_dataset_file, 'rb'))
        else:
            preprocessed_dataset: Dataset = pickle.load(open(preprocessed_testing_dataset_file, 'rb'))
    else:
        json_string = open(file_name, "r")

        # pre-processing
        python_obj: dict = json.load(json_string)
        custom_jericho_dataset: JerichoDataset = preprocessing.map_json_to_python_obj(python_obj)

        # create pre processed dataset:
        preprocessed_dataset: Dataset = preprocessing.map_all_state_transitions_to_preprocessed_state(dataset=custom_jericho_dataset)

        # save dataset to cache file
        if isTraining:
            pickle.dump(preprocessed_dataset, open(preprocessed_training_dataset_file, 'wb'))
        else:
            pickle.dump(preprocessed_dataset, open(preprocessed_testing_dataset_file, 'wb'))

    return preprocessed_dataset


def get_tokenized_text_and_graph_ids(pipeline_type: PipelineType, dataset: DatasetDict = None) -> DatasetDict:
    if dataset is None:
        # load from cached file
        tokenized_dataset: DatasetDict = pickle.load(open(tokenized_dataset_file, 'rb'))
    else:
        tokenized_dataset: DatasetDict = tokenize_text_and_graph(dataset, pipeline_type)
        # save dataset to cache file
        pickle.dump(tokenized_dataset, open(tokenized_dataset_file, 'wb'))

    return tokenized_dataset


def remove_unnecessary_columns(dataset: Dataset, column_names: Union[List[str], str]) -> Dataset:
    if type(column_names) is str:
        column_names = [column_names]
    for column in column_names:
        dataset = dataset.remove_columns(column)
    return dataset
