from typing import List, Dict

import dataset
from datasets import Dataset, DatasetDict
from transformers import BertTokenizer, BatchEncoding
from transformers.file_utils import PaddingStrategy, ExplicitEnum

from data_models.preprocessing.PreprocessedDataset import PreprocessedDataset
from data_models.tokenized.TokenizedDataset import TokenizedDataset

# from data_models.tokenized.TokenizedState import TokenizedState
from tokenizer.GraphTokenizer import GraphTokenizer

special_text_tokens: Dict[str, str] = {
    "cls_token": "[OBS]",
    "sep_token": "[ACT]",
}

special_graph_tokens: Dict[str, str] = {
    "cls_token": "[GRAPH]",
    "sep_token": "[TRIPLE]",
}

special_tokens_double_graph = {
    "cls_token": "[GRAPH]",
    "sep_token": "[TRIPLE]",
    "additional_special_tokens": ["[OBS]", "[ACT]"]
}

special_tokens_double_text = {
    "cls_token": "[OBS]",
    "sep_token": "[ACT]"
   #  "additional_special_tokens": ["[GRAPH]", "[TRIPLE]"]
}


double_tokenizer_graph = BertTokenizer.from_pretrained("bert-base-uncased")
double_tokenizer_text = BertTokenizer.from_pretrained("bert-base-uncased")

# add custom tokens
double_tokenizer_graph.add_special_tokens(special_tokens_dict=special_tokens_double_graph)
double_tokenizer_text.add_special_tokens(special_tokens_dict=special_tokens_double_text)


class PipelineType(ExplicitEnum):
    TEXT = "text"
    GRAPH = "graph"


def text_tokenize_function(state):
    return double_tokenizer_text(
        text=state["text"],
        truncation=True,
        add_special_tokens=False,
        padding=PaddingStrategy.MAX_LENGTH,
        max_length=512,
        # return_tensors="pt"
    )


def text_plus_1_tokenize_function(state):
    return double_tokenizer_text(
        text=state["text_plus_1"],
        truncation=True,
        add_special_tokens=False,
        padding=PaddingStrategy.MAX_LENGTH,
        max_length=512,
        # return_tensors="pt"
    )


def graph_tokenize_function(state):
    return double_tokenizer_text(
        text=state["graph"],
        truncation=True,
        add_special_tokens=False,
        padding=PaddingStrategy.MAX_LENGTH,
        max_length=512,
        # return_tensors="pt"
    )


def graph_plus_1_tokenize_function(state):
    return double_tokenizer_text(
        text=state["graph_diff"],
        truncation=True,
        add_special_tokens=False,
        padding=PaddingStrategy.MAX_LENGTH,
        max_length=512,
        # return_tensors="pt"
    )

def tokenize_text_and_graph(preprocessed_dataset: DatasetDict, pipeline_type: PipelineType) -> DatasetDict:
    tokenized_datasets = preprocessed_dataset
    if pipeline_type == PipelineType.TEXT:
        tokenized_datasets = tokenized_datasets.map(text_plus_1_tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.rename_column("input_ids", "labels")
        tokenized_datasets = tokenized_datasets.remove_columns(["attention_mask"])
        tokenized_datasets = tokenized_datasets.remove_columns(["token_type_ids"])
        tokenized_datasets = tokenized_datasets.map(text_tokenize_function, batched=True)
        print("Text-Separation-Token: " + str(double_tokenizer_text.convert_tokens_to_ids(special_text_tokens["sep_token"])))
        print("Text-Pad-Token: " + str(double_tokenizer_text.convert_tokens_to_ids("[PAD]")))
    else:
        tokenized_datasets = tokenized_datasets.map(graph_plus_1_tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.rename_column("input_ids", "labels")
        tokenized_datasets = tokenized_datasets.map(graph_tokenize_function, batched=True)
        print("Graph-Separation-Token: " + str(double_tokenizer_graph.convert_tokens_to_ids(special_graph_tokens["sep_token"])))
        print("Graph-Pad-Token: " + str(double_tokenizer_graph.convert_tokens_to_ids("[PAD]")))
        # train_dataset = tokenized_datasets["train"]
        # test_dataset = tokenized_datasets["test"]
        #
        # tokenized_train_dataset = double_tokenizer_graph.encode(train_dataset, "graph", False)
        # tokenized_test_dataset = double_tokenizer_graph.encode(test_dataset, "graph", False)
        #
        # temp_tokenized_train_dataset = double_tokenizer_graph.encode(train_dataset, "graph_diff", False)
        # temp_tokenized_test_dataset = double_tokenizer_graph.encode(test_dataset, "graph_diff", False)
        #
        # tokenized_train_dataset = tokenized_train_dataset.add_column("labels", temp_tokenized_train_dataset["input_ids"].tolist())
        # tokenized_test_dataset = tokenized_test_dataset.add_column("labels", temp_tokenized_test_dataset["input_ids"].tolist())
        #
        # tokenized_datasets = DatasetDict({"train": tokenized_train_dataset, "test": tokenized_test_dataset})

    tokenized_datasets = tokenized_datasets.remove_columns(["graph"])
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.remove_columns(["text_plus_1"])
    tokenized_datasets = tokenized_datasets.remove_columns(["graph_diff"])
    tokenized_datasets.set_format("torch")

    return tokenized_datasets
