import sys
import torch
from torch.types import Device
from torch.utils.data import Dataset
from transformers import BertConfig, BertForMaskedLM

from file_loader.FileLoader import remove_unnecessary_columns
from file_loader.FileLoader import get_preprocessed_and_tokenized_text_and_graph
from tokenizer.GraphTokenizer import GraphTokenizer

config = BertConfig(
    vocab_size=30522,
    hidden_size=768,
    num_attention_heads=6,
    num_hidden_layers=6,
    intermediate_size=3072,
    max_position_embeddings=1024,
    type_vocab_size=2,
    initializer_range=0.02,
    layer_norm_eps=1e-12,
    pad_token_id=0,
    position_embedding_type="absolute",
    use_cache=True,
    classifier_dropout=None,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)


def get_or_train_model(tokenizer: GraphTokenizer, device: Device) -> (BertForMaskedLM, Dataset):
    train_dataset: Dataset = get_preprocessed_and_tokenized_text_and_graph(isTraining=True, file_name="JerichoWorld-main/data/train.json")  # JerichoWorld-main/data/small_training_set.json")
    train_dataset = remove_unnecessary_columns(train_dataset, "text")
    train_dataset = remove_unnecessary_columns(train_dataset, "text_plus_1")
    train_dataset = remove_unnecessary_columns(train_dataset, "graph_diff")
    tokenized_dataset: Dataset = tokenizer.encode(train_dataset, "graph", shouldMask=True)

    model = BertForMaskedLM(config=config)
    return model, tokenized_dataset


def evaluate(tokenizer: GraphTokenizer) -> Dataset:
    eval_dataset: Dataset = get_preprocessed_and_tokenized_text_and_graph(isTraining=False, file_name="JerichoWorld-main/data/test.json")  # JerichoWorld-main/data/small_test_set.json")
    eval_dataset = remove_unnecessary_columns(eval_dataset, "text")
    eval_dataset = remove_unnecessary_columns(eval_dataset, "text_plus_1")
    eval_dataset = remove_unnecessary_columns(eval_dataset, "graph_diff")
    eval_dataset: Dataset = tokenizer.encode(eval_dataset, "graph", shouldMask=True)

    return eval_dataset



