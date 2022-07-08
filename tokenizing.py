import numpy as np
from datasets import DatasetDict
from transformers import PreTrainedTokenizerFast
from transformers.file_utils import PaddingStrategy, ExplicitEnum
from scipy import stats

tokenizer = PreTrainedTokenizerFast(tokenizer_file="jericho/jericho_1024_15800.json")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.add_special_tokens({'mask_token': '[MASK]'})


class GameType(ExplicitEnum):
    TRAIN = "train"
    TEST = "test"
    ALL = "ALL"

    LOOSE = "LOOSE"
    KARN = "KARN"
    BALLYHOO = "BALLYHOO"
    ZORK2 = "ZORK2"
    ADVENTURELAND = "ADVENTURELAND"
    OMNIQUEST = "OMNIQUEST"
    WEAPON = "WEAPON"
    NINEOFIVE = "905"
    WISHBRINGER = "WISHBRINGER"
    NIGHT = "NIGHT"
    TRYST205 = "TRYST205"
    ZORK3 = "ZORK3"
    MURDAC = "MURDAC"
    AFFLICTED = "AFFLICTED"
    MOONLIT = "MOONLIT"
    DRAGON = "DRAGON"
    REVERB = "REVERB"
    JEWEL = "JEWEL"
    ENTER = "ENTER"
    SNACKTIME = "SNACKTIME"
    ENCHANTER = "ENCHANTER"
    ACORNCOURT = "ACORNCOURT"
    HUNTDARK = "HUNTDARK"
    GOLD = "GOLD"
    YOMOMMA = "YOMOMMA"
    INHUMANE = "INHUMANE"
    ZENON = "ZENON"
    DETECTIVE = "DETECTIVE"
    ZORK1 = "ZORK1"
    BALANCES = "BALANCES"
    LUDICORP = "LUDICORP"
    PENTARI = "PENTARI"
    ZTUU = "ZTUU"
    LIBRARY = "LIBRARY"
    TEMPLE = "TEMPLE"
    DEEPHOME = "DEEPHOME"


class PipelineType(ExplicitEnum):
    TEXT = "text"
    GRAPH = "graph"


def text_tokenize_function_without_padding(state):
    return tokenizer(
        text=state["text"],
        add_special_tokens=False,
        padding=PaddingStrategy.DO_NOT_PAD,
    )


def text_tokenize_function(state):
    return tokenizer(
        text=state["text"],
        truncation=True,
        add_special_tokens=False,
        padding=PaddingStrategy.MAX_LENGTH,
        max_length=1024,
    )


def text_plus_1_tokenize_function(state):
    return tokenizer(
        text=state["text_plus_1"],
        truncation=True,
        add_special_tokens=False,
        padding=PaddingStrategy.MAX_LENGTH,
        max_length=1024,
    )


def graph_tokenize_function(state):
    return tokenizer(
        text=state["graph"],
        truncation=True,
        add_special_tokens=False,
        padding=PaddingStrategy.MAX_LENGTH,
        max_length=1024,
    )


def graph_plus_1_tokenize_function(state):
    return tokenizer(
        text=state["graph_diff"],
        truncation=True,
        add_special_tokens=False,
        padding=PaddingStrategy.MAX_LENGTH,
        max_length=1024,
    )


def get_length(liste):
    return len(liste)


is_running_get_length = True


def tokenize_text_and_graph(preprocessed_dataset: DatasetDict, pipeline_type: PipelineType, remove_labels: bool = False, game_type: GameType = GameType.ALL) -> DatasetDict:
    tokenized_datasets = preprocessed_dataset
    if pipeline_type == PipelineType.TEXT:
        if remove_labels:
            tokenized_datasets = tokenized_datasets.map(text_tokenize_function, batched=True)
        else:
            if is_running_get_length:
                with open("JerichoWorld-main/data/input_length.txt", "a") as outfile:
                    # train = tokenized_datasets.map(text_tokenize_function_without_padding, batched=True)['train']['input_ids']
                    test = tokenized_datasets.map(text_tokenize_function_without_padding, batched=True)['test']['input_ids']
                    all_states = test  # + train
                    all_state_lengths = np.array(list(map(len, all_states)))
                    if (game_type == GameType.TEST) or (game_type == GameType.TRAIN) or (game_type == GameType.ALL):
                        with open("JerichoWorld-main/data/all_len_" + game_type.value + ".txt", "a") as all_len_outfile:
                            all_len_outfile.write("\n".join(str(item) for item in all_state_lengths))

                    mode = stats.mode(all_state_lengths)
                    text = "Game: " + game_type.value + \
                           "\n    Standard Deviation: " + str(np.std(all_state_lengths)) + \
                           "\n    Mean: " + str(np.mean(all_state_lengths)) + \
                           "\n    Median: " + str(np.median(all_state_lengths)) + \
                           "\n    Mode: " + str(mode[0][0]) + \
                           "\n    Max length: " + str(np.max(all_state_lengths)) + \
                           "\n    Min length: " + str(np.min(all_state_lengths)) + \
                           "\n--------------\n"
                    outfile.write(text)
            else:
                tokenized_datasets = tokenized_datasets.map(text_plus_1_tokenize_function, batched=True)
                tokenized_datasets = tokenized_datasets.rename_column("input_ids", "labels")
                tokenized_datasets = tokenized_datasets.remove_columns(["attention_mask"])
                tokenized_datasets = tokenized_datasets.map(text_tokenize_function, batched=True)
        print("Text-ACT-Token: " + str(tokenizer.convert_tokens_to_ids("[ACT]")))
        print("Text-GRAPH-Token: " + str(tokenizer.convert_tokens_to_ids("[GRAPH]")))
        print("Text-PAD-Token: " + str(tokenizer.convert_tokens_to_ids("[PAD]")))
    else:
        tokenized_datasets = tokenized_datasets.map(graph_plus_1_tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.rename_column("input_ids", "labels")
        tokenized_datasets = tokenized_datasets.map(graph_tokenize_function, batched=True)
        print("Text-ACT-Token: " + str(tokenizer.convert_tokens_to_ids("[ACT]")))
        print("Text-GRAPH-Token: " + str(tokenizer.convert_tokens_to_ids("[GRAPH]")))
        print("Text-PAD-Token: " + str(tokenizer.convert_tokens_to_ids("[PAD]")))

    tokenized_datasets = tokenized_datasets.remove_columns(["graph"])
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.remove_columns(["text_plus_1"])
    tokenized_datasets = tokenized_datasets.remove_columns(["graph_diff"])
    tokenized_datasets.set_format("torch")

    return tokenized_datasets
