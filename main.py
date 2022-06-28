import random

import transformers
from typing import Dict, List, TextIO
import json

from datasets import DatasetDict
from torch import LongTensor, tensor
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import BertTokenizer, BertModel, BertConfig, PretrainedConfig, get_scheduler
import torch

import math
import numpy
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.metrics._scorer import metric
from torch import nn
from transformers.training_args import OptimizerNames

import config
import file_loader.FileLoader
import datasets

<<<<<<< HEAD:main_graph_pipeline.py
from bert_models import GraphModel
import preprocessing
from torch.utils.data import DataLoader
from transformers import logging, GPT2Tokenizer, TFGPT2LMHeadModel, TrainingArguments, IntervalStrategy, Trainer, EarlyStoppingCallback, Seq2SeqTrainingArguments, Seq2SeqTrainer
=======
from Aggregator.Aggregator import Aggregator
from Decoder.ActionDecoder import ActionDecoder
from Decoder.GraphDecoder import GraphDecoder
from bert_models import TextModel, GraphModel
from data_models import *
import preprocessing
import pickle
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from transformers import logging
from data_models.preprocessing.PreprocessedDataset import PreprocessedDataset
from data_models.raw_state.JerichoDataset import JerichoDataset
from data_models.tokenized.TokenizedDataset import TokenizedDataset
>>>>>>> parent of fade205... both models completely trained:main.py

from tokenizing import PipelineType

if __name__ == "__main__":
    torch.cuda.empty_cache()
#     torch.cuda.memory_summary(device=None, abbreviated=False)

    train_dataset, eval_dataset = preprocessing.preprocessing(PipelineType.GRAPH)

    # train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=8)
    # eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=8)

    # num_epochs = config.num_epochs
    # num_training_steps = config.num_training_steps(train_dataloader)

    graph_encoder_model = GraphModel.get_graph_encoder_model()
    graph_decoder_model = GraphModel.get_graph_decoder_model()
    encoder_decoder_model = GraphModel.get_encoder_decoder_model(encoder=graph_encoder_model, decoder=graph_decoder_model)
    # encoder_decoder_model, optimizer, device, lr_schedular, progress_bar = GraphModel.get_encoder_decoder_model(encoder=graph_encoder_model, decoder=graph_decoder_model, num_training_steps=num_training_steps)


    training_args = Seq2SeqTrainingArguments(
        output_dir="graph_trainer",
        evaluation_strategy=IntervalStrategy.STEPS,
        learning_rate=3e-4,
        num_train_epochs=30,
        weight_decay=0.01,
        optim=OptimizerNames.ADAMW_TORCH,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        load_best_model_at_end=True,
        generation_num_beams=15,
    )

    trainer = Seq2SeqTrainer(
        model=encoder_decoder_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=15)]
    )

    trainer.train()

    # eval_results = trainer.evaluate()
    # print(f"Eval Outputs: {eval_results}")
    # print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    trainer.save_model(output_dir="complete_trained_model_graph")
