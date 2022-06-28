from typing import Dict

import math
import numpy
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.metrics._scorer import metric
from torch import nn
from transformers.training_args import OptimizerNames

import config

from bert_models import GraphModel
import preprocessing
from torch.utils.data import DataLoader
from transformers import logging, GPT2Tokenizer, TFGPT2LMHeadModel, TrainingArguments, IntervalStrategy, Trainer, EarlyStoppingCallback, Seq2SeqTrainingArguments, Seq2SeqTrainer

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
        num_train_epochs=50,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=35)]
    )

    trainer.train()

    # eval_results = trainer.evaluate()
    # print(f"Eval Outputs: {eval_results}")
    # print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    trainer.save_model(output_dir="complete_trained_model_graph")
