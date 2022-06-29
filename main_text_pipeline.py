from typing import Dict

import math
import numpy
import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers.training_args import OptimizerNames

from bert_models import TextModel, GraphEncoderModel
import preprocessing
from torch.utils.data import DataLoader
from transformers import logging, GPT2Tokenizer, IntervalStrategy, Trainer, TrainingArguments, EarlyStoppingCallback, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, \
    EncoderDecoderModel

from file_loader.FileLoader import remove_unnecessary_columns
from tokenizing import PipelineType

if __name__ == "__main__":
    torch.cuda.empty_cache()

    train_dataset, eval_dataset = preprocessing.preprocessing(PipelineType.TEXT)

    # train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=8)
    # eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=8)
    #
    # num_epochs = config.num_epochs
    # num_training_steps = config.num_training_steps(train_dataloader)
    #
    text_encoder_model = TextModel.get_text_encoder_data()
    text_decoder_model = TextModel.get_text_decoder_data()
    # encoder_decoder_model, optimizer, device, lr_schedular, progress_bar = TextModel.get_encoder_decoder_model(text_encoder_model, text_decoder_model)  # , num_training_steps)
    encoder_decoder_model = TextModel.get_encoder_decoder_model(text_encoder_model, text_decoder_model)  # , num_training_steps)

    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # encoder_decoder_model = nn.DataParallel(encoder_decoder_model, device_ids=[2, 3])
    # ids: list = [2, 3]
    # encoder_decoder_model = DistributedDataParallel(encoder_decoder_model, device_ids=ids, output_device=2)
    # torch.cuda.set_device(device=device)
    # encoder_decoder_model = nn.DataParallel(encoder_decoder_model, device_ids=[2, 3])
    # encoder_decoder_model.to(device)

    # for epoch in range(num_epochs):
    #     for batch in train_dataloader:
    #         encoder_outputs = TextModel.text_model_train_loop(
    #             batch_dataset=batch,
    #             model=encoder_decoder_model,
    #             optimizer=optimizer,
    #             device=device,
    #             lr_scheduler=lr_schedular,
    #             progress_bar=progress_bar
    #         )

    # def compute_metrics(p):
    #     pred, labels = p
    #     pred = np.argmax(pred, axis=-1)
    #     f1 = f1_score(y_true=labels, y_pred=pred)
    #     precision_score()
    #     em = np.all(pred == labels, axis=1).mean()
    #     return {"f1": f1, "em": em}

    training_args = Seq2SeqTrainingArguments(
        output_dir="text_trainer",
        evaluation_strategy=IntervalStrategy.STEPS,
        learning_rate=1e-3,
        num_train_epochs=30,
        weight_decay=0.01,
        optim=OptimizerNames.ADAMW_TORCH,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        load_best_model_at_end=True,
        # fp16=True,
        generation_num_beams=15,
        # seed=9235,
    )

    trainer = Seq2SeqTrainer(
        model=encoder_decoder_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()

    trainer.save_model(output_dir="complete_trained_model_text_not_pretrained_gpt_2")
