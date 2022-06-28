<<<<<<< HEAD
import math
import torch

from transformers import TrainingArguments, Trainer, IntervalStrategy, EarlyStoppingCallback
from transformers.training_args import OptimizerNames

from bert_models import GraphEncoderModel
=======
from datasets import DatasetDict
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertConfig

import file_loader
import g_config
import preprocessing
from file_loader import FileLoader
>>>>>>> parent of fade205... both models completely trained
from tokenizer.GraphTokenizer import GraphTokenizer

config = BertConfig(
    vocab_size=30522,
    hidden_size=768,
    num_attention_heads=6,
    num_hidden_layers=6,
    intermediate_size=3072,
    max_position_embeddings=512,
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


if __name__ == "__main__":
    print(torch.__version__)

    train_dataset: Dataset = file_loader.FileLoader.get_preprocessed_and_tokenized_text_and_graph(isTraining=True, file_name="JerichoWorld-main/data/small_training_set.json")
    eval_dataset: Dataset = file_loader.FileLoader.get_preprocessed_and_tokenized_text_and_graph(isTraining=False, file_name="JerichoWorld-main/data/small_test_set.json")

    train_dataset = FileLoader.remove_unnecessary_columns(train_dataset)
    eval_dataset = FileLoader.remove_unnecessary_columns(eval_dataset)

    # dataset_dict = DatasetDict({"train": train_dataset, "test": eval_dataset})

    encoder = BertModel(config=config)
    tokenizer = GraphTokenizer()
<<<<<<< HEAD
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model, train_dataset = GraphEncoderModel.get_or_train_model(tokenizer=tokenizer, device=device)
    print("---Model loaded---")
    eval_dataset = GraphEncoderModel.evaluate(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="results",
        evaluation_strategy=IntervalStrategy.STEPS,
        learning_rate=3e-4,
        num_train_epochs=30,
        weight_decay=0.01,
        optim=OptimizerNames.ADAMW_TORCH,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Eval Outputs: {eval_results}")
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    # model.save_pretrained(save_directory="graph_model_pretrained")
    trainer.save_model(output_dir="graph_model_pretrained")
=======

    tokenized_dataset = tokenizer.encode(train_dataset)


    tokenized_datasets: DatasetDict = FileLoader.get_tokenized_text_and_graph_ids()
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]

    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=16)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=16)

    num_epochs = g_config.num_epochs
    num_training_steps = g_config.num_training_steps(train_dataloader)

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

>>>>>>> parent of fade205... both models completely trained

