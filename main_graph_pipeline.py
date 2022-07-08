import torch
from transformers.training_args import OptimizerNames

from bert_models import GraphModel
import preprocessing
from transformers import IntervalStrategy, EarlyStoppingCallback, Seq2SeqTrainingArguments, Seq2SeqTrainer

from tokenizing import PipelineType


def run_training(seed: int):
    torch.cuda.empty_cache()

    train_dataset, eval_dataset = preprocessing.do_preprocessing(PipelineType.GRAPH)

    graph_encoder_model = GraphModel.get_graph_encoder_model()
    graph_decoder_model = GraphModel.get_graph_decoder_model()
    encoder_decoder_model = GraphModel.get_encoder_decoder_model(encoder=graph_encoder_model, decoder=graph_decoder_model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="graph_trainer",
        evaluation_strategy=IntervalStrategy.STEPS,
        learning_rate=1e-3,
        num_train_epochs=50,
        weight_decay=0.01,
        optim=OptimizerNames.ADAMW_TORCH,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        load_best_model_at_end=True,
        generation_num_beams=15,
        seed=seed,
    )

    trainer = Seq2SeqTrainer(
        model=encoder_decoder_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=15)]
    )

    trainer.train()

    trainer.save_model(output_dir="complete_trained_model_graph_1024_seed_" + str(seed))


if __name__ == "__main__":
    seeds = [3540, 6843, 7561]
    for seed in seeds:
        run_training(seed=seed)
