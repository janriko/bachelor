import math
import torch

from transformers import TrainingArguments, Trainer, IntervalStrategy, EarlyStoppingCallback
from transformers.training_args import OptimizerNames

from bert_models import GraphEncoderModel
from tokenizer.GraphTokenizer import GraphTokenizer

if __name__ == "__main__":
    print(torch.__version__)

    # torch.cuda.empty_cache()

    tokenizer = GraphTokenizer()
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

