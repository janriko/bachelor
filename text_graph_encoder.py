import math
import torch

from transformers import TrainingArguments, Trainer, IntervalStrategy, EarlyStoppingCallback, BertModel, AutoTokenizer, PreTrainedTokenizerFast, DataCollatorForLanguageModeling, BertForMaskedLM, \
    BertConfig
from transformers.training_args import OptimizerNames

import preprocessing

from tokenizing import PipelineType

config = BertConfig(
    vocab_size=15800,
    hidden_size=768,
    num_attention_heads=8,
    num_hidden_layers=6,
    intermediate_size=4096,
    max_position_embeddings=1024,
    position_embedding_type="absolute",
    use_cache=False,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    output_scores=True,
    output_hidden_states=True,
    output_attentions=True,
    pad_token_id=4
)

if __name__ == "__main__":
    print(torch.__version__)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file="jericho/jericho_1024_15800.json")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = BertForMaskedLM(config=config)
    print("---Model loaded---")
    train_dataset, eval_dataset = preprocessing.do_preprocessing(PipelineType.TEXT, remove_labels=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir="results",
        evaluation_strategy=IntervalStrategy.STEPS,
        learning_rate=1e-3,
        num_train_epochs=40,
        weight_decay=0.01,
        optim=OptimizerNames.ADAMW_TORCH,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=15)],
        data_collator=data_collator
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Eval Outputs: {eval_results}")
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    # model.save_pretrained(save_directory="graph_model_pretrained")
    trainer.save_model(output_dir="text_graph_encoder_pretrained_1024")

