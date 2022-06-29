import math
import torch

from transformers import TrainingArguments, Trainer, IntervalStrategy, EarlyStoppingCallback, BertModel, AutoTokenizer, PreTrainedTokenizerFast, DataCollatorForLanguageModeling, BertForMaskedLM, \
    BertConfig
from transformers.training_args import OptimizerNames

import preprocessing
from bert_models import GraphEncoderModel, TextModel
from tokenizer.GraphTokenizer import GraphTokenizer
from pathlib import Path
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
# from tokenizers import Tokenizer
from tokenizers.tokenizers import Tokenizer

from tokenizing import PipelineType

# config = BertConfig(  # Empty, not pretrained
#     vocab_size=30522,  # 15800
#     hidden_size=768,
#     num_attention_heads=8,
#     num_hidden_layers=6,
#     intermediate_size=3072,  # 4096
#     max_position_embeddings=512,  # 1024
#     position_embedding_type="absolute",
#     use_cache=False,
#     hidden_act="gelu",
#     hidden_dropout_prob=0.1,
#     attention_probs_dropout_prob=0.1,
# )

config = BertConfig(  # Empty, not pretrained
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
    attention_probs_dropout_prob=0.1,
)

if __name__ == "__main__":
    print(torch.__version__)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file="jericho/jericho.json")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # model = BertForMaskedLM.from_pretrained("bert-base-uncased", config=config)
    model = BertForMaskedLM(config=config)
    print("---Model loaded---")
    train_dataset, eval_dataset = preprocessing.preprocessing(PipelineType.TEXT, remove_labels=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)


    training_args = TrainingArguments(
        output_dir="results",
        evaluation_strategy=IntervalStrategy.STEPS,
        learning_rate=1e-4,
        num_train_epochs=50,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=30)],
        data_collator=data_collator
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Eval Outputs: {eval_results}")
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    # model.save_pretrained(save_directory="graph_model_pretrained")
    trainer.save_model(output_dir="text_graph_encoder_pretrained_empty")

