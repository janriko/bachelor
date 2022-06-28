from tokenize import Whitespace

import torch
from numpy.distutils.cpuinfo import cpu
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import PretrainedConfig, get_scheduler, TFGPT2LMHeadModel, GPT2Config, EncoderDecoderConfig, EncoderDecoderModel, BertConfig, GPT2Model, AutoModel, BertForMaskedLM, BertModel, \
    SpeechEncoderDecoderModel, SpeechEncoderDecoderConfig, GPT2LMHeadModel
from tqdm.auto import tqdm
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from bert_models import GraphEncoderModel
from tokenizer.GraphTokenizer import GraphTokenizer

graph_start_token_id = 2082  # '[GRAPH]'
graph_triple_token_id = 1999  # '[TRIPLE]'
graph_pad_token_id = 0  # ''

gpt_2_config = GPT2Config(
    vocab_size=50257,
    # tokenizer_class=GraphTokenizer,
    is_decoder=True,
    add_cross_attention=True,
    n_inner=3072,  # feed forward
    n_positions=1024,
    n_embd=768,
    n_head=6,
    n_layer=6,
)

# gpt_2_config = GPT2Config(
#     vocab_size=50257,
#     tokenizer_class=Whitespace,
#     is_decoder=True,
#     add_cross_attention=True,
#     n_inner=3072,  # feed forward
#     n_positions=1024,
#     n_embd=768,
#     n_head=6,
#     n_layer=6,
# )

encoder_decoder_config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config=GraphEncoderModel.config, decoder_config=gpt_2_config)
encoder_decoder_config.decoder_start_token_id = graph_start_token_id
encoder_decoder_config.pad_token_id = graph_triple_token_id
encoder_decoder_config.sep_token_id = graph_pad_token_id


def get_encoder_decoder_model(encoder, decoder) -> EncoderDecoderModel:  # (EncoderDecoderModel, AdamW, cpu, LambdaLR, tqdm):
    encoder_decoder_model: EncoderDecoderModel = EncoderDecoderModel(config=encoder_decoder_config, encoder=encoder, decoder=decoder)
    # optimizer: AdamW = AdamW(encoder_decoder_model.parameters(), lr=3e-4)
    # lr_scheduler = get_scheduler(
    #     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    # )
    # device: cpu = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #
    # progress_bar = tqdm(range(num_training_steps))
    #
    # encoder_decoder_model.to(device)
    # encoder_decoder_model.train()

    return encoder_decoder_model  # , optimizer, device, lr_scheduler, progress_bar


def get_graph_encoder_model() -> BertModel:
    # graph_encoder_model: BertForMaskedLM = torch.load(f="./graph_model_pretrained")
    graph_encoder_model: BertModel = BertModel.from_pretrained("./graph_model_pretrained/")
    return graph_encoder_model


def graph_model_train_loop(batch_dataset, model: EncoderDecoderModel, optimizer, device, lr_scheduler, progress_bar) -> BaseModelOutputWithPoolingAndCrossAttentions:
    batch_dataset = {k: v.to(device) for k, v in batch_dataset.items()}
    decoder_input_ids = model.prepare_decoder_input_ids_from_labels(batch_dataset["graph_labels"])
    outputs = model.forward(input_ids=batch_dataset["input_ids"], decoder_input_ids=decoder_input_ids,  labels=batch_dataset["graph_labels"], return_dict=True)  # , return_dict=True, attention_mask=batch_dataset["attention_mask"])
    # outputs = model(input_ids=batch_dataset["input_ids"], labels=batch_dataset["graph_labels"])  # , return_dict=True, attention_mask=batch_dataset["attention_mask"])
    loss, logits = outputs.loss, outputs.logits
    loss.backward()

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    progress_bar.update(1)

    return outputs


def get_graph_decoder_model() -> GPT2LMHeadModel:
    model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained("gpt2", config=gpt_2_config)
    return model

