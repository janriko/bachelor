from tokenize import Whitespace

import torch
from transformers import PretrainedConfig, BertModel, get_scheduler, TFGPT2LMHeadModel, GPT2Config, EncoderDecoderConfig, EncoderDecoderModel, BertConfig, GPT2Model, AutoModel, BertLMHeadModel, \
    GPT2LMHeadModel
from tqdm.auto import tqdm
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

import tokenizing
from tokenizing import double_tokenizer_text

text_action_token_id = 30523  # '[ACT]'
text_pad_token_id = 0  # '[PAD]'

bert_config = BertConfig(
    vocab_size=30522,
    hidden_size=768,
    num_attention_heads=8,
    num_hidden_layers=6,
    intermediate_size=3072,
    max_position_embeddings=512,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    output_scores=True,
    output_hidden_states=True,
    output_attentions=True,
    use_cache=False,
)

gpt_2_config = GPT2Config(
    vocab_size=50257,
    tokenizer_class=Whitespace,
    is_decoder=True,
    add_cross_attention=True,
    n_inner=3072,  # feed forward
    n_positions=1024,
    n_embd=768,
    n_head=6,
    n_layer=6,
)

encoder_decoder_config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config=bert_config, decoder_config=gpt_2_config, output_scores=True)
encoder_decoder_config.decoder_start_token_id = double_tokenizer_text.cls_token_id
encoder_decoder_config.pad_token_id = double_tokenizer_text.pad_token_id


def get_encoder_decoder_model(encoder, decoder) -> EncoderDecoderModel:  # , num_training_steps) -> (BertModel, AdamW, cpu, LambdaLR, tqdm):  # (EncoderDecoderModel, AdamW, cpu, LambdaLR, tqdm):
    encoder_decoder_model: EncoderDecoderModel = EncoderDecoderModel(config=encoder_decoder_config, encoder=encoder, decoder=decoder)
    encoder_decoder_model.encoder.resize_token_embeddings(len(tokenizing.double_tokenizer_text))
    # encoder_decoder_model.decoder.resize_token_embeddings(len(tokenizing.double_tokenizer_text))
    # encoder_decoder_model: BertModel = encoder
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


def get_text_encoder_data() -> BertModel:
    text_encoder_model: BertModel = BertModel.from_pretrained("bert-base-uncased", config=bert_config)  # maybe use longformer-base-###
    # text_encoder_model.resize_token_embeddings(30522 + 2)
    return text_encoder_model


def text_model_train_loop(batch_dataset, model: EncoderDecoderModel, optimizer, device, lr_scheduler, progress_bar) -> BaseModelOutputWithPoolingAndCrossAttentions:
    batch_dataset = {k: v.to(device) for k, v in batch_dataset.items()}
    outputs = model.forward(input_ids=batch_dataset["input_ids"], decoder_input_ids=batch_dataset["labels"], decoder_attention_mask=batch_dataset["attention_mask"], return_dict=True)  # , return_dict=True, attention_mask=batch_dataset["attention_mask"])
    loss: torch.FloatTensor = outputs.loss
    logits: torch.FloatTensor = outputs.logits
    loss.backward()

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    progress_bar.update(1)

    return outputs


def get_text_decoder_data() -> GPT2LMHeadModel:
    # model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained("gpt2", config=gpt_2_config)
    model: GPT2LMHeadModel = GPT2LMHeadModel(config=gpt_2_config)
    return model

# def text_decoder_train_loop(batch_dataset, model, optimizer, device, lr_scheduler, progress_bar) -> BaseModelOutputWithPoolingAndCrossAttentions:
#     batch_dataset = {k: v.to(device) for k, v in batch_dataset.items()}
#     outputs = model(**batch_dataset)
#     loss = outputs.loss
#     loss.backward()
#
#     optimizer.step()
#     lr_scheduler.step()
#     optimizer.zero_grad()
#     progress_bar.update(1)
#
#     return outputs
