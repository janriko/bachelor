from tokenize import Whitespace
from transformers import BertModel, GPT2Config, EncoderDecoderConfig, EncoderDecoderModel, GPT2LMHeadModel

import text_graph_encoder

text_action_token_id = 4  # '[ACT]'
text_pad_token_id = 1  # '[PAD]'

gpt_2_config = GPT2Config(
    vocab_size=50257,
    tokenizer_class=Whitespace,
    is_decoder=True,
    add_cross_attention=True,
    n_inner=4096,  # feed forward
    n_positions=512,
    n_embd=768,
    n_head=8,
    n_layer=6,
)


def get_encoder_decoder_model(encoder, decoder) -> EncoderDecoderModel:
    encoder_decoder_config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config=text_graph_encoder.config, decoder_config=gpt_2_config, output_scores=True)
    encoder_decoder_config.decoder_start_token_id = 1  # 50266  # double_tokenizer_text.cls_token_id
    encoder_decoder_config.pad_token_id = 4  # 3  # double_tokenizer_text.pad_token_id

    encoder_decoder_model: EncoderDecoderModel = EncoderDecoderModel(config=encoder_decoder_config, encoder=encoder, decoder=decoder)
    encoder_decoder_model.encoder.resize_token_embeddings(15800)

    return encoder_decoder_model


def get_text_encoder_data() -> BertModel:
    text_encoder_model: BertModel = BertModel.from_pretrained("./text_graph_encoder_pretrained_1024/")
    return text_encoder_model


def get_text_decoder_data() -> GPT2LMHeadModel:
    model: GPT2LMHeadModel = GPT2LMHeadModel(config=gpt_2_config)
    return model
