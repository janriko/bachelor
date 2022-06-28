from torch.nn import Module, Linear, ModuleList
from transformers import AutoModel, PreTrainedModel


class Encoder(Module):
    def __init__(self, tokenizer, num_attention, num_layer):
        super().__init__()
        self.language_model: PreTrainedModel = AutoModel.from_pretrained(self.cfg.model.from_pretrained)
        self.tokenizer = tokenizer
        self.num_attention = num_attention
        self.num_layer = num_layer

    def freeze_lm(self):
        ...

    def forward(self, input):
        x = self.language_model(input)
        x = self.classifier(x)
        return x

    def instantiate_classifer(self):
        layer_stacks = [Linear(layer.n_in, layer.n_out) for layer in range(len(self.num_layer))]
        return ModuleList(layer_stacks)

    def train(self):
        pass


text_encoder = Encoder(...)
graph_encoder = Encoder(...)

