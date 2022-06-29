from tokenizer.GraphTokenizer import GraphTokenizer
from pathlib import Path
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
# from tokenizers import Tokenizer
from tokenizers.tokenizers import Tokenizer

if __name__ == "__main__":
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[OBS]", "[ACT]", "[GRAPH]", "[TRIPLE]", "[PAD]", "[MASK]"])  ##TODO: add your special tokens here
    tokenizer.train(files=["data/all_obs_act_graph_triple"], trainer=trainer)
    max_sequence_length = 1024
    tokenizer.enable_truncation(max_length=max_sequence_length)
    tokenizer.enable_padding(length=max_sequence_length)
    output_path = "./jericho/"
    if not Path(output_path).exists():
        Path(output_path).mkdir()
    else:
        pass
    tokenizer.save(output_path + "jericho.json")
    print(tokenizer.get_vocab_size())
