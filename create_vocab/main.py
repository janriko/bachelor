import json

vocabulary: set = {'[GRAPH]', '[TRIPLE]', '[MASK]'}


def add_dict_entries_to_vocab(python_obj: dict):
    for liste_index in range(len(python_obj)):
        for state_trans_index in range(len(python_obj[liste_index])):
            trans = python_obj[liste_index][state_trans_index]
            state = trans["state"]["graph"]
            next_state = trans["next_state"]["graph"]
            graphs_lists = state + next_state

            [vocabulary.update(triple) for triple in graphs_lists]


if __name__ == '__main__':
    addIndices = True

    json_string_train = open("../JerichoWorld-main/data/train.json", "r")
    json_string_test = open("../JerichoWorld-main/data/test.json", "r")
    vocab_file = open("graph_encoder_vocabulary_training.txt", "w")

    # pre-processing
    python_obj_train: dict = json.load(json_string_train)
    python_obj_test: dict = json.load(json_string_test)

    add_dict_entries_to_vocab(python_obj_train)
    add_dict_entries_to_vocab(python_obj_test)

    if addIndices:
        index_list = (range(0, len(vocabulary)))
        vocab_dict = dict(zip(vocabulary, index_list))
        vocab_str = str(vocab_dict)
    else:
        vocab_str = str(vocabulary)
    vocab_file.write(vocab_str)
