from typing import List

import numpy
import numpy as np
import sklearn
import torch
from numpy import ndarray
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import EncoderDecoderModel

import preprocessing
from tokenizing import PipelineType

def run_eval(game: preprocessing.GameType, model_name: str, encoder_decoder_model: EncoderDecoderModel):
    torch.cuda.empty_cache()

    train_dataset, eval_dataset = preprocessing.do_preprocessing(PipelineType.GRAPH, game=game)

    def convert_tokens_to_str(pred_or_label, with_sep_token: bool):
        separator = ","
        return_list = []
        for state in pred_or_label:
            return_state = []
            first_token_string = ""
            if with_sep_token:
                first_token_string += str(state[0])
                first_token_string += separator
            first_token_string = str(state[1]) + separator + str(state[2]) + separator + str(state[3])
            return_state.append(first_token_string)
            for index in range(4, len(state), 4):
                token_string = ""
                if with_sep_token:
                    token_string += str(state[index])
                    token_string += separator
                token_string = str(state[index + 1]) + separator + str(state[index + 2]) + separator + str(state[index + 3])
                return_state.append(token_string)
            return_list.append(np.array(return_state))
        return return_list

    def convert_tokens_to_str_token_level(pred_or_label, with_sep_token: bool):
        return_list = []
        for state in pred_or_label:
            return_state = []
            for index in range(0, len(state), 4):
                if with_sep_token:
                    return_state.append(str(state[index]))
                return_state.append(str(state[index + 1]))
                return_state.append(str(state[index + 2]))
                return_state.append(str(state[index + 3]))

            return_list.append(np.array(return_state))
        return return_list

    def trim_prediction(prediction):
        trimmed_pred = []
        for n in range(len(prediction)):
            end_index = np.where((prediction[n] == "4") | (prediction[n] == "4,4,4") | (prediction[n] == "4,4,4,4"))[0][0]
            trimmed_pred.extend([prediction[n][:end_index]])
        return trimmed_pred

    def compute_f1(prediction: List[ndarray], truth: List[ndarray]):
        f1 = 0
        for n in range(min(len(prediction), len(truth))):
            common_tokens = set(prediction[n]) & set(truth[n])
            if len(common_tokens) == 0:
                f1 += 0
            prec = len(common_tokens) / len(prediction[n])
            rec = len(common_tokens) / len(truth[n])
            if (prec and rec) == 0:
                f1 += 0
            else:
                f1 += 2 * (prec * rec) / (prec + rec)
        return f1 / len(prediction)


    def compute_exact_match(prediction, truth):
        truth = trim_prediction(truth)
        prediction = trim_prediction(prediction)
        ac = 0
        for n in range(len(prediction)):
            index = min(len(truth[n]), len(prediction[n])) - 1
            ac += sklearn.metrics.accuracy_score(truth[n][:index], prediction[n][:index])
        return ac / len(prediction)

        # em = 0
        # for n in range(min(len(prediction), len(truth))):
        #     em += int(str(prediction[n]) == str(truth[n]))
        # return em / len(prediction)


    def compute_metrics(pred, labels):
        with open("graph_predictions", "w") as outfile:
            outfile.write("\n".join(str(item) for item in pred))
        with open("graph_labels", "w") as outfile:
            outfile.write("\n".join(str(item) for item in labels))

        token_pred_without_sep = convert_tokens_to_str_token_level(pred, False)
        token_pred_with_sep = convert_tokens_to_str_token_level(pred, True)
        token_labels_without_sep = convert_tokens_to_str_token_level(labels, False)
        token_labels_with_sep = convert_tokens_to_str_token_level(labels, True)

        token_f1_without_sep = compute_f1(prediction=token_pred_without_sep, truth=token_labels_without_sep)
        token_f1_with_sep = compute_f1(prediction=token_pred_with_sep, truth=token_labels_with_sep)
        token_em_without_sep = compute_exact_match(prediction=token_pred_without_sep, truth=token_labels_without_sep)
        token_em_with_sep = compute_exact_match(prediction=token_pred_with_sep, truth=token_labels_with_sep)

        graph_pred_without_sep = convert_tokens_to_str(pred, False)
        graph_pred_with_sep = convert_tokens_to_str(pred, True)
        graph_labels_without_sep = convert_tokens_to_str(labels, False)
        graph_labels_with_sep = convert_tokens_to_str(labels, True)

        f1_graph_without_sep = compute_f1(truth=graph_labels_without_sep, prediction=graph_pred_without_sep)
        f1_graph_with_sep = compute_f1(truth=graph_labels_with_sep, prediction=graph_pred_with_sep)
        em_graph_without_sep = compute_exact_match(truth=graph_labels_without_sep, prediction=graph_pred_without_sep)
        em_graph_with_sep = compute_exact_match(truth=graph_labels_with_sep, prediction=graph_pred_with_sep)

        return {"f1_token_without_sep": token_f1_without_sep, "em_token_without_sep": token_em_without_sep, "f1_token_with_sep": token_f1_with_sep, "em_token_with_sep": token_em_with_sep,
                "f1_graph_without_sep": f1_graph_without_sep, "em_graph_without_sep": em_graph_without_sep, "f1_graph_with_sep": f1_graph_with_sep, "em_graph_with_sep": em_graph_with_sep}

    eval_dataloader = DataLoader(eval_dataset, batch_size=16)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    encoder_decoder_model.eval()
    y_pred = []
    y_true = []

    encoder_decoder_model.to(device)
    for _, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = encoder_decoder_model(**batch)

        logits = outputs.logits.cpu().data.numpy()
        predictions = numpy.argmax(logits, axis=-1)
        y_pred.extend(predictions)
        y_true.extend(batch['labels'].cpu().data.numpy())

    outputs = compute_metrics(y_pred, y_true)
    print("outputs: ")
    print(outputs)
    with open("graph_eval_outputs", "a") as outfile:
        text = "Outputs for model: " + model_name + "\nFor the game: " + game.value + "\n" + str(outputs) + "\n--------------\n"
        outfile.write(text)


if __name__ == "__main__":
    all_games = [
        preprocessing.GameType.PENTARI,
        #
        preprocessing.GameType.ZORK1,
        preprocessing.GameType.LIBRARY,
        preprocessing.GameType.DETECTIVE,
        preprocessing.GameType.BALANCES,
        #
        preprocessing.GameType.ZTUU,
        preprocessing.GameType.LUDICORP,
        preprocessing.GameType.DEEPHOME,
        preprocessing.GameType.TEMPLE,
        preprocessing.GameType.TEST
    ]

    all_seeds = [3540, 6843, 7561]

    # run_eval(preprocessing.GameType.ZORK, ""
    for seed in all_seeds:
        model_folder = "./complete_trained_model_graph_512_seed_" + str(seed) + "/"
        encoder_decoder_model: EncoderDecoderModel = EncoderDecoderModel.from_pretrained(model_folder)
        for game in all_games:
            print(game)
            run_eval(game=game, model_name=str(model_folder), encoder_decoder_model=encoder_decoder_model)
