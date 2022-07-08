from typing import List

import evaluate
import numpy
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import EncoderDecoderModel

import preprocessing
from tokenizing import PipelineType, GameType


def run_eval(game: preprocessing.GameType, model_name: str, encoder_decoder_model: EncoderDecoderModel):
    torch.cuda.empty_cache()

    train_dataset, eval_dataset = preprocessing.do_preprocessing(PipelineType.TEXT, game=game)

    def trim_prediction(prediction):
        trimmed_pred = []
        for n in range(len(prediction)):
            end_index = np.where(prediction[n] == 4)[0][0]
            trimmed_pred.extend([prediction[n][:end_index]])
        return trimmed_pred

    def compute_f1(prediction: List[List], truth: List[List]):
        f1 = 0
        for n in range(min(len(prediction), len(truth))):
            common_tokens = set(prediction[n]) & set(truth[n])

            prec = len(common_tokens) / len(prediction[n])
            rec = len(common_tokens) / len(truth[n])

            if (prec and rec) == 0:
                f1 += 0
            else:
                f1 += 2 * (prec * rec) / (prec + rec)

        return f1 / len(prediction)

    accuracy_metric = evaluate.load("accuracy")

    def compute_exact_match(prediction, truth):
        truth = trim_prediction(truth)
        prediction = trim_prediction(prediction)
        ac = 0
        for n in range(len(prediction)):
            index = min(len(truth[n]), len(prediction[n])) - 1
            acc_dic = accuracy_metric.compute(references=truth[n][:index], predictions=prediction[n][:index])
            ac += acc_dic["accuracy"]
        return ac / len(prediction)

    def compute_metrics(pred, labels):
        with open("text_predictions", "w") as outfile:
            outfile.write("\n".join(str(item) for item in trim_prediction(pred)))
        with open("text_labels", "w") as outfile:
            outfile.write("\n".join(str(item) for item in trim_prediction(labels)))
        f1 = compute_f1(prediction=pred, truth=labels)
        em = compute_exact_match(prediction=pred, truth=labels)
        return {"f1": f1, "em": em}

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
    with open("text_eval_outputs", "a") as outfile:
        text = "Outputs for model: " + model_name + "\nFor the game: " + game.value + "\n" + str(outputs) + "\n--------------\n"
        outfile.write(text)


if __name__ == "__main__":
    all_games = [
        GameType.ZORK1,
        GameType.LIBRARY,
        GameType.DETECTIVE,
        GameType.BALANCES,
        GameType.PENTARI,
        GameType.ZTUU,
        GameType.LUDICORP,
        GameType.DEEPHOME,
        GameType.TEMPLE,
        GameType.TEST
    ]

    all_seeds = [3540, 6843, 7561]

    # run_eval(preprocessing.GameType.ZORK, "")
    for seed in all_seeds:
        model_folder = "./complete_trained_model_text_512_seed_" + str(seed) + "/"
        encoder_decoder_model: EncoderDecoderModel = EncoderDecoderModel.from_pretrained(model_folder)
        for game in all_games:
            run_eval(game=game, model_name=str(model_folder), encoder_decoder_model=encoder_decoder_model)
