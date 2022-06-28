from typing import List

import numpy
import numpy as np
import torch
from numpy import ndarray
from sklearn.metrics import f1_score, precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import EncoderDecoderModel, Seq2SeqTrainingArguments, IntervalStrategy, Seq2SeqTrainer, EarlyStoppingCallback
from transformers.training_args import OptimizerNames
from evaluate import load

import preprocessing
from tokenizing import PipelineType

if __name__ == "__main__":
    torch.cuda.empty_cache()

    train_dataset, eval_dataset = preprocessing.preprocessing(PipelineType.GRAPH, preprocessing.GameType.ZORK)
    encoder_decoder_model: EncoderDecoderModel = EncoderDecoderModel.from_pretrained("./complete_trained_model_graph/")


    def convert_tokens_to_str(pred_or_label, with_sep_token: bool) -> List:
        return_list = []
        for state in pred_or_label:
            return_state = []
            first_token_string = ""
            if with_sep_token:
                first_token_string += str(state[0])
            first_token_string = str(state[1]) + str(state[2]) + str(state[3])
            return_state.append(first_token_string)
            for index in range(4, len(state), 4):
                token_string = ""
                if with_sep_token:
                    token_string += str(state[index])
                token_string = str(state[index + 1]) + str(state[index + 2]) + str(state[index + 3])
                return_state.append(token_string)
            return_list.append(return_state)
        return return_list

    def convert_tokens_to_str_token_level(pred_or_label, with_sep_token: bool) -> List:
        return_list = []
        for state in pred_or_label:
            return_state = []
            for index in range(0, len(state), 4):
                if with_sep_token:
                    return_state.append(str(state[index]))
                return_state.append(str(state[index + 1]))
                return_state.append(str(state[index + 2]))
                return_state.append(str(state[index + 3]))
            return_list.append(return_state)
        return return_list


    exact_match_metric = load("exact_match")
    f1_metric = load("f1")


    def trim_prediction(prediction):
        trimmed_pred = []
        for n in range(len(prediction)):
            end_index = np.where(prediction[n] == 0)[0][0]
            trimmed_pred.extend([prediction[n][:end_index]])
        return trimmed_pred


    def compute_f1(prediction: List[List], truth: List[List]):
        # f1_metric.compute(predictions=prediction, references=truth)
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
        # return exact_match_metric.compute(predictions=prediction, references=truth)
        em = 0
        for n in range(min(len(prediction), len(truth))):
            em += int(str(prediction[n]) == str(truth[n]))
        return em / len(prediction)


    def compute_metrics(pred, labels):
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

        # pred = trim_prediction(prediction=pred)
        # labels = trim_prediction(prediction=labels)
        # print("Predictions:\n")
        # print(pred)
        with open("graph_predictions", "w") as outfile:
            outfile.write("\n".join(str(item) for item in token_pred_with_sep))
        # print("\nLabels:\n")
        # print(labels)
        with open("graph_labels", "w") as outfile:
            outfile.write("\n".join(str(item) for item in token_labels_with_sep))
        return {"f1_token_without_sep": f1_graph_without_sep, "em_token_without_sep": em_graph_without_sep, "f1_token_with_sep": f1_graph_with_sep, "em_token_with_sep": em_graph_with_sep,
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
        # print(f"prediction: {predictions.device} type: {type(predictions)} ")
        # print(f"logits: {logits.device} type: {type(logits)} ")
        # print(f"labels: {batch['labels'].device} type: {type(batch['labels'])} ")
        y_pred.extend(predictions)
        y_true.extend(batch['labels'].cpu().data.numpy())
        # metric_em.add_batch(predictions=predictions, references=batch["labels"])
        # metric_f1.add_batch(predictions=predictions, references=batch["labels"])

    outputs = compute_metrics(y_pred, y_true)
    print("outputs: ")
    print(outputs)

#
    # # torch.backends.cuda.matmul.allow_tf32 = False
    # # torch.backends.cudnn.allow_tf32 = False
    #
    # training_args = Seq2SeqTrainingArguments(
    #     output_dir="graph_trainer",
    #     evaluation_strategy=IntervalStrategy.STEPS,
    #     learning_rate=3e-4,
    #     num_train_epochs=1,
    #     weight_decay=0.01,
    #     optim=OptimizerNames.ADAMW_TORCH,
    #     per_device_train_batch_size=1,
    #     per_device_eval_batch_size=1,
    #     load_best_model_at_end=True,
    #      eval_accumulation_steps=4,
    #     # generation_num_beams=15,
    # )
    #
    # trainer = Seq2SeqTrainer(
    #     model=encoder_decoder_model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     compute_metrics=compute_metrics,
    #     callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    # )
    #
    # eval_results = trainer.evaluate()
    # print(f"Eval Outputs: {eval_results}")
