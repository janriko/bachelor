from typing import List

import numpy
import numpy as np
import torch
from datasets import load_metric
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

    train_dataset, eval_dataset = preprocessing.preprocessing(PipelineType.TEXT, preprocessing.GameType.ALL)
    encoder_decoder_model: EncoderDecoderModel = EncoderDecoderModel.from_pretrained("./complete_trained_model_text/")

    def trim_prediction(prediction):
        trimmed_pred = []
        for n in range(len(prediction)):
            end_index = np.where(prediction[n] == 0)[0][0]
            trimmed_pred.extend([prediction[n][:end_index]])
        return trimmed_pred

    def compute_f1(prediction: List[List], truth: List[List]):
        # f1_metric.compute(predictions=prediction, references=truth)
        f1 = 0
        for n in range(len(prediction)):
            common_tokens = set(prediction[n]) & set(truth[n])
            if len(common_tokens) == 0:
                f1 += 0
            prec = len(common_tokens) / len(prediction[n])
            rec = len(common_tokens) / len(truth[n])

            if (prec and rec) == 0:
                f1 += 0
            else:
                f1 += 2 * (prec * rec) / (prec + rec)

            # f1 += 2 * (prec * rec) / (prec + rec)
        return f1 / len(prediction)


    exact_match_metric = load("exact_match")
    f1_metric = load("f1")

    def compute_exact_match(prediction, truth):
        # return exact_match_metric.compute(predictions=prediction, references=truth)
        em = 0
        for n in range(len(prediction)):
            em += int(str(prediction[n]) == str(truth[n]))
        return em / len(prediction)

    def compute_metrics(pred, labels):
        # pred = trim_prediction(prediction=pred)
        with open("text_predictions", "w") as outfile:
            outfile.write("\n".join(str(item) for item in pred))
        # labels = trim_prediction(prediction=labels)
        with open("text_labels", "w") as outfile:
            outfile.write("\n".join(str(item) for item in labels))
        f1 = compute_f1(prediction=pred, truth=labels)
        em = compute_exact_match(prediction=pred, truth=labels)
        return {"f1": f1, "em": em}


    # metric_f1 = load_metric("f1")
    # metric_em = load_metric("exact_match")

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
        #metric_em.add_batch(predictions=predictions, references=batch["labels"])
        #metric_f1.add_batch(predictions=predictions, references=batch["labels"])

    outputs = compute_metrics(y_pred, y_true)
    print("outputs: ")
    print(outputs)


    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False
    #
    # training_args = Seq2SeqTrainingArguments(
    #     output_dir="text_trainer",
    #     evaluation_strategy=IntervalStrategy.STEPS,
    #     learning_rate=3e-4,
    #     num_train_epochs=1,
    #     weight_decay=0.01,
    #     optim=OptimizerNames.ADAMW_TORCH,
    #     per_device_train_batch_size=1,
    #     per_device_eval_batch_size=1,
    #     load_best_model_at_end=True,
    #     eval_accumulation_steps=4,
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
