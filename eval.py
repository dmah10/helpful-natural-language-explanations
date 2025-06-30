import numpy as np
import evaluate


def compute_metrics(preds, labels):
    f1 = evaluate.load("evaluate/metrics/f1/f1.py")
    accuracy = evaluate.load("evaluate/metrics/accuracy/accuracy.py")
    precision = evaluate.load("evaluate/metrics/precision/precision.py")
    recall = evaluate.load("evaluate/metrics/recall/recall.py")

    weighted_f1 = list(
        f1.compute(predictions=preds, references=labels, average="weighted").values()
    )[0]
    acc = list(accuracy.compute(predictions=preds, references=labels).values())[0]
    macro_f1 = list(
        f1.compute(predictions=preds, references=labels, average="macro").values()
    )[0]
    macro_recall = list(
        recall.compute(predictions=preds, references=labels, average="macro").values()
    )[0]
    macro_precision = list(
        precision.compute(
            predictions=preds, references=labels, average="macro"
        ).values()
    )[0]
    micro_f1 = list(
        f1.compute(predictions=preds, references=labels, average="micro").values()
    )[0]
    micro_recall = list(
        recall.compute(predictions=preds, references=labels, average="micro").values()
    )[0]
    micro_precision = list(
        precision.compute(
            predictions=preds, references=labels, average="micro"
        ).values()
    )[0]

    metrics = {
        "weighted_f1": weighted_f1,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "macro_precision": macro_precision,
        "micro_f1": micro_f1,
        "micro_recall": micro_recall,
        "micro_precision": micro_precision,
    }

    return metrics


def evaluate_model(model, trainer, tokenized_test, args):
    preds_output = trainer.predict(tokenized_test)
    preds = np.argmax(preds_output.predictions, axis=1)

    metrics = compute_metrics(preds, preds_output.label_ids)

    return metrics
