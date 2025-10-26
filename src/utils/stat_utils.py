from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import json

def get_json_records(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def stat_ppl(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    li = []
    for record in data:
        li.append(record['ppl'])
    return {
        "size": len(li),
        "min": min(li),
        "max": max(li),
        "mean": float(np.mean(li)),
        "median": float(np.median(li)),
        "std": float(np.std(li))
    }


def calculate_threshold(filename):
    return stat_ppl(filename)["mean"]


def evaluate_metrics_ppl(records: list[dict], threshold):
    data = records
    y_true = [item['label'] for item in data]
    y_pred = []
    for item in data:
        if item['ppl'] <= threshold:
            y_pred.append(True)
        else:
            y_pred.append(False)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
    }

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print("=" * 50)

    return metrics


def evaluate_metrics_charsetn(records: list[dict]):
    data = records

    y_true = [item['label'] for item in data]
    y_pred = []
    for item in data:
        if item['output']:
            if item['result']['confidence'] > 0.7:
                y_pred.append(True)
            else:
                y_pred.append(not item['label'])
        if not item['output']:
            if item['result']['confidence'] > 0.1:
                y_pred.append(False)
            else:
                y_pred.append(not item['label'])

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),

    }

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print("=" * 50)

    return metrics


def evaluate_metrics_llm(records: list[dict]):
    data=records

    y_true = []
    y_pred = []

    for item in data:
        y_true.append(item['label'])
        if item['output']['is_linguistic_acceptable']:
            if item['output']['linguistic_acceptability'] > 0.5:
                y_pred.append(1)
            else:
                y_pred.append(not item['label'])
        if not item['output']['is_linguistic_acceptable']:
            if item['output']['linguistic_acceptability'] < 0.5:
                y_pred.append(0)
            else:
                y_pred.append(not item['label'])

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
    }

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print("=" * 50)

    return metrics


if __name__ == '__main__':
    print(stat_ppl("../../data/threshold_samples_25/shift_jis_pos_samples.json"))
