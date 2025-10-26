from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import json



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

# def calculate_threshold()