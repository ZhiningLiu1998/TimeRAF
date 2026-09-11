import json
import os

import numpy as np

from utils.metrics import metric


def compute_metrics(pred, true):
    mae, mse, rmse, mape, mspe = metric(pred, true)
    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "mape": float(mape),
        "mspe": float(mspe),
    }


def compute_pems_metrics(pred, true):
    pred = np.asarray(pred)
    true = np.asarray(true)
    error = pred - true
    with np.errstate(divide="ignore", invalid="ignore"):
        percentage_error = np.abs(error / true)
    percentage_error = np.where(percentage_error > 5, 0, percentage_error)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mape": float(np.mean(percentage_error)),
    }


def compute_protocol_metrics(pred, true, task_family):
    if task_family == "pems":
        return compute_pems_metrics(pred, true)
    metrics = compute_metrics(pred, true)
    return {"mse": metrics["mse"], "mae": metrics["mae"]}


def subset_metrics(pred, true, mask):
    mask = np.asarray(mask).astype(bool)
    if mask.sum() == 0:
        return None
    return compute_metrics(pred[mask], true[mask])


def evaluate_predictions(pred, true, event_scores, percentile=75.0):
    threshold = float(np.percentile(event_scores, percentile))
    heavy_mask = event_scores >= threshold
    low_mask = event_scores < threshold
    return {
        "overall": compute_metrics(pred, true),
        "event_heavy": subset_metrics(pred, true, heavy_mask),
        "non_event": subset_metrics(pred, true, low_mask),
        "event_score_threshold": threshold,
        "event_heavy_count": int(heavy_mask.sum()),
        "non_event_count": int(low_mask.sum()),
    }


def save_json(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
