import numpy as np


def similarity_to_weights(scores, mode="similarity_weighted"):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return scores
    if mode == "uniform":
        return np.ones_like(scores) / len(scores)
    shifted = scores - scores.max()
    weights = np.exp(shifted)
    weights_sum = weights.sum()
    if weights_sum <= 0:
        return np.ones_like(scores) / len(scores)
    return weights / weights_sum


def aggregate_retrieved_targets(retrieved_items, key, aggregation="similarity_weighted"):
    if not retrieved_items:
        raise ValueError("No retrieved items to aggregate")
    scores = [item["score"] for item in retrieved_items]
    weights = similarity_to_weights(scores, aggregation)
    stacked = np.stack([item[key] for item in retrieved_items], axis=0)
    return np.tensordot(weights, stacked, axes=(0, 0))


def correct_forecast(y_base, retrieved_items, correction_mode="future_residual", aggregation="similarity_weighted", blend_alpha=1.0):
    if not retrieved_items:
        return y_base.copy(), np.zeros_like(y_base)

    if correction_mode == "future_residual" and all(item.get("future_residual") is not None for item in retrieved_items):
        correction = aggregate_retrieved_targets(retrieved_items, "future_residual", aggregation=aggregation)
        final = y_base + blend_alpha * correction
        return final, correction

    target = aggregate_retrieved_targets(retrieved_items, "Y_future", aggregation=aggregation)
    correction = target - y_base
    final = y_base + blend_alpha * correction
    return final, correction
