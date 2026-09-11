from dataclasses import dataclass

import numpy as np

from ts_rag.residual_utils import l2_event_score, percentile_threshold


@dataclass
class DetectedEvent:
    center: int
    start: int
    end: int
    score: float
    E_res: np.ndarray
    E_raw: np.ndarray
    Y_future: np.ndarray
    Y_base: np.ndarray
    future_residual: np.ndarray
    active_mask: np.ndarray
    metadata: dict


def _non_maximum_suppression(candidate_indices, min_distance):
    selected = []
    for idx in candidate_indices:
        if all(abs(idx - prev) >= min_distance for prev in selected):
            selected.append(idx)
    return selected


def detect_events(
    residual_seq,
    raw_seq,
    y_future,
    y_base,
    dataset_name,
    split,
    sample_index,
    patch_len=24,
    top_k=3,
    score_percentile=90.0,
    min_distance=12,
    active_var_percentile=75.0,
):
    scores = l2_event_score(residual_seq)
    threshold = percentile_threshold(scores, score_percentile)
    candidates = np.where(scores >= threshold)[0]
    if candidates.size == 0:
        candidates = np.array([int(scores.argmax())], dtype=int)
    ordered = list(candidates[np.argsort(scores[candidates])[::-1]])
    centers = _non_maximum_suppression(ordered, min_distance)[:top_k]

    events = []
    for center in centers:
        half = patch_len // 2
        start = max(0, center - half)
        end = min(residual_seq.shape[0], start + patch_len)
        start = max(0, end - patch_len)

        e_res = residual_seq[start:end]
        e_raw = raw_seq[start:end]
        max_abs = np.max(np.abs(e_res), axis=0)
        active_thresh = percentile_threshold(max_abs, active_var_percentile)
        active_mask = max_abs >= active_thresh
        metadata = {
            "dataset_name": dataset_name,
            "split": split,
            "sample_index": int(sample_index),
            "time_index": int(center),
            "event_score": float(scores[center]),
            "active_variable_count": int(active_mask.sum()),
        }
        events.append(
            DetectedEvent(
                center=int(center),
                start=int(start),
                end=int(end),
                score=float(scores[center]),
                E_res=e_res,
                E_raw=e_raw,
                Y_future=y_future,
                Y_base=y_base,
                future_residual=y_future - y_base,
                active_mask=active_mask.astype(np.float32),
                metadata=metadata,
            )
        )
    return events, scores
