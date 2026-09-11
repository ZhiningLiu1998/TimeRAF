import numpy as np

from ts_rag.event_detection import detect_events
from ts_rag.residual_utils import moving_average_residual, standardize_patch


def _flatten_patch(patch, normalize=True):
    data = standardize_patch(patch) if normalize else patch
    return data.reshape(-1)


def cosine_similarity(a, b, eps=1e-8):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(np.dot(a, b) / denom)


def euclidean_similarity(a, b):
    return float(-np.linalg.norm(a - b))


def mask_similarity(mask_a, mask_b):
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    return float(inter / union)


class EventRetriever:
    def __init__(self, memory_bank, metric="cosine", mask_lambda=0.1):
        self.memory_bank = memory_bank
        self.metric = metric
        self.mask_lambda = mask_lambda

    def _score(self, query_patch, query_mask, item):
        q = _flatten_patch(query_patch)
        k = _flatten_patch(item.E_res)
        if self.metric == "cosine":
            base_score = cosine_similarity(q, k)
        elif self.metric == "euclidean":
            base_score = euclidean_similarity(q, k)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        mask_score = mask_similarity(query_mask, item.active_mask)
        return base_score + self.mask_lambda * mask_score

    def retrieve_from_patch(self, query_event, top_k=5):
        scored = []
        for item in self.memory_bank.items:
            score = self._score(query_event["E_res"], query_event["active_mask"], item)
            scored.append(
                {
                    "score": score,
                    "E_res": item.E_res,
                    "E_raw": item.E_raw,
                    "Y_future": item.Y_future,
                    "future_residual": item.future_residual,
                    "active_mask": item.active_mask,
                    "metadata": item.metadata,
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def retrieve_from_window(self, x, y_future, y_base, args, sample_index, split="test"):
        residual_seq = moving_average_residual(x, kernel_size=args.residual_smooth_kernel)
        events, scores = detect_events(
            residual_seq=residual_seq,
            raw_seq=x,
            y_future=y_future,
            y_base=y_base,
            dataset_name=args.dataset,
            split=split,
            sample_index=sample_index,
            patch_len=args.event_patch_len,
            top_k=1,
            score_percentile=args.event_score_percentile,
            min_distance=args.event_min_distance,
            active_var_percentile=args.active_var_percentile,
        )
        query = {
            "E_res": events[0].E_res,
            "E_raw": events[0].E_raw,
            "active_mask": events[0].active_mask,
            "metadata": events[0].metadata,
            "event_scores": scores,
        }
        return query, self.retrieve_from_patch(query, top_k=args.retrieval_top_k)


class RawWindowRetriever:
    def __init__(self, train_bundle, metric="cosine"):
        self.metric = metric
        self.train_x = train_bundle["x"]
        self.train_y = train_bundle["y"]
        self.train_base = train_bundle.get("y_base")

    def _score(self, query_x, candidate_x):
        q = _flatten_patch(query_x)
        k = _flatten_patch(candidate_x)
        if self.metric == "cosine":
            return cosine_similarity(q, k)
        if self.metric == "euclidean":
            return euclidean_similarity(q, k)
        raise ValueError(f"Unsupported metric: {self.metric}")

    def retrieve(self, query_x, top_k=5):
        scores = np.array([self._score(query_x, item) for item in self.train_x])
        order = np.argsort(scores)[::-1][:top_k]
        items = []
        for idx in order:
            item = {
                "score": float(scores[idx]),
                "X": self.train_x[idx],
                "Y_future": self.train_y[idx],
                "metadata": {"sample_index": int(idx)},
            }
            if self.train_base is not None:
                item["future_residual"] = self.train_y[idx] - self.train_base[idx]
            items.append(item)
        return items
