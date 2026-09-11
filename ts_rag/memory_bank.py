import pickle
import random
from dataclasses import dataclass, field

import numpy as np

from ts_rag.event_detection import detect_events
from ts_rag.residual_utils import moving_average_residual


@dataclass
class MemoryItem:
    E_res: np.ndarray
    E_raw: np.ndarray
    Y_future: np.ndarray
    active_mask: np.ndarray
    metadata: dict
    Y_base: np.ndarray = None
    future_residual: np.ndarray = None


@dataclass
class EventMemoryBank:
    items: list = field(default_factory=list)
    dataset_name: str = ""
    config: dict = field(default_factory=dict)

    def add(self, item):
        self.items.append(item)

    def stats(self):
        if not self.items:
            return {
                "event_count": 0,
                "mean_event_score": 0.0,
                "mean_active_variables": 0.0,
            }
        scores = [item.metadata["event_score"] for item in self.items]
        active_counts = [item.metadata["active_variable_count"] for item in self.items]
        return {
            "event_count": len(self.items),
            "mean_event_score": float(np.mean(scores)),
            "mean_active_variables": float(np.mean(active_counts)),
        }

    def sample(self, count=3, seed=2021):
        if not self.items:
            return []
        rng = random.Random(seed)
        return rng.sample(self.items, min(count, len(self.items)))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"dataset_name": self.dataset_name, "config": self.config, "items": self.items}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return cls(items=payload["items"], dataset_name=payload["dataset_name"], config=payload["config"])


def build_memory_bank(bundle, args):
    bank = EventMemoryBank(
        dataset_name=args.dataset,
        config={
            "event_patch_len": args.event_patch_len,
            "event_top_k": args.event_top_k,
            "event_score_percentile": args.event_score_percentile,
            "active_var_percentile": args.active_var_percentile,
            "residual_smooth_kernel": args.residual_smooth_kernel,
        },
    )

    for sample_index, (x, y, y_base) in enumerate(zip(bundle["x"], bundle["y"], bundle["y_base"])):
        residual_seq = moving_average_residual(x, kernel_size=args.residual_smooth_kernel)
        events, _ = detect_events(
            residual_seq=residual_seq,
            raw_seq=x,
            y_future=y,
            y_base=y_base,
            dataset_name=args.dataset,
            split=bundle["split"],
            sample_index=sample_index,
            patch_len=args.event_patch_len,
            top_k=args.event_top_k,
            score_percentile=args.event_score_percentile,
            min_distance=args.event_min_distance,
            active_var_percentile=args.active_var_percentile,
        )
        for event in events:
            bank.add(
                MemoryItem(
                    E_res=event.E_res.astype(np.float32),
                    E_raw=event.E_raw.astype(np.float32),
                    Y_future=event.Y_future.astype(np.float32),
                    active_mask=event.active_mask.astype(np.float32),
                    metadata=event.metadata,
                    Y_base=event.Y_base.astype(np.float32),
                    future_residual=event.future_residual.astype(np.float32),
                )
            )
        if args.memory_event_limit and len(bank.items) >= args.memory_event_limit:
            bank.items = bank.items[: args.memory_event_limit]
            break
    return bank
