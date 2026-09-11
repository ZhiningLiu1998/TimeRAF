import os
import pickle

import numpy as np

from ts_rag.config import build_setting
from ts_rag.correction import correct_forecast
from ts_rag.evaluation import evaluate_predictions, save_json
from ts_rag.exp_wrapper import ExpLongTermForecastRAG
from ts_rag.memory_bank import EventMemoryBank, build_memory_bank
from ts_rag.retriever import EventRetriever, RawWindowRetriever


def _artifact_dir(args, setting):
    path = os.path.join(args.output_dir, setting)
    os.makedirs(path, exist_ok=True)
    return path


def train_or_load_experiment(args, setting):
    exp = ExpLongTermForecastRAG(args)
    if args.train_model:
        exp.train(setting)
    else:
        exp.load_checkpoint(setting, args.load_checkpoint)
    return exp


def export_prediction_bundle(exp, args, split, setting):
    bundle = exp.predict_split(flag=split, inverse=args.inverse)
    artifact_dir = _artifact_dir(args, setting)
    if args.save_predictions:
        path = os.path.join(artifact_dir, f"{split}_predictions.pkl")
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
    return bundle


def load_prediction_bundle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_or_build_memory_bank(train_bundle, args, setting):
    artifact_dir = _artifact_dir(args, setting)
    bank_path = args.memory_bank_path or os.path.join(artifact_dir, "memory_bank.pkl")
    if args.memory_bank_path and os.path.exists(args.memory_bank_path):
        return EventMemoryBank.load(args.memory_bank_path), bank_path

    bank = build_memory_bank(train_bundle, args)
    if args.save_memory_bank:
        bank.save(bank_path)
    save_json(bank.stats(), os.path.join(artifact_dir, "memory_bank_stats.json"))
    return bank, bank_path


def run_all_methods(train_bundle, test_bundle, memory_bank, args):
    event_retriever = EventRetriever(memory_bank, metric=args.retrieval_metric, mask_lambda=args.mask_lambda)
    raw_retriever = RawWindowRetriever(train_bundle, metric=args.retrieval_metric)

    baseline_pred = test_bundle["y_base"]
    retrieval_only_pred = []
    residual_rag_pred = []
    raw_rag_pred = []
    event_scores = []
    case_payloads = []

    for idx, (x, y, y_base) in enumerate(zip(test_bundle["x"], test_bundle["y"], test_bundle["y_base"])):
        query_event, retrieved_events = event_retriever.retrieve_from_window(
            x=x,
            y_future=y,
            y_base=y_base,
            args=args,
            sample_index=idx,
            split=test_bundle["split"],
        )
        event_scores.append(float(query_event["metadata"]["event_score"]))

        raw_neighbors = raw_retriever.retrieve(x, top_k=args.retrieval_top_k)
        retrieval_only = np.mean(np.stack([item["Y_future"] for item in raw_neighbors], axis=0), axis=0)
        retrieval_only_pred.append(retrieval_only)

        corrected_pred, correction = correct_forecast(
            y_base=y_base,
            retrieved_items=retrieved_events,
            correction_mode=args.correction_mode,
            aggregation=args.aggregation,
            blend_alpha=args.blend_alpha,
        )
        residual_rag_pred.append(corrected_pred)

        raw_corrected, _ = correct_forecast(
            y_base=y_base,
            retrieved_items=raw_neighbors,
            correction_mode=args.correction_mode,
            aggregation=args.aggregation,
            blend_alpha=args.blend_alpha,
        )
        raw_rag_pred.append(raw_corrected)

        case_payloads.append(
            {
                "sample_index": idx,
                "x": x,
                "y": y,
                "y_base": y_base,
                "y_corr": corrected_pred,
                "y_retrieval": retrieval_only,
                "correction": correction,
                "query_event": query_event,
                "retrieved": retrieved_events,
                "raw_neighbors": raw_neighbors,
                "event_score": float(query_event["metadata"]["event_score"]),
            }
        )

    outputs = {
        "baseline_only": baseline_pred,
        "retrieval_only": np.stack(retrieval_only_pred, axis=0),
        "residual_event_rag": np.stack(residual_rag_pred, axis=0),
        "raw_window_rag": np.stack(raw_rag_pred, axis=0),
        "true": test_bundle["y"],
        "event_scores": np.asarray(event_scores, dtype=np.float32),
        "cases": case_payloads,
    }
    return outputs


def evaluate_and_save(outputs, args, setting):
    artifact_dir = _artifact_dir(args, setting)
    results = {}
    for name in ["baseline_only", "retrieval_only", "residual_event_rag", "raw_window_rag"]:
        if name == "raw_window_rag" and not args.raw_retrieval_correction:
            continue
        results[name] = evaluate_predictions(
            pred=outputs[name],
            true=outputs["true"],
            event_scores=outputs["event_scores"],
            percentile=args.event_split_percentile,
        )
    save_json(results, os.path.join(artifact_dir, "evaluation.json"))
    with open(os.path.join(artifact_dir, "case_payloads.pkl"), "wb") as f:
        pickle.dump(outputs["cases"], f)
    return results
