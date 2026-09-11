import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ts_rag.config import build_parser, build_setting, prepare_args
from ts_rag.evaluation import compute_metrics, save_json
from ts_rag.pipeline import export_prediction_bundle, train_or_load_experiment


ETT_DATASETS = {"ETTh1", "ETTh2", "ETTm1", "ETTm2"}


def _jsonable_args(args):
    payload = {}
    for key, value in vars(args).items():
        if key == "device":
            payload[key] = str(value)
        elif isinstance(value, (str, int, float, bool, list, type(None))):
            payload[key] = value
        else:
            payload[key] = str(value)
    return payload


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = build_parser("Train and export a reproducible ETT TimeMixer baseline")
    args = prepare_args(parser.parse_args())
    if args.dataset not in ETT_DATASETS:
        parser.error("run_ett_baseline.py only supports ETTh1, ETTh2, ETTm1, and ETTm2")
    if args.model != "TimeMixer":
        parser.error("The reference baseline must use --model TimeMixer")

    setting = build_setting(args)
    start = time.time()
    exp = train_or_load_experiment(args, setting)

    metrics = {}
    shapes = {}
    for split in ("train", "val", "test"):
        bundle = export_prediction_bundle(exp, args, split=split, setting=setting)
        metrics[split] = compute_metrics(bundle["y_base"], bundle["y"])
        shapes[split] = {key: list(bundle[key].shape) for key in ("x", "y", "y_base")}

    artifact_dir = os.path.join(args.output_dir, setting)
    data_file = os.path.join(args.root_path, args.data_path)
    metadata = {
        "setting": setting,
        "elapsed_seconds": time.time() - start,
        "data_sha256": _sha256(data_file),
        "args": _jsonable_args(args),
        "shapes": shapes,
        "metrics": metrics,
    }
    save_json(metadata, os.path.join(artifact_dir, "baseline_run.json"))

    print(f"setting={setting}")
    print(f"artifact_dir={artifact_dir}")
    for split, values in metrics.items():
        print(f"{split}: mse={values['mse']:.8f}, mae={values['mae']:.8f}")


if __name__ == "__main__":
    main()
