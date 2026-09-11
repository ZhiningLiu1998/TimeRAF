import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ts_rag.config import build_parser, build_setting, prepare_args
from ts_rag.evaluation import compute_metrics, save_json
from ts_rag.exp_wrapper import ExpLongTermForecastRAG
from ts_rag.post_test import predict_post_test


def main():
    parser = build_parser("Export predictions for the unused post-test ETT timeline")
    args = prepare_args(parser.parse_args())
    if args.dataset not in {"ETTh1", "ETTh2", "ETTm1", "ETTm2"}:
        parser.error("Only ETT datasets have a defined post-test timeline")

    setting = build_setting(args)
    exp = ExpLongTermForecastRAG(args)
    checkpoint = exp.load_checkpoint(setting, args.load_checkpoint)
    dataset, bundle = predict_post_test(exp, args)

    artifact_dir = os.path.join(args.output_dir, setting)
    os.makedirs(artifact_dir, exist_ok=True)
    with open(os.path.join(artifact_dir, "post_test_predictions.pkl"), "wb") as file:
        pickle.dump(bundle, file)
    metrics = compute_metrics(bundle["y_base"], bundle["y"])
    payload = {
        "setting": setting,
        "checkpoint": checkpoint,
        "first_forecast_origin": dataset.first_forecast_origin,
        "final_data_index": dataset.final_data_index,
        "forecast_origin_count": len(dataset),
        "metrics": metrics,
    }
    save_json(payload, os.path.join(artifact_dir, "post_test_baseline.json"))
    print(f"setting={setting}")
    print(f"checkpoint={checkpoint}")
    print(f"forecast_origin_count={len(dataset)}")
    print(f"post_test_baseline={metrics}")


if __name__ == "__main__":
    main()
