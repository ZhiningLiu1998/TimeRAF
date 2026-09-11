import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ts_rag.config import build_parser, build_setting, prepare_args
from ts_rag.pipeline import (
    evaluate_and_save,
    export_prediction_bundle,
    get_or_build_memory_bank,
    run_all_methods,
    train_or_load_experiment,
)


def main():
    parser = build_parser("Run residual-event RAG forecasting benchmark")
    args = prepare_args(parser.parse_args())
    setting = build_setting(args)
    exp = train_or_load_experiment(args, setting)
    train_bundle = export_prediction_bundle(exp, args, split="train", setting=setting)
    test_bundle = export_prediction_bundle(exp, args, split="test", setting=setting)
    memory_bank, bank_path = get_or_build_memory_bank(train_bundle, args, setting)
    outputs = run_all_methods(train_bundle, test_bundle, memory_bank, args)
    results = evaluate_and_save(outputs, args, setting)
    print(f"setting={setting}")
    print(f"artifact_dir={os.path.join(args.output_dir, setting)}")
    print(f"memory_bank_path={bank_path}")
    for name, payload in results.items():
        print(name, payload["overall"])


if __name__ == "__main__":
    main()
