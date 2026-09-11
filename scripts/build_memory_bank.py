import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ts_rag.config import build_parser, build_setting, prepare_args
from ts_rag.pipeline import export_prediction_bundle, get_or_build_memory_bank, train_or_load_experiment


def main():
    parser = build_parser("Build residual event memory bank for time-series RAG")
    args = prepare_args(parser.parse_args())
    setting = build_setting(args)
    exp = train_or_load_experiment(args, setting)
    train_bundle = export_prediction_bundle(exp, args, split="train", setting=setting)
    bank, bank_path = get_or_build_memory_bank(train_bundle, args, setting)
    print(f"setting={setting}")
    print(f"memory_bank_path={bank_path}")
    print(bank.stats())


if __name__ == "__main__":
    main()
