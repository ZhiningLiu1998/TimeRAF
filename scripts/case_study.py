import os
import pickle
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
from ts_rag.visualization import plot_case_study


def main():
    parser = build_parser("Generate residual-event RAG case studies")
    args = prepare_args(parser.parse_args())
    setting = build_setting(args)
    exp = train_or_load_experiment(args, setting)
    train_bundle = export_prediction_bundle(exp, args, split="train", setting=setting)
    test_bundle = export_prediction_bundle(exp, args, split="test", setting=setting)
    memory_bank, _ = get_or_build_memory_bank(train_bundle, args, setting)
    outputs = run_all_methods(train_bundle, test_bundle, memory_bank, args)
    evaluate_and_save(outputs, args, setting)

    cases = outputs["cases"]
    ranked = sorted(
        cases,
        key=lambda item: ((item["y_base"] - item["y"]) ** 2).mean() - ((item["y_corr"] - item["y"]) ** 2).mean(),
        reverse=True,
    )
    artifact_dir = os.path.join(args.output_dir, setting, "case_studies")
    os.makedirs(artifact_dir, exist_ok=True)
    selected = ranked[: args.case_count]
    for rank, case in enumerate(selected):
        save_path = os.path.join(artifact_dir, f"case_{rank+1}_sample_{case['sample_index']}.png")
        plot_case_study(case, variables=args.viz_variables, save_path=save_path)
    with open(os.path.join(artifact_dir, "selected_cases.pkl"), "wb") as f:
        pickle.dump(selected, f)
    print(f"saved_cases={len(selected)}")
    print(f"case_dir={artifact_dir}")


if __name__ == "__main__":
    main()
