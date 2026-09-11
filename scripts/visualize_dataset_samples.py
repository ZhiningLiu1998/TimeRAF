import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ts_rag.config import build_parser, build_setting, prepare_args
from ts_rag.data_utils import load_window_bundle
from ts_rag.visualization import plot_dataset_samples


def main():
    parser = build_parser("Visualize dataset windows")
    args = prepare_args(parser.parse_args())
    setting = build_setting(args)
    test_bundle = load_window_bundle(args, split="test", max_samples=3)
    save_path = f"{args.output_dir}/{setting}/dataset_samples.png"
    plot_dataset_samples(test_bundle, args.dataset, variables=args.viz_variables, save_path=save_path)
    print(f"saved={save_path}")


if __name__ == "__main__":
    main()
