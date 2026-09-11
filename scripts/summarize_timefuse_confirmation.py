import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ts_rag.matrix import (
    build_confirmation_summary,
    load_manifest,
    save_json_atomic,
)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize the frozen held-out TimeFuse matrix complement"
    )
    parser.add_argument(
        "--manifest",
        default="./docs/timefuse_experiment_manifest.jsonl",
    )
    parser.add_argument("--matrix-summary", required=True)
    parser.add_argument(
        "--protocol",
        default="./docs/timefuse_confirmation_protocol.json",
    )
    parser.add_argument(
        "--output",
        default="./docs/timefuse_confirmation_summary.json",
    )
    args = parser.parse_args()

    with open(args.matrix_summary, "r", encoding="utf-8") as source:
        matrix_summary = json.load(source)
    with open(args.protocol, "r", encoding="utf-8") as source:
        protocol = json.load(source)
    summary = build_confirmation_summary(
        load_manifest(args.manifest),
        matrix_summary,
        protocol,
    )
    save_json_atomic(summary, args.output)
    print(
        json.dumps(
            {
                "selection": summary["selection"],
                "scope": summary["scope"],
                "criteria": summary["confirmatory_gate"]["criteria"],
                "confirmatory_gate_passed": summary[
                    "confirmatory_gate"
                ]["confirmatory_gate_passed"],
                "output": args.output,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
