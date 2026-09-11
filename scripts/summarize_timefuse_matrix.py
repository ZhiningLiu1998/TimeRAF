import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ts_rag.matrix import (
    PUBLICATION_EXPECTED_CELLS,
    build_matrix_summary,
    load_manifest,
    save_json_atomic,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize a TimeFuse matrix run")
    parser.add_argument(
        "--manifest", default="./docs/timefuse_experiment_manifest.jsonl"
    )
    parser.add_argument(
        "--output-root", default="./ts_rag_outputs/timefuse_matrix_full"
    )
    parser.add_argument("--summary")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--source-revision")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--require-publication-scope", action="store_true")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    if (
        args.require_publication_scope
        and len(manifest) != PUBLICATION_EXPECTED_CELLS
    ):
        raise ValueError(
            "Primary publication summary requires exactly "
            f"{PUBLICATION_EXPECTED_CELLS} manifest cells"
        )
    summary_path = Path(
        args.summary or Path(args.output_root) / "matrix_summary.json"
    )
    summary = build_matrix_summary(
        manifest,
        args.output_root,
        smoke=args.smoke,
        seed=args.seed,
        source_revision=args.source_revision,
    )
    save_json_atomic(summary, summary_path)

    counts = summary["counts"]
    print(
        "expected={expected} completed={completed} improved={improved} "
        "not_improved={not_improved} failed={failed} running={running} "
        "pending={pending} incomplete={incomplete}".format(
            expected=counts.get("expected", 0),
            completed=counts.get("completed", 0),
            improved=counts.get("improved", 0),
            not_improved=counts.get("not_improved", 0),
            failed=counts.get("failed", 0),
            running=counts.get("running", 0),
            pending=counts.get("pending", 0),
            incomplete=counts.get("incomplete", 0),
        )
    )
    print(f"all_completed={summary['all_completed']}")
    print(f"all_improved={summary['all_improved']}")
    print(
        "development_gate_passed="
        f"{summary['publication_gate']['development_gate_passed']}"
    )
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
