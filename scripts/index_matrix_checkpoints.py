import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ts_rag.matrix import load_manifest, save_json_atomic
from ts_rag.matrix_checkpoints import (
    build_matrix_checkpoint_catalog,
    validate_selected_checkpoint_catalog,
)


def main():
    parser = argparse.ArgumentParser(
        description="Build a reusable catalog from completed matrix checkpoints"
    )
    parser.add_argument("summary")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--checkpoint-path-root",
        help=(
            "Write checkpoint paths relative to this directory. "
            "Greenland replay catalogs must set this to their archived "
            "checkpoint root."
        ),
    )
    parser.add_argument(
        "--manifest",
        help="Limit the catalog to exactly the cells in this manifest.",
    )
    parser.add_argument(
        "--stage-root",
        help=(
            "Stage selected checkpoints below this project-relative root "
            "using hardlinks when possible."
        ),
    )
    parser.add_argument(
        "--output",
        default="./docs/timefuse_matrix_checkpoint_catalog.json",
    )
    parser.add_argument(
        "--recovery-receipt",
        help=(
            "Required when summary is a composed numerical-recovery output; "
            "binds checkpoint selection to the verified composition receipt."
        ),
    )
    parser.add_argument(
        "--recovery-protocol",
        help=(
            "Required when summary is a composed numerical-recovery output; "
            "records the full cohort, including retained first-pass "
            "checkpoints."
        ),
    )
    parser.add_argument("--hash", action="store_true")
    args = parser.parse_args()
    if args.stage_root and not args.manifest:
        parser.error("--stage-root requires --manifest")
    if args.stage_root and not args.hash:
        parser.error("--stage-root requires --hash")
    selected_cell_ids = (
        [row["id"] for row in load_manifest(args.manifest)]
        if args.manifest
        else None
    )

    catalog = build_matrix_checkpoint_catalog(
        args.summary,
        Path(args.project_root),
        compute_hashes=args.hash,
        checkpoint_path_root=args.checkpoint_path_root,
        checkpoint_stage_root=args.stage_root,
        selected_cell_ids=selected_cell_ids,
        recovery_receipt=args.recovery_receipt,
        recovery_protocol=args.recovery_protocol,
    )
    if selected_cell_ids is not None:
        validate_selected_checkpoint_catalog(
            catalog,
            selected_cell_ids,
            expected_source_revision=catalog.get("source_revision"),
            require_hashes=args.hash,
        )
    save_json_atomic(catalog, args.output)
    print(
        f"checkpoints={catalog['checkpoint_count']} "
        f"usable={catalog['usable_count']} "
        f"missing={catalog['missing_count']} "
        f"missing_selected={catalog['missing_selected_count']} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
