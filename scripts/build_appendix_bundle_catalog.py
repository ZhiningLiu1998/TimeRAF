import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ts_rag.appendix_rag import (
    load_appendix_manifest,
    sha256_file,
)
from ts_rag.matrix import resolve_source_revision, save_json_atomic


def _bundle(value):
    cell_id, separator, path = value.partition("=")
    if not separator or not cell_id or not path:
        raise argparse.ArgumentTypeError("--bundle must use CELL_ID=PATH")
    return cell_id, path


def _manifest_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Build a hash-verified Appendix prediction bundle catalog"
    )
    parser.add_argument(
        "--manifest",
        default="./docs/timefuse_appendix_experiment_manifest.jsonl",
    )
    parser.add_argument("--bundle", action="append", type=_bundle, default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision")
    args = parser.parse_args()
    args.source_revision = args.source_revision or resolve_source_revision()

    rows = load_appendix_manifest(args.manifest)
    cells = {row["id"]: row for row in rows}
    bundles = {}
    for cell_id, raw_path in args.bundle:
        if cell_id not in cells:
            parser.error(f"Unknown appendix cell: {cell_id}")
        if cell_id in bundles:
            parser.error(f"Duplicate appendix cell: {cell_id}")
        path = Path(raw_path)
        if not path.is_file():
            parser.error(f"Prediction bundle does not exist: {path}")
        bundles[cell_id] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "baseline": cells[cell_id]["baseline"],
            "dataset": cells[cell_id]["dataset"],
            "task_family": cells[cell_id]["task_family"],
        }
    evaluable_ids = {
        row["id"] for row in rows if row["paper_status"]["evaluable"]
    }
    payload = {
        "schema_version": 1,
        "source_revision": args.source_revision,
        "manifest": args.manifest,
        "manifest_sha256": _manifest_sha256(args.manifest),
        "counts": {
            "reported": len(rows),
            "evaluable": len(evaluable_ids),
            "cataloged": len(bundles),
            "missing_evaluable": len(evaluable_ids - set(bundles)),
        },
        "bundles": bundles,
    }
    save_json_atomic(payload, args.output)
    print(json.dumps(payload["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
