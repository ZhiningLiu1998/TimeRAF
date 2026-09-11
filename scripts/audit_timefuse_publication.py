import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.greenland_common import METHOD_REVISION
from ts_rag.appendix_matrix import APPENDIX_METHOD_REVISION
from ts_rag.matrix import save_json_atomic
from ts_rag.publication import audit_timefuse_publication


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fail-closed audit of all TimeFuse publication evidence"
    )
    parser.add_argument("--project-root", required=True)
    parser.add_argument(
        "--primary-manifest",
        default="docs/timefuse_experiment_manifest.jsonl",
    )
    parser.add_argument("--primary-summary", required=True)
    parser.add_argument("--recovery-receipt")
    parser.add_argument("--recovery-protocol", required=True)
    parser.add_argument(
        "--confirmation-protocol",
        default="docs/timefuse_confirmation_protocol.json",
    )
    parser.add_argument("--confirmation-summary", required=True)
    parser.add_argument(
        "--replay-manifest",
        default="docs/timefuse_checkpoint_replay_manifest.jsonl",
    )
    parser.add_argument("--replay-receipt", required=True)
    parser.add_argument(
        "--appendix-manifest",
        default="docs/timefuse_appendix_experiment_manifest.jsonl",
    )
    parser.add_argument("--appendix-catalog", required=True)
    parser.add_argument("--appendix-summary", required=True)
    parser.add_argument("--method-revision", default=METHOD_REVISION)
    parser.add_argument(
        "--appendix-method-revision",
        default=APPENDIX_METHOD_REVISION,
    )
    parser.add_argument("--appendix-launcher-revision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()

    def project_path(value):
        path = Path(value)
        return path if path.is_absolute() else project_root / path

    try:
        report = audit_timefuse_publication(
            project_root=project_root,
            primary_manifest=project_path(args.primary_manifest),
            primary_summary=project_path(args.primary_summary),
            recovery_receipt=(
                project_path(args.recovery_receipt)
                if args.recovery_receipt
                else None
            ),
            recovery_protocol=project_path(args.recovery_protocol),
            confirmation_protocol=project_path(
                args.confirmation_protocol
            ),
            confirmation_summary=project_path(args.confirmation_summary),
            replay_manifest=project_path(args.replay_manifest),
            replay_receipt=project_path(args.replay_receipt),
            appendix_manifest=project_path(args.appendix_manifest),
            appendix_catalog=project_path(args.appendix_catalog),
            appendix_summary=project_path(args.appendix_summary),
            method_revision=args.method_revision,
            appendix_method_revision=args.appendix_method_revision,
            appendix_launcher_revision=args.appendix_launcher_revision,
        )
    except Exception as error:
        report = {
            "schema_version": 1,
            "generated_unix": time.time(),
            "project_root": str(project_root),
            "method_revision": args.method_revision,
            "appendix_method_revision": args.appendix_method_revision,
            "appendix_launcher_revision": args.appendix_launcher_revision,
            "publication_ready": False,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
        save_json_atomic(report, args.output)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    save_json_atomic(report, args.output)
    print(
        json.dumps(
            {
                "criteria": report["criteria"],
                "publication_ready": report["publication_ready"],
                "output": args.output,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["publication_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
