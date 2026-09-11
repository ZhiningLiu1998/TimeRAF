#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_manifest(protocol: dict) -> list[dict]:
    rows: list[dict] = []

    ts_rag = protocol["systems"]["ts_rag"]
    for dataset in ts_rag["datasets"]:
        rows.append(
            {
                "id": f"ts_rag/{dataset}/c512/h64",
                "method": "ts_rag",
                "dataset": dataset,
                "backbone": ts_rag["backbone"],
                "context_length": 512,
                "prediction_length": 64,
                "metrics": ts_rag["metrics"],
            }
        )

    raf = protocol["systems"]["raf"]
    for benchmark in ("benchmark_1", "benchmark_2"):
        scope = raf[benchmark]
        for dataset in scope["datasets"]:
            for context_length in scope["context_lengths"]:
                rows.append(
                    {
                        "id": (
                            f"raf/{benchmark}/{dataset}/"
                            f"c{context_length}/h{scope['prediction_length']}"
                        ),
                        "method": "raf",
                        "benchmark": benchmark,
                        "dataset": dataset,
                        "backbone": raf["backbone"],
                        "context_length": context_length,
                        "prediction_length": scope["prediction_length"],
                        "metrics": raf["metrics"],
                    }
                )

    ratd = protocol["systems"]["ratd"]
    rows.append(
        {
            "id": "ratd/electricity/c96/h168",
            "method": "ratd",
            "dataset": "electricity",
            "backbone": ratd["backbone"],
            "context_length": 96,
            "prediction_length": 168,
            "metrics": ratd["metrics"],
        }
    )

    rows.sort(
        key=lambda row: (
            row["method"],
            row.get("benchmark", ""),
            row["dataset"],
            row["context_length"],
            row["prediction_length"],
        )
    )
    expected = protocol["acceptance"]["exact_pair_count"]
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} rows, generated {len(rows)}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest IDs are not unique")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/native_retrieval_baseline_protocol.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    rows = build_manifest(protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
