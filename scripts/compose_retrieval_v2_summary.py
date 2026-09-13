#!/usr/bin/env python3
"""Compose the accepted fixed-forecast summary with the unified-portfolio result.

The accepted 585-cell summary supplies the frozen base forecast and every
comparison system. The v2 policy pass supplies the method row and its two
ablations. Both inputs are hash gated and the base metrics must agree exactly.

``timeraf_no_folds`` restates the accepted method inside the new run, so it acts
as the reproduction control. It must select the same candidate in every cell,
classify every strict win identically, and match the accepted metrics within the
project's frozen numerical consistency rule. Exact byte equality is not
required: the two runs differ only in BLAS thread count, which reorders float
reductions at the 1e-5 relative level, the same tolerance the project already
records for cross-hardware replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

POLICY_SYSTEMS = {
    "timeraf": "p_v2_folded",
    "timeraf_horizon_prior": "p_v2_horizon_gated",
    "timeraf_no_drift": "p_v1_folded",
    "timeraf_no_folds": "p_v1_reference",
}
CARRIED_SYSTEMS = (
    "base",
    "analog_future",
    "raft_adapted",
    "saraf_adapted",
    "residual_retrieval",
)


NEW_FEATURE_KEYS = ("include_level_features",)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _params_without_new_keys(params):
    """Drop parameter keys that only exist in the unified portfolio."""

    payload = dict(params)
    config = payload.get("feature_config")
    if isinstance(config, dict):
        payload["feature_config"] = {
            key: value
            for key, value in config.items()
            if key not in NEW_FEATURE_KEYS
        }
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-summary", required=True)
    parser.add_argument("--accepted-sha256", required=True)
    parser.add_argument("--v2-summary", required=True)
    parser.add_argument("--v2-sha256", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if sha256_file(args.accepted_summary) != args.accepted_sha256:
        raise SystemExit("accepted summary hash mismatch")
    if sha256_file(args.v2_summary) != args.v2_sha256:
        raise SystemExit("v2 summary hash mismatch")

    accepted = json.loads(Path(args.accepted_summary).read_text())
    v2 = json.loads(Path(args.v2_summary).read_text())
    v2_cells = {row["cell_id"]: row for row in v2["cell_states"]}
    if not v2["no_lookahead_all_passed"]:
        raise SystemExit("v2 run did not pass every no-lookahead check")

    reproduction = {
        "policy": POLICY_SYSTEMS["timeraf_no_folds"],
        "identical_predictions": 0,
        "metric_comparisons": 0,
        "max_relative_difference": 0.0,
        "tolerance": "abs <= 1e-5 + 1e-4 * max(|a|, |b|)",
        "same_selected_candidate": True,
        "same_strict_win_class": True,
    }
    cell_states = []
    for cell in accepted["cell_states"]:
        cell_id = cell["cell_id"]
        source = v2_cells[cell_id]
        base = cell["systems"]["base"]["test_metrics"]
        for name, value in source["test_baseline"].items():
            if float(value) != float(base[name]):
                raise SystemExit(f"base forecast drifted for {cell_id}/{name}")

        systems = {name: cell["systems"][name] for name in CARRIED_SYSTEMS}
        for system, policy in POLICY_SYSTEMS.items():
            outcome = source["rungs"][policy]
            if not outcome["no_lookahead"]["passed"]:
                raise SystemExit(f"no-lookahead check failed for {cell_id}")
            systems[system] = {
                "system": system,
                "policy": policy,
                "test_metrics": outcome["test_corrected"],
                "metric_gain_percent": outcome["metric_gain_percent"],
                "strict_win": bool(outcome["strict_win"]),
                "prediction_sha256": outcome["prediction_sha256"],
                "selected": outcome["selected"],
                "candidate_count": outcome["candidate_count"],
            }
        accepted_method = cell["systems"]["timeraf"]
        control = systems["timeraf_no_folds"]
        if (
            control["selected"]["method"]
            != accepted_method["validation"]["selected"]["method"]
            or _params_without_new_keys(control["selected"]["params"])
            != accepted_method["validation"]["selected"]["params"]
        ):
            raise SystemExit(
                f"reproduction control selected a different candidate: {cell_id}"
            )
        if control["strict_win"] != bool(accepted_method["strict_win"]):
            raise SystemExit(
                f"reproduction control changed the strict-win class: {cell_id}"
            )
        for name, value in control["test_metrics"].items():
            reference = float(accepted_method["test_metrics"][name])
            observed = float(value)
            if abs(observed - reference) > 1e-5 + 1e-4 * max(
                abs(observed), abs(reference)
            ):
                raise SystemExit(
                    f"reproduction control exceeded the tolerance: {cell_id}/{name}"
                )
            reproduction["metric_comparisons"] += 1
            reproduction["max_relative_difference"] = max(
                reproduction["max_relative_difference"],
                abs(observed - reference) / max(abs(reference), 1e-12),
            )
        reproduction["identical_predictions"] += int(
            control["prediction_sha256"] == accepted_method["prediction_sha256"]
        )
        cell_states.append(
            {
                "cell_id": cell_id,
                "dataset": cell["dataset"],
                "model": cell["model"],
                "pred_len": cell["pred_len"],
                "task_family": cell["task_family"],
                "state": cell["state"],
                "systems": systems,
            }
        )

    if len(cell_states) != 585:
        raise SystemExit(f"expected 585 cells, composed {len(cell_states)}")

    document = {
        "schema_version": 1,
        "generated_unix": time.time(),
        "cell_states": cell_states,
        "counts": {"completed": len(cell_states), "failed": 0},
        "expected_cells": 585,
        "expected_systems": sorted(set(CARRIED_SYSTEMS) | set(POLICY_SYSTEMS)),
        "provenance": {
            "accepted_summary_sha256": args.accepted_sha256,
            "v2_summary_sha256": args.v2_sha256,
            "v2_protocol_sha256": sha256_file(args.protocol),
            "v2_method_revision": v2["run_metadata"]["method_revision"],
            "v2_launcher_revision": v2["run_metadata"]["launcher_revision"],
            "selected_policy": POLICY_SYSTEMS["timeraf"],
            "policy_systems": POLICY_SYSTEMS,
            "base_forecasts_identical": True,
            "accepted_method_reproduction": reproduction,
            "no_lookahead_all_passed": True,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    print(f"sha256 {sha256_file(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
