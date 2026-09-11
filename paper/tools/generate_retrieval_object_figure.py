#!/usr/bin/env python3
"""Render how the better retrieval object depends on base-forecaster accuracy.

For every frozen backbone and task family we compute the median worst-metric
paired gain of Residual-kNN and of Analog-kNN over the identical base forecast,
and plot their difference. Positive means retrieving the model's own error beats
retrieving an analogous historical future.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXPECTED_SOURCE_SHA256 = (
    "a9cdf53dc50ad7b3200da49c8e9bc16db8dd48af41b3fed33513033449c44c2a"
)

FAMILIES = [
    ("long_term", "Long-term", (0 / 255, 143 / 255, 157 / 255), "o"),
    ("pems", "PEMS", (194 / 255, 133 / 255, 50 / 255), "s"),
    ("epf", "EPF", (119 / 255, 87 / 255, 174 / 255), "^"),
]

MODEL_LABELS = {
    "Nonstationary_Transformer": "NSTrans",
    "iTransformer": "iTrans",
    "FEDformer": "FEDf",
    "Autoformer": "AutoF",
    "Informer": "InF",
    "TimeMixer": "TMixer",
    "TimeXer": "TXer",
    "PatchTST": "PTST",
    "DLinear": "DLin",
    "TimesNet": "TNet",
    "LightTS": "LTS",
}

INK = (38 / 255, 47 / 255, 57 / 255)
GRAY = (151 / 255, 162 / 255, 170 / 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "docs"
        / "publication_results"
        / "retrieval_baseline_summary.json",
    )
    parser.add_argument("--expected-sha256", default=EXPECTED_SOURCE_SHA256)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "latex"
        / "figures"
        / "retrieval_object_vs_strength.pdf",
    )
    return parser.parse_args()


def spearman(first: list[float], second: list[float]) -> float:
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        result = [0.0] * len(values)
        for position, index in enumerate(order):
            result[index] = position + 1
        return result

    left, right = ranks(first), ranks(second)
    mean_left, mean_right = fmean(left), fmean(right)
    numerator = sum(
        (left[i] - mean_left) * (right[i] - mean_right) for i in range(len(left))
    )
    denominator = (
        sum((value - mean_left) ** 2 for value in left)
        * sum((value - mean_right) ** 2 for value in right)
    ) ** 0.5
    return numerator / denominator


def main() -> None:
    args = parse_args()
    payload = args.input.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != args.expected_sha256:
        raise ValueError(f"source SHA-256 mismatch: {digest}")
    cells = json.loads(payload)["cell_states"]
    if len(cells) != 585:
        raise ValueError(f"expected 585 cells, found {len(cells)}")

    base_error: dict[str, list[float]] = defaultdict(list)
    worst: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for cell in cells:
        family = cell["cell_id"].split("/", 1)[0]
        model = cell["model"]
        if family == "long_term":
            base_error[model].append(cell["systems"]["base"]["test_metrics"]["mse"])
        for system in ("analog_future", "residual_retrieval"):
            gains = cell["systems"][system]["metric_gain_percent"]
            worst[(family, model, system)].append(min(gains.values()))

    strength = {model: fmean(values) for model, values in base_error.items()}
    order = sorted(strength, key=strength.get)

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "text.usetex": False,
        }
    )
    figure, axis = plt.subplots(figsize=(7.2, 2.15))
    positions = list(range(len(order)))
    receipt_series = {}
    for family, label, color, marker in FAMILIES:
        values = [
            median(worst[(family, model, "residual_retrieval")])
            - median(worst[(family, model, "analog_future")])
            for model in order
        ]
        rho = spearman([strength[model] for model in order], values)
        receipt_series[family] = {
            "advantage_percentage_points": dict(zip(order, values)),
            "spearman_rho_vs_base_mse": rho,
        }
        axis.plot(
            positions,
            values,
            marker=marker,
            markersize=3.6,
            linewidth=1.1,
            color=color,
            label=f"{label} ($\\rho$ = {rho:+.2f})",
        )
    axis.axhline(0.0, color=GRAY, linewidth=0.9, linestyle=(0, (4, 3)))
    axis.set_yscale("symlog", linthresh=1.0, linscale=0.9)
    axis.set_ylim(-60, 40)
    axis.set_yticks([-20, -10, -1, 0, 1, 5])
    axis.set_yticklabels(["-20", "-10", "-1", "0", "+1", "+5"])
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [MODEL_LABELS.get(model, model) for model in order],
        rotation=40,
        ha="right",
        fontsize=7.6,
        color=INK,
    )
    axis.set_xlim(-0.55, len(order) - 0.45)
    axis.set_ylabel(
        "residual $-$ analog\nmedian gain (pp)", fontsize=7.8, color=INK
    )
    axis.set_xlabel(
        "frozen backbone, ordered from most to least accurate "
        "(long-term base MSE)",
        fontsize=7.8,
        color=INK,
    )
    axis.tick_params(labelsize=7.6, colors=INK, length=2)
    axis.text(
        0.012,
        0.93,
        "retrieving the model's own error is better",
        transform=axis.transAxes,
        fontsize=7.2,
        color=(0 / 255, 110 / 255, 120 / 255),
        va="top",
    )
    axis.text(
        0.012,
        0.07,
        "copying an analogous future is better",
        transform=axis.transAxes,
        fontsize=7.2,
        color=(160 / 255, 105 / 255, 40 / 255),
        va="bottom",
    )
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(GRAY)
        axis.spines[spine].set_linewidth(0.7)
    axis.legend(
        loc="center right",
        fontsize=7.4,
        frameon=False,
        handlelength=1.6,
        borderaxespad=0.2,
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        args.output,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
        metadata={"CreationDate": None},
    )
    plt.close(figure)

    receipt = {
        "schema_version": 1,
        "source_sha256": args.expected_sha256,
        "statistic": (
            "median over cells of the smallest paired relative gain across the "
            "metrics reported for that cell"
        ),
        "backbone_order": order,
        "long_term_base_mse": strength,
        "series": receipt_series,
    }
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "retrieval_object_vs_strength.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
