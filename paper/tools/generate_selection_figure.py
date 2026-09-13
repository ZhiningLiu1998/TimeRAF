#!/usr/bin/env python3
"""Render the validation-selected correction family composition figure.

The figure shows that the family validation selects is not a global constant:
it changes with the backbone and with the dataset regime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

EXPECTED_SOURCE_SHA256 = (
    "924d5e851f7f940d6ef7b7d3503c5281f1d744a41320254b36950f40684952de"
)

FAMILIES = [
    ("historical_residual", "Residual retrieval", (0 / 255, 143 / 255, 157 / 255)),
    ("seasonal_blend", "Seasonal", (194 / 255, 133 / 255, 50 / 255)),
    ("overlap_blend", "Overlap", (72 / 255, 143 / 255, 183 / 255)),
    ("causal_bias", "Causal bias", (151 / 255, 162 / 255, 170 / 255)),
]

MODEL_ORDER = [
    "TimeXer",
    "TimeMixer",
    "PAttn",
    "iTransformer",
    "TimesNet",
    "PatchTST",
    "DLinear",
    "FreTS",
    "FEDformer",
    "Nonstationary_Transformer",
    "LightTS",
    "Informer",
    "Autoformer",
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

DATASET_ORDER = [
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "weather",
    "electricity",
    "traffic",
    "PEMS03",
    "PEMS04",
    "PEMS07",
    "PEMS08",
    "NP",
    "PJM",
    "BE",
    "FR",
    "DE",
]

DATASET_LABELS = {
    "weather": "Weath",
    "electricity": "Elec",
    "traffic": "Traff",
    "PEMS03": "PEMS03",
    "PEMS04": "PEMS04",
    "PEMS07": "PEMS07",
    "PEMS08": "PEMS08",
}

INK = (38 / 255, 47 / 255, 57 / 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "docs"
        / "publication_results"
        / "a100_composed_summary.json",
    )
    parser.add_argument("--expected-sha256", default=EXPECTED_SOURCE_SHA256)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "latex"
        / "figures"
        / "selection_adaptivity.pdf",
    )
    return parser.parse_args()


def load_cells(args: argparse.Namespace) -> list[dict]:
    payload = args.input.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != args.expected_sha256:
        raise ValueError(f"source SHA-256 mismatch: {digest}")
    cells = json.loads(payload)["cell_states"]
    if len(cells) != 585:
        raise ValueError(f"expected 585 cells, found {len(cells)}")
    known = {family for family, _, _ in FAMILIES}
    for cell in cells:
        if cell["method"] not in known:
            raise ValueError(f"unexpected family {cell['method']}")
    return cells


def compose(cells: list[dict], key: str) -> dict[str, Counter]:
    result: dict[str, Counter] = defaultdict(Counter)
    for cell in cells:
        result[cell[key]][cell["method"]] += 1
    return result


def draw_panel(axis, order, labels, counts, title) -> None:
    positions = range(len(order))
    bottoms = [0.0] * len(order)
    for family, _, color in FAMILIES:
        heights = []
        for name in order:
            total = sum(counts[name].values())
            heights.append(100.0 * counts[name][family] / total)
        axis.bar(
            positions,
            heights,
            bottom=bottoms,
            color=color,
            width=0.74,
            edgecolor="white",
            linewidth=0.6,
        )
        bottoms = [base + height for base, height in zip(bottoms, heights)]
    axis.set_xticks(list(positions))
    axis.set_xticklabels(
        [labels.get(name, name) for name in order],
        rotation=45,
        ha="right",
        fontsize=7.4,
        color=INK,
    )
    axis.set_ylim(0, 100)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.tick_params(axis="y", labelsize=7.4, colors=INK, length=2)
    axis.tick_params(axis="x", length=0)
    axis.set_ylabel("selected cells (%)", fontsize=8, color=INK)
    axis.set_title(title, fontsize=8.8, color=INK, pad=4, loc="left")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color((151 / 255, 162 / 255, 170 / 255))
        axis.spines[spine].set_linewidth(0.7)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color=(221 / 255, 226 / 255, 230 / 255), linewidth=0.6)


def main() -> None:
    args = parse_args()
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "text.usetex": False,
            "svg.hashsalt": "timeraf",
        }
    )
    cells = load_cells(args)
    by_model = compose(cells, "model")
    by_dataset = compose(cells, "dataset")

    figure, axes = plt.subplots(
        1, 2, figsize=(10.4, 2.45), gridspec_kw={"width_ratios": [13, 16]}
    )
    draw_panel(
        axes[0],
        MODEL_ORDER,
        MODEL_LABELS,
        by_model,
        "(a) By backbone (45 cells each)",
    )
    draw_panel(
        axes[1],
        DATASET_ORDER,
        DATASET_LABELS,
        by_dataset,
        "(b) By dataset (52, 39, or 13 cells each)",
    )
    axes[1].set_ylabel("")
    handles = [
        Patch(facecolor=color, edgecolor="white", label=label)
        for _, label, color in FAMILIES
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
        fontsize=8,
        handlelength=1.3,
        handleheight=0.8,
        columnspacing=1.6,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
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
        "families": [family for family, _, _ in FAMILIES],
        "by_model": {name: dict(counter) for name, counter in by_model.items()},
        "by_dataset": {name: dict(counter) for name, counter in by_dataset.items()},
    }
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "selection_adaptivity.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
