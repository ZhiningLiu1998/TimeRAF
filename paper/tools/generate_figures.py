#!/usr/bin/env python3
"""Generate deterministic vector figures for the TimeRAF manuscript."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence


TEAL = (0 / 255, 143 / 255, 157 / 255)
BLUE = (72 / 255, 143 / 255, 183 / 255)
GREEN = (66 / 255, 145 / 255, 123 / 255)
GOLD = (194 / 255, 133 / 255, 50 / 255)
RED = (194 / 255, 91 / 255, 70 / 255)
PURPLE = (119 / 255, 87 / 255, 174 / 255)
INK = (38 / 255, 47 / 255, 57 / 255)
GRAY = (91 / 255, 105 / 255, 116 / 255)
MID_GRAY = (151 / 255, 162 / 255, 170 / 255)
LIGHT_GRAY = (221 / 255, 226 / 255, 230 / 255)
CANVAS = (246 / 255, 248 / 255, 249 / 255)
PALE_BLUE = (226 / 255, 239 / 255, 248 / 255)
PALE_GREEN = (222 / 255, 241 / 255, 235 / 255)
PALE_GOLD = (251 / 255, 239 / 255, 216 / 255)
PALE_RED = (249 / 255, 228 / 255, 223 / 255)
PALE_PURPLE = (238 / 255, 231 / 255, 248 / 255)
HEADER_BLUE = (166 / 255, 207 / 255, 232 / 255)
HEADER_GREEN = (155 / 255, 216 / 255, 202 / 255)
HEADER_GOLD = (241 / 255, 186 / 255, 111 / 255)
HEADER_RED = (238 / 255, 160 / 255, 137 / 255)
HEADER_PURPLE = (199 / 255, 174 / 255, 232 / 255)
WHITE = (1.0, 1.0, 1.0)
REAL_CASE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "real_retrieval_case.json"
)


def _number(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def load_real_case(path: Path = REAL_CASE_PATH) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing real retrieval case {path}; run "
            "paper/tools/select_real_retrieval_case.py first"
        )
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported real-case schema in {path}")
    required = (
        "query_context_normalized",
        "neighbor_contexts_normalized",
        "neighbor_futures",
        "neighbor_residuals",
        "retrieved_correction",
        "query_truth",
        "query_base_forecast",
        "query_corrected_forecast",
    )
    missing = [key for key in required if key not in payload.get("series", {})]
    if missing:
        raise KeyError(f"Real retrieval case is missing: {', '.join(missing)}")
    for key in required:
        series_group = payload["series"][key]
        series_values = (
            series_group
            if not series_group or isinstance(series_group[0], (int, float))
            else [value for series in series_group for value in series]
        )
        if not series_values or not all(math.isfinite(value) for value in series_values):
            raise ValueError(f"Real retrieval case has invalid series: {key}")
    return payload


def compact_series(
    values: Sequence[float],
    max_points: int,
) -> tuple[float, ...]:
    values = tuple(float(value) for value in values)
    if len(values) <= max_points:
        return values
    compact = []
    for index in range(max_points):
        start = round(index * len(values) / max_points)
        end = round((index + 1) * len(values) / max_points)
        chunk = values[start:max(end, start + 1)]
        compact.append(sum(chunk) / len(chunk))
    return tuple(compact)


def shared_range(
    series: Sequence[Sequence[float]],
    *,
    padding: float = 0.08,
) -> tuple[float, float]:
    values = [float(value) for row in series for value in row]
    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum
    if span == 0:
        span = max(abs(maximum), 1.0)
    return minimum - padding * span, maximum + padding * span


class PdfCanvas:
    """Minimal PDF 1.4 canvas using built-in Helvetica fonts."""

    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height
        self.commands: list[str] = [
            "1 J",
            "1 j",
            f"0 0 {_number(width)} {_number(height)} re",
            "1 1 1 rg",
            "f",
        ]

    def set_fill(self, color: tuple[float, float, float]) -> None:
        self.commands.append("{} {} {} rg".format(*(_number(v) for v in color)))

    def set_stroke(self, color: tuple[float, float, float]) -> None:
        self.commands.append("{} {} {} RG".format(*(_number(v) for v in color)))

    def set_line_width(self, width: float) -> None:
        self.commands.append(f"{_number(width)} w")

    def set_dash(self, pattern: Sequence[float] = ()) -> None:
        values = " ".join(_number(value) for value in pattern)
        self.commands.append(f"[{values}] 0 d")

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: tuple[float, float, float] | None = None,
        stroke: tuple[float, float, float] | None = None,
        line_width: float = 0.8,
    ) -> None:
        if fill is not None:
            self.set_fill(fill)
        if stroke is not None:
            self.set_stroke(stroke)
            self.set_line_width(line_width)
        self.commands.append(
            f"{_number(x)} {_number(y)} {_number(width)} {_number(height)} re"
        )
        self.commands.append("B" if fill is not None and stroke is not None else
                             "f" if fill is not None else "S")

    def rounded_rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        radius: float = 6.0,
        *,
        fill: tuple[float, float, float] | None = None,
        stroke: tuple[float, float, float] | None = None,
        line_width: float = 0.8,
    ) -> None:
        radius = min(radius, width / 2, height / 2)
        kappa = 0.5522847498
        k = radius * kappa
        if fill is not None:
            self.set_fill(fill)
        if stroke is not None:
            self.set_stroke(stroke)
            self.set_line_width(line_width)
        commands = [
            f"{_number(x + radius)} {_number(y)} m",
            f"{_number(x + width - radius)} {_number(y)} l",
            (
                f"{_number(x + width - radius + k)} {_number(y)} "
                f"{_number(x + width)} {_number(y + radius - k)} "
                f"{_number(x + width)} {_number(y + radius)} c"
            ),
            f"{_number(x + width)} {_number(y + height - radius)} l",
            (
                f"{_number(x + width)} {_number(y + height - radius + k)} "
                f"{_number(x + width - radius + k)} {_number(y + height)} "
                f"{_number(x + width - radius)} {_number(y + height)} c"
            ),
            f"{_number(x + radius)} {_number(y + height)} l",
            (
                f"{_number(x + radius - k)} {_number(y + height)} "
                f"{_number(x)} {_number(y + height - radius + k)} "
                f"{_number(x)} {_number(y + height - radius)} c"
            ),
            f"{_number(x)} {_number(y + radius)} l",
            (
                f"{_number(x)} {_number(y + radius - k)} "
                f"{_number(x + radius - k)} {_number(y)} "
                f"{_number(x + radius)} {_number(y)} c"
            ),
            "h",
            (
                "B" if fill is not None and stroke is not None else
                "f" if fill is not None else "S"
            ),
        ]
        self.commands.extend(commands)

    def polygon(
        self,
        points: Sequence[tuple[float, float]],
        *,
        fill: tuple[float, float, float] | None = None,
        stroke: tuple[float, float, float] | None = None,
        line_width: float = 0.8,
    ) -> None:
        if not points:
            return
        if fill is not None:
            self.set_fill(fill)
        if stroke is not None:
            self.set_stroke(stroke)
            self.set_line_width(line_width)
        x0, y0 = points[0]
        commands = [f"{_number(x0)} {_number(y0)} m"]
        commands.extend(
            f"{_number(x)} {_number(y)} l" for x, y in points[1:]
        )
        commands.append("h")
        commands.append(
            "B" if fill is not None and stroke is not None else
            "f" if fill is not None else "S"
        )
        self.commands.extend(commands)

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: tuple[float, float, float] = INK,
        width: float = 0.8,
        dash: Sequence[float] = (),
    ) -> None:
        self.set_stroke(color)
        self.set_line_width(width)
        self.set_dash(dash)
        self.commands.append(
            f"{_number(x1)} {_number(y1)} m {_number(x2)} {_number(y2)} l S"
        )
        self.set_dash()

    def polyline(
        self,
        points: Sequence[tuple[float, float]],
        *,
        color: tuple[float, float, float] = INK,
        width: float = 1.0,
        dash: Sequence[float] = (),
    ) -> None:
        if len(points) < 2:
            return
        self.set_stroke(color)
        self.set_line_width(width)
        self.set_dash(dash)
        x0, y0 = points[0]
        commands = [f"{_number(x0)} {_number(y0)} m"]
        commands.extend(
            f"{_number(x)} {_number(y)} l" for x, y in points[1:]
        )
        commands.append("S")
        self.commands.extend(commands)
        self.set_dash()

    def circle(
        self,
        x: float,
        y: float,
        radius: float,
        *,
        fill: tuple[float, float, float] | None = None,
        stroke: tuple[float, float, float] | None = None,
        line_width: float = 0.8,
    ) -> None:
        kappa = 0.5522847498
        k = radius * kappa
        if fill is not None:
            self.set_fill(fill)
        if stroke is not None:
            self.set_stroke(stroke)
            self.set_line_width(line_width)
        commands = [
            f"{_number(x + radius)} {_number(y)} m",
            (
                f"{_number(x + radius)} {_number(y + k)} "
                f"{_number(x + k)} {_number(y + radius)} "
                f"{_number(x)} {_number(y + radius)} c"
            ),
            (
                f"{_number(x - k)} {_number(y + radius)} "
                f"{_number(x - radius)} {_number(y + k)} "
                f"{_number(x - radius)} {_number(y)} c"
            ),
            (
                f"{_number(x - radius)} {_number(y - k)} "
                f"{_number(x - k)} {_number(y - radius)} "
                f"{_number(x)} {_number(y - radius)} c"
            ),
            (
                f"{_number(x + k)} {_number(y - radius)} "
                f"{_number(x + radius)} {_number(y - k)} "
                f"{_number(x + radius)} {_number(y)} c"
            ),
            "h",
            (
                "B" if fill is not None and stroke is not None else
                "f" if fill is not None else "S"
            ),
        ]
        self.commands.extend(commands)

    def text(
        self,
        x: float,
        y: float,
        text: str,
        *,
        size: float = 9.0,
        color: tuple[float, float, float] = INK,
        bold: bool = False,
        align: str = "left",
    ) -> None:
        width = size * 0.52 * len(text)
        if align == "center":
            x -= width / 2
        elif align == "right":
            x -= width
        font = "F2" if bold else "F1"
        escaped = _escape_pdf_text(text)
        self.set_fill(color)
        self.commands.append(
            f"BT /{font} {_number(size)} Tf "
            f"{_number(x)} {_number(y)} Td ({escaped}) Tj ET"
        )

    def save(self, path: Path) -> None:
        content = ("\n".join(self.commands) + "\n").encode("ascii")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                b"<< /Type /Page /Parent 2 0 R "
                + f"/MediaBox [0 0 {_number(self.width)} {_number(self.height)}] ".encode()
                + b"/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> "
                + b"/Contents 4 0 R >>"
            ),
            (
                f"<< /Length {len(content)} >>\nstream\n".encode("ascii")
                + content
                + b"endstream"
            ),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        ]

        output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for index, obj in enumerate(objects, start=1):
            offsets.append(len(output))
            output.extend(f"{index} 0 obj\n".encode("ascii"))
            output.extend(obj)
            output.extend(b"\nendobj\n")

        xref_offset = len(output)
        output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        output.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        output.extend(
            (
                f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n"
            ).encode("ascii")
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(output)


def arrow(
    canvas: PdfCanvas,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: tuple[float, float, float] = GRAY,
    width: float = 1.4,
    head: float = 5.0,
) -> None:
    angle = math.atan2(y2 - y1, x2 - x1)
    end_x = x2 - head * 0.7 * math.cos(angle)
    end_y = y2 - head * 0.7 * math.sin(angle)
    canvas.line(x1, y1, end_x, end_y, color=color, width=width)
    left = (
        x2 - head * math.cos(angle - math.pi / 6),
        y2 - head * math.sin(angle - math.pi / 6),
    )
    right = (
        x2 - head * math.cos(angle + math.pi / 6),
        y2 - head * math.sin(angle + math.pi / 6),
    )
    canvas.polygon([(x2, y2), left, right], fill=color)


def draw_series(
    canvas: PdfCanvas,
    x: float,
    y: float,
    width: float,
    height: float,
    values: Sequence[float],
    *,
    color: tuple[float, float, float],
    line_width: float = 1.5,
    dash: Sequence[float] = (),
) -> None:
    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum or 1.0
    points = [
        (
            x + width * index / (len(values) - 1),
            y + height * (value - minimum) / span,
        )
        for index, value in enumerate(values)
    ]
    canvas.polyline(points, color=color, width=line_width, dash=dash)


def draw_series_scaled(
    canvas: PdfCanvas,
    x: float,
    y: float,
    width: float,
    height: float,
    values: Sequence[float],
    *,
    minimum: float,
    maximum: float,
    color: tuple[float, float, float],
    line_width: float = 1.5,
    dash: Sequence[float] = (),
) -> None:
    """Draw comparable traces against a shared vertical scale."""
    span = maximum - minimum or 1.0
    points = [
        (
            x + width * index / (len(values) - 1),
            y + height * (value - minimum) / span,
        )
        for index, value in enumerate(values)
    ]
    canvas.polyline(points, color=color, width=line_width, dash=dash)


def pill(
    canvas: PdfCanvas,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    *,
    fill: tuple[float, float, float] = LIGHT_GRAY,
    stroke: tuple[float, float, float] = INK,
    color: tuple[float, float, float] = INK,
    size: float = 9.0,
) -> None:
    canvas.rounded_rect(
        x, y, width, height, height / 2,
        fill=fill, stroke=stroke, line_width=0.8,
    )
    canvas.text(
        x + width / 2,
        y + height / 2 - size * 0.34,
        label,
        size=size,
        color=color,
        bold=True,
        align="center",
    )


def card(
    canvas: PdfCanvas,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    *,
    fill: tuple[float, float, float],
    header: tuple[float, float, float],
    icon: str | None = None,
    title_size: float = 10.5,
    header_height: float = 27.0,
) -> None:
    canvas.rounded_rect(
        x, y, width, height, 7,
        fill=fill, stroke=INK, line_width=0.85,
    )
    canvas.rounded_rect(
        x, y + height - header_height, width, header_height, 7,
        fill=header,
    )
    canvas.rect(
        x,
        y + height - header_height,
        width,
        max(1.0, header_height - 7),
        fill=header,
    )
    canvas.line(
        x,
        y + height - header_height,
        x + width,
        y + height - header_height,
        color=INK,
        width=0.75,
    )
    title_x = x + 11
    if icon is not None:
        canvas.circle(
            x + 15,
            y + height - header_height / 2,
            9,
            fill=WHITE,
            stroke=INK,
            line_width=0.65,
        )
        canvas.text(
            x + 15,
            y + height - header_height / 2 - 3.2,
            icon,
            size=9.2,
            color=TEAL,
            bold=True,
            align="center",
        )
        title_x = x + 29
    canvas.text(
        title_x,
        y + height - header_height / 2 - title_size * 0.34,
        title,
        size=title_size,
        color=INK,
        bold=True,
    )
    canvas.rounded_rect(
        x, y, width, height, 7,
        stroke=INK, line_width=0.85,
    )


def section_frame(
    canvas: PdfCanvas,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    label_width: float,
) -> None:
    canvas.rounded_rect(
        x, y, width, height, 10,
        fill=WHITE, stroke=INK, line_width=0.9,
    )
    pill(
        canvas,
        x + (width - label_width) / 2,
        y + height - 12,
        label_width,
        24,
        label,
        fill=(238 / 255, 241 / 255, 243 / 255),
        size=9.2,
    )


def panel(
    canvas: PdfCanvas,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    number: int,
    *,
    fill: tuple[float, float, float],
    accent: tuple[float, float, float],
) -> None:
    canvas.rect(x, y, width, height, fill=fill, stroke=MID_GRAY, line_width=0.75)
    canvas.rect(x, y + height - 28, width, 28, fill=accent)
    canvas.circle(x + 14, y + height - 14, 8, fill=WHITE)
    canvas.text(
        x + 14,
        y + height - 17,
        str(number),
        size=9,
        color=accent,
        bold=True,
        align="center",
    )
    canvas.text(x + 27, y + height - 18, title, size=11, color=WHITE, bold=True)


def stadium_traffic_case() -> dict[str, tuple[float, ...]]:
    """Return the shared stadium-release example used in Figures 1 and 2."""
    # Illustrative five-minute traffic counts around a stadium event. The
    # ordinary evening profile falls after rush hour; pickup traffic builds
    # just before the event ends, then departing attendees create a surge.
    query_regular = (
        58, 61, 64, 63, 60, 56, 53, 51, 50,
        48, 47, 46, 45, 43, 40, 38, 36, 34,
    )
    query_cue = (0,) * 14 + (1, 4, 8, 13)
    query_context = tuple(
        regular + cue for regular, cue in zip(query_regular, query_cue)
    )
    query_base = (
        45, 43, 41, 39, 37, 36, 35, 34, 33, 32, 31, 30,
    )
    event = (
        2, 8, 22, 40, 48, 43, 32, 20, 11, 5, 2, 0,
    )
    query_truth = tuple(
        base + correction for base, correction in zip(query_base, event)
    )

    # The nearest raw window is an ordinary evening with no stadium release.
    whole_context = (
        63, 66, 69, 68, 65, 61, 58, 56, 55,
        53, 52, 51, 50, 48, 45, 43, 41, 39,
    )
    whole_future = (
        38, 37, 36, 35, 34, 34, 35, 37, 40, 42, 41, 39,
    )

    # An earlier event night has a different traffic level but the same
    # post-event underprediction by the trained model.
    failure_context = (
        72, 74, 73, 70, 66, 63, 61, 60, 59,
        57, 55, 53, 51, 49, 48, 49, 52, 57,
    )
    failure_base = (
        55, 53, 51, 49, 47, 45, 44, 43, 42, 41, 40, 39,
    )
    failure_truth = tuple(
        base + correction for base, correction in zip(failure_base, event)
    )
    return {
        "query_context": query_context,
        "query_base": query_base,
        "query_truth": query_truth,
        "event": event,
        "whole_context": whole_context,
        "whole_future": whole_future,
        "failure_context": failure_context,
        "failure_base": failure_base,
        "failure_truth": failure_truth,
    }


def grid_outage_case() -> dict[str, tuple[float, ...]]:
    """Return a rare feeder-outage and staged-restoration example."""
    # Two failed reclosures make the developing fault visible before the
    # forecast origin. The future then contains an outage plateau, staged
    # restoration, and a cold-load-pickup overshoot.
    query_context = (
        70, 78, 84, 78, 70, 64, 70, 78, 84,
        78, 70, 64, 70, 66, 88, 61, 70, 54,
    )
    periodic_context = (
        70, 78, 84, 78, 70, 64, 70, 78, 84,
        78, 70, 64, 70, 78, 84, 78, 70, 64,
    )
    query_base = (
        70, 78, 84, 78, 70, 64, 70, 78, 84, 78, 70, 64,
    )
    event = (
        -18, -38, -40, -40, -28, -27,
        -14, -13, 8, 18, 9, 3,
    )
    query_truth = tuple(
        base + correction for base, correction in zip(query_base, event)
    )

    # A prior storm event has the same fault and restoration sequence under a
    # lower load level. Copying its future transfers that obsolete level.
    failure_context = (
        50, 58, 64, 58, 50, 44, 50, 58, 64,
        58, 50, 44, 50, 46, 68, 41, 50, 34,
    )
    failure_base = (
        50, 58, 64, 58, 50, 44, 50, 58, 64, 58, 50, 44,
    )
    failure_truth = tuple(
        base + correction for base, correction in zip(failure_base, event)
    )
    return {
        "query_context": query_context,
        "periodic_context": periodic_context,
        "query_base": query_base,
        "query_truth": query_truth,
        "event": event,
        "whole_context": failure_context,
        "whole_future": failure_truth,
        "failure_context": failure_context,
        "failure_base": failure_base,
        "failure_truth": failure_truth,
    }


def _motivation_figure(
    path: Path,
    *,
    case: dict[str, tuple[float, ...]],
    labels: dict[str, str],
    cue_length: int,
    validate_stadium_distances: bool = False,
    legend_outside: bool = False,
    decompose_event: bool = False,
    show_residual_steps: bool = False,
) -> None:
    """Contrast whole-window transfer with model-relative error memory."""
    query_context = case["query_context"]
    query_base = case["query_base"]
    query_truth = case["query_truth"]
    event = case["event"]
    whole_context = case["whole_context"]
    whole_future = case["whole_future"]
    failure_context = case["failure_context"]
    failure_base = case["failure_base"]
    failure_truth = case["failure_truth"]

    def anchored(
        context: Sequence[float],
        future: Sequence[float],
    ) -> tuple[float, ...]:
        """Include the final observation so every future trace is continuous."""
        return (float(context[-1]), *(float(value) for value in future))

    query_base_trace = anchored(query_context, query_base)
    query_truth_trace = anchored(query_context, query_truth)
    whole_future_trace = anchored(whole_context, whole_future)
    whole_copy_trace = anchored(query_context, whole_future)
    failure_base_trace = anchored(failure_context, failure_base)
    failure_truth_trace = anchored(failure_context, failure_truth)
    residual_trace = (0.0, *event)
    corrected_trace = tuple(
        base + residual
        for base, residual in zip(query_base_trace, residual_trace)
    )
    if corrected_trace != query_truth_trace:
        raise ValueError("Illustrative correction must recover the query truth")

    if validate_stadium_distances:
        routine_distance = sum(
            (query - analog) ** 2
            for query, analog in zip(query_context[:14], whole_context[:14])
        )
        cue_distance = sum(
            (query - analog) ** 2
            for query, analog in zip(query_context[14:], whole_context[14:])
        )
        ordinary_distance = routine_distance + cue_distance
        earlier_event_distance = sum(
            (query - earlier) ** 2
            for query, earlier in zip(query_context, failure_context)
        )
        if (
            routine_distance <= cue_distance
            or ordinary_distance >= earlier_event_distance
        ):
            raise ValueError(
                "Illustrative retrieval distances contradict the labels"
            )

    canvas = PdfCanvas(720, 220)
    time_split = 17 / 29
    failure_fill = (252 / 255, 247 / 255, 246 / 255)
    success_fill = (245 / 255, 250 / 255, 248 / 255)
    future_fill = (246 / 255, 247 / 255, 248 / 255)
    plot_border = (211 / 255, 216 / 255, 220 / 255)

    canvas.rect(220, 113, 490, 99, fill=failure_fill)
    canvas.rect(220, 8, 490, 97, fill=success_fill)
    canvas.line(210, 8, 210, 212, color=LIGHT_GRAY, width=0.8)

    def draw_window(
        x: float,
        y: float,
        width: float,
        height: float,
        contexts: Sequence[
            tuple[
                Sequence[float],
                tuple[float, float, float],
                float,
                Sequence[float],
            ]
        ],
        futures: Sequence[
            tuple[
                Sequence[float],
                tuple[float, float, float],
                float,
                Sequence[float],
            ]
        ],
        *,
        minimum: float,
        maximum: float,
        context_fraction: float = time_split,
    ) -> None:
        context_width = width * context_fraction
        future_width = width - context_width
        canvas.rect(x, y, width, height, fill=WHITE,
                    stroke=plot_border, line_width=0.7)
        canvas.rect(x + context_width, y, future_width, height,
                    fill=future_fill)
        canvas.line(
            x + context_width,
            y,
            x + context_width,
            y + height,
            color=MID_GRAY,
            width=0.8,
            dash=(2, 2),
        )
        for values, color, line_width, dash in contexts:
            draw_series_scaled(
                canvas,
                x,
                y,
                context_width,
                height,
                values,
                minimum=minimum,
                maximum=maximum,
                color=color,
                line_width=line_width,
                dash=dash,
            )
        for values, color, line_width, dash in futures:
            draw_series_scaled(
                canvas,
                x + context_width,
                y,
                future_width,
                height,
                values,
                minimum=minimum,
                maximum=maximum,
                color=color,
                line_width=line_width,
                dash=dash,
            )

    def draw_cross(x: float, y: float) -> None:
        canvas.circle(x, y, 8.5, fill=WHITE, stroke=RED, line_width=1.0)
        canvas.line(x - 3.2, y - 3.2, x + 3.2, y + 3.2,
                    color=RED, width=1.6)
        canvas.line(x - 3.2, y + 3.2, x + 3.2, y - 3.2,
                    color=RED, width=1.6)

    def draw_check(x: float, y: float) -> None:
        canvas.circle(x, y, 8.5, fill=WHITE, stroke=GREEN, line_width=1.0)
        canvas.line(x - 3.7, y, x - 0.8, y - 3.0,
                    color=GREEN, width=1.6)
        canvas.line(x - 0.8, y - 3.0, x + 4.2, y + 3.6,
                    color=GREEN, width=1.6)

    query_min, query_max = shared_range(
        (query_context, query_base_trace, query_truth_trace),
        padding=0.12,
    )
    comparison_min, comparison_max = shared_range(
        (
            query_context,
            query_base_trace,
            query_truth_trace,
            failure_context,
            failure_base_trace,
            failure_truth_trace,
        ),
        padding=0.10,
    )
    cue_start = len(query_context) - cue_length

    canvas.text(18, 198, labels["query_title"], size=11.2,
                color=INK, bold=True)
    if decompose_event:
        periodic_context = case.get("periodic_context")
        if periodic_context is None:
            raise ValueError("Event decomposition requires periodic_context")
        context_event = tuple(
            observed - periodic
            for observed, periodic in zip(query_context, periodic_context)
        )
        periodic_trace = (*periodic_context, *query_base)
        event_pattern = (*context_event, *event)
        observed_trace = (*query_context, *query_truth)
        reconstructed = tuple(
            periodic + rare
            for periodic, rare in zip(periodic_trace, event_pattern)
        )
        if reconstructed != observed_trace:
            raise ValueError("Periodic baseline and event must reconstruct load")

        load_min, load_max = shared_range(
            (periodic_trace, observed_trace),
            padding=0.08,
        )
        periodic_min, periodic_max = shared_range(
            (periodic_trace,),
            padding=0.12,
        )
        event_extent = max(abs(value) for value in event_pattern) * 1.12
        split_x = 18 + 180 * time_split

        def draw_component(
            y: float,
            height: float,
            traces: Sequence[
                tuple[
                    Sequence[float],
                    tuple[float, float, float],
                    float,
                    Sequence[float],
                ]
            ],
            *,
            minimum: float,
            maximum: float,
            show_zero: bool = False,
        ) -> None:
            canvas.rect(18, y, 180, height, fill=WHITE,
                        stroke=plot_border, line_width=0.7)
            canvas.rect(split_x, y, 180 * (1 - time_split), height,
                        fill=future_fill)
            canvas.line(split_x, y, split_x, y + height,
                        color=MID_GRAY, width=0.7, dash=(2, 2))
            if show_zero:
                zero_y = y + height * (0 - minimum) / (maximum - minimum)
                canvas.line(18, zero_y, 198, zero_y,
                            color=LIGHT_GRAY, width=0.7, dash=(2, 2))
            for values, color, width, dash in traces:
                draw_series_scaled(
                    canvas,
                    18,
                    y,
                    180,
                    height,
                    values,
                    minimum=minimum,
                    maximum=maximum,
                    color=color,
                    line_width=width,
                    dash=dash,
                )

        canvas.text(18, 181, "periodic baseline", size=8.3,
                    color=BLUE, bold=True)
        draw_component(
            154,
            22,
            ((periodic_trace, BLUE, 1.6, (4, 2)),),
            minimum=periodic_min,
            maximum=periodic_max,
        )
        canvas.circle(108, 144, 6.5, fill=WHITE,
                      stroke=MID_GRAY, line_width=0.7)
        canvas.text(108, 140.5, "+", size=11, color=RED,
                    bold=True, align="center")

        canvas.text(18, 135, "rare fault pattern", size=8.3,
                    color=RED, bold=True)
        draw_component(
            108,
            22,
            ((event_pattern, RED, 1.8, ()),),
            minimum=-event_extent,
            maximum=event_extent,
            show_zero=True,
        )
        canvas.circle(108, 99, 6.5, fill=WHITE,
                      stroke=MID_GRAY, line_width=0.7)
        canvas.text(108, 95.5, "=", size=10, color=INK,
                    bold=True, align="center")

        canvas.text(18, 78, "observed = baseline + event", size=8.3,
                    color=INK, bold=True)
        draw_component(
            47,
            25,
            (
                (periodic_trace, BLUE, 1.2, (4, 2)),
                (observed_trace, INK, 1.8, ()),
            ),
            minimum=load_min,
            maximum=load_max,
        )
        canvas.text(71, 36, "history", size=8.0,
                    color=GRAY, align="center")
        canvas.text(161, 36, "future", size=8.0,
                    color=GRAY, align="center")
        canvas.text(
            108,
            8,
            "rare event is added on top of periodic load",
            size=8.2,
            color=RED,
            bold=True,
            align="center",
        )
        branch_y = 59
    else:
        draw_window(
            18,
            63,
            180,
            112,
            ((query_context, INK, 1.7, ()),),
            (
                (query_truth_trace, INK, 1.8, ()),
                (query_base_trace, BLUE, 1.7, (4, 2)),
            ),
            minimum=query_min,
            maximum=query_max,
        )
        draw_series_scaled(
            canvas,
            18 + 180 * time_split * cue_start / (len(query_context) - 1),
            63,
            180 * time_split * 3 / (len(query_context) - 1),
            112,
            query_context[cue_start:],
            minimum=query_min,
            maximum=query_max,
            color=RED,
            line_width=2.4,
        )
        canvas.text(71, 52, "history", size=9.0,
                    color=GRAY, align="center")
        canvas.text(161, 52, "forecast", size=9.0,
                    color=GRAY, align="center")
        origin_x = 18 + 180 * time_split
        canvas.text(origin_x, 178, labels["origin"], size=9.0,
                    color=GRAY, align="center")
        if legend_outside:
            legend_y = (39, 28)
            miss_y = 8
        else:
            legend_y = (161, 149)
            miss_y = 21
        canvas.line(28, legend_y[0], 39, legend_y[0],
                    color=INK, width=1.6)
        canvas.text(43, legend_y[0] - 4, "actual", size=9.0, color=INK)
        canvas.line(28, legend_y[1], 39, legend_y[1],
                    color=BLUE, width=1.6, dash=(4, 2))
        canvas.text(43, legend_y[1] - 4, "model forecast",
                    size=9.0, color=BLUE)
        canvas.text(108, miss_y, labels["miss"], size=9.4,
                    color=RED, bold=True, align="center")
        branch_y = 112

    # The query branches into the two retrieval targets.
    canvas.line(198, branch_y, 212, branch_y, color=MID_GRAY, width=1.0)
    canvas.line(212, 57, 212, 161, color=MID_GRAY, width=1.0)
    arrow(canvas, 212, 161, 226, 161, color=RED, width=1.2, head=4)
    arrow(canvas, 212, 57, 226, 57, color=GREEN, width=1.2, head=4)

    # Conventional retrieval follows the common rhythm.
    canvas.text(230, 198, "(b) Direct retrieval", size=11.2,
                color=RED, bold=True)
    canvas.text(230, 184, labels["direct_subtitle"],
                size=9.0, color=GRAY)
    top_min, top_max = shared_range(
        (query_context, whole_context, whole_future_trace),
        padding=0.10,
    )
    draw_window(
        230,
        132,
        207,
        47,
        (
            (query_context, INK, 1.2, ()),
            (whole_context, RED, 1.8, (4, 2)),
        ),
        (
            (whole_future_trace, RED, 1.8, (4, 2)),
        ),
        minimum=top_min,
        maximum=top_max,
    )
    arrow(canvas, 445, 155, 474, 155, color=RED, width=1.2, head=4)
    canvas.text(459.5, 165, "copy future", size=9.0,
                color=RED, align="center")
    draw_window(
        482,
        132,
        205,
        47,
        ((query_context, INK, 1.0, ()),),
        (
            (query_truth_trace, INK, 1.5, ()),
            (whole_copy_trace, RED, 2.0, (4, 2)),
        ),
        minimum=query_min,
        maximum=query_max,
    )
    canvas.text(584.5, 118, labels["direct_result"], size=9.0,
                color=RED, bold=True, align="center")
    draw_cross(699, 155)

    # Error memory retrieves the model-relative component from another regime.
    canvas.text(230, 93, "(c) Residual retrieval", size=11.2,
                color=GREEN, bold=True)
    canvas.text(230, 80, labels["residual_subtitle"],
                size=9.0, color=GRAY)
    history_x = 230
    history_y = 38
    history_width = 207
    history_height = 38
    draw_window(
        history_x,
        history_y,
        history_width,
        history_height,
        ((failure_context, MID_GRAY, 1.2, ()),),
        (
            (failure_truth_trace, INK, 1.5, ()),
            (failure_base_trace, BLUE, 1.5, (4, 2)),
        ),
        minimum=comparison_min,
        maximum=comparison_max,
    )
    draw_series_scaled(
        canvas,
        history_x
        + history_width * time_split * cue_start / (len(failure_context) - 1),
        history_y,
        history_width
        * time_split
        * (cue_length - 1)
        / (len(failure_context) - 1),
        history_height,
        failure_context[cue_start:],
        minimum=comparison_min,
        maximum=comparison_max,
        color=RED,
        line_width=2.0,
    )
    residual_min, residual_max = shared_range((residual_trace,), padding=0.12)
    residual_x = history_x + history_width * time_split
    residual_y = 18
    residual_width = history_width * (1 - time_split)
    residual_height = 15
    zero_y = (
        residual_y
        + residual_height * (0 - residual_min) / (residual_max - residual_min)
    )
    canvas.text(230, 22, labels["error_label"], size=9.0, color=RED)
    if show_residual_steps:
        arrow(
            canvas,
            history_x + history_width - 8,
            history_y - 1,
            history_x + history_width - 8,
            residual_y + residual_height + 1,
            color=RED,
            width=0.9,
            head=3.0,
        )
    canvas.line(
        residual_x,
        zero_y,
        history_x + history_width,
        zero_y,
        color=LIGHT_GRAY,
        width=0.7,
    )
    draw_series_scaled(
        canvas,
        residual_x,
        residual_y,
        residual_width,
        residual_height,
        residual_trace,
        minimum=residual_min,
        maximum=residual_max,
        color=RED,
        line_width=1.7,
    )
    canvas.line(437, zero_y, 454, zero_y, color=GREEN, width=1.2)
    canvas.line(454, zero_y, 454, 57, color=GREEN, width=1.2)
    arrow(canvas, 454, 57, 474, 57, color=GREEN, width=1.2, head=4)
    canvas.text(462, 66, labels.get("add_action", "add error"), size=9.0,
                color=GREEN, align="center")
    draw_window(
        482,
        history_y,
        205,
        history_height,
        ((query_context, INK, 1.0, ()),),
        (
            (query_base_trace, BLUE, 1.4, (4, 2)),
            (query_truth_trace, INK, 1.6, ()),
            (corrected_trace, GREEN, 2.0, (6, 3)),
        ),
        minimum=comparison_min,
        maximum=comparison_max,
    )
    canvas.text(584.5, 8, labels["residual_result"], size=9.0,
                color=GREEN, bold=True, align="center")
    draw_check(699, 52)
    canvas.save(path)


def intro_motivation_stadium(path: Path) -> None:
    """Preserve the stadium-release version of the motivation figure."""
    _motivation_figure(
        path,
        case=stadium_traffic_case(),
        labels={
            "query_title": "(a) Today: stadium event",
            "origin": "event ends",
            "miss": "Forecast misses departure surge",
            "direct_subtitle": "Nearest match: similar history, no event",
            "direct_result": "Copied future still misses surge",
            "residual_subtitle":
                "Prior event: same missed surge, different level",
            "error_label": "stored model error",
            "add_action": "add error",
            "residual_result": "Keep current base; recover surge",
        },
        cue_length=4,
        validate_stadium_distances=True,
    )


def intro_motivation(path: Path) -> None:
    """Draw the rare grid-fault motivation example used in the manuscript."""
    _motivation_figure(
        path,
        case=grid_outage_case(),
        labels={
            "query_title": "(a) Periodic load + rare fault",
            "origin": "feeder trips",
            "miss": "Forecast misses outage and recovery",
            "direct_subtitle": "Prior outage: same flickers, lower load",
            "direct_result": "Copied recovery has the wrong level",
            "residual_subtitle": "1  Match prior outage",
            "error_label": "2  Extract actual - forecast",
            "add_action": "3  Add error",
            "residual_result":
                "Corrected = today's baseline + stored error",
        },
        cue_length=6,
        legend_outside=True,
        decompose_event=True,
        show_residual_steps=True,
    )


def method_overview(path: Path) -> None:
    """Draw validation-time memory construction and forecast-time correction."""
    case = stadium_traffic_case()
    query_context = case["query_context"]
    query_base = case["query_base"]
    query_truth = case["query_truth"]
    failure_context = case["failure_context"]
    failure_base = case["failure_base"]
    failure_truth = case["failure_truth"]
    event = case["event"]

    def anchored(
        context: Sequence[float],
        future: Sequence[float],
    ) -> tuple[float, ...]:
        return (float(context[-1]), *(float(value) for value in future))

    query_base_trace = anchored(query_context, query_base)
    query_truth_trace = anchored(query_context, query_truth)
    failure_base_trace = anchored(failure_context, failure_base)
    failure_truth_trace = anchored(failure_context, failure_truth)
    event_trace = (0.0, *(float(value) for value in event))
    corrected_trace = tuple(
        base + correction
        for base, correction in zip(query_base_trace, event_trace)
    )
    if corrected_trace != query_truth_trace:
        raise ValueError("Figure 2 correction must recover the Figure 1 surge")

    memory_residuals = tuple(
        tuple(scale * value for value in event_trace)
        for scale in (1.0, 0.85, 1.15)
    )
    traffic_min, traffic_max = shared_range(
        (
            query_context,
            query_base_trace,
            query_truth_trace,
            failure_context,
            failure_base_trace,
            failure_truth_trace,
        ),
        padding=0.10,
    )
    residual_min, residual_max = shared_range(
        memory_residuals,
        padding=0.10,
    )

    canvas = PdfCanvas(720, 324)
    canvas.rounded_rect(
        4, 4, 712, 316, 11,
        fill=CANVAS, stroke=TEAL, line_width=1.5,
    )

    time_split = 17 / 29
    future_fill = (244 / 255, 238 / 255, 226 / 255)

    def draw_stadium_window(
        x: float,
        y: float,
        width: float,
        height: float,
        context: Sequence[float],
        futures: Sequence[
            tuple[
                Sequence[float],
                tuple[float, float, float],
                float,
                Sequence[float],
            ]
        ],
        *,
        highlight_cue: bool,
    ) -> None:
        context_width = width * time_split
        future_width = width - context_width
        canvas.rect(x, y, width, height, fill=PALE_BLUE,
                    stroke=BLUE, line_width=0.7)
        canvas.rect(x + context_width, y, future_width, height,
                    fill=future_fill)
        draw_series_scaled(
            canvas,
            x,
            y,
            context_width,
            height,
            context,
            minimum=traffic_min,
            maximum=traffic_max,
            color=MID_GRAY,
            line_width=1.1,
        )
        if highlight_cue:
            cue_start = len(context) - 4
            cue_x = (
                x
                + context_width * cue_start / (len(context) - 1)
            )
            cue_width = context_width * 3 / (len(context) - 1)
            draw_series_scaled(
                canvas,
                cue_x,
                y,
                cue_width,
                height,
                context[cue_start:],
                minimum=traffic_min,
                maximum=traffic_max,
                color=RED,
                line_width=1.8,
            )
        canvas.line(
            x + context_width,
            y,
            x + context_width,
            y + height,
            color=MID_GRAY,
            width=0.7,
            dash=(2, 2),
        )
        for values, color, line_width, dash in futures:
            draw_series_scaled(
                canvas,
                x + context_width,
                y,
                future_width,
                height,
                values,
                minimum=traffic_min,
                maximum=traffic_max,
                color=color,
                line_width=line_width,
                dash=dash,
            )

    def draw_check(x: float, y: float) -> None:
        canvas.circle(x, y, 7.5, fill=WHITE, stroke=GREEN, line_width=0.9)
        canvas.line(x - 3.1, y, x - 0.7, y - 2.5,
                    color=GREEN, width=1.4)
        canvas.line(x - 0.7, y - 2.5, x + 3.7, y + 3.1,
                    color=GREEN, width=1.4)

    section_frame(
        canvas,
        14,
        177,
        692,
        113,
        "1. VALIDATION: BUILD MEMORY AND LOCK ONE POLICY",
        312,
    )

    canvas.text(81, 266, "Earlier stadium event", size=8.5, color=INK,
                bold=True, align="center")
    draw_stadium_window(
        25,
        204,
        112,
        48,
        failure_context,
        (
            (failure_truth_trace, INK, 1.4, ()),
            (failure_base_trace, BLUE, 1.3, (3, 2)),
        ),
        highlight_cue=True,
    )
    canvas.text(55, 194, "history", size=7.0, color=GRAY, align="center")
    canvas.text(111, 194, "forecast / actual", size=7.0,
                color=GRAY, align="center")

    arrow(canvas, 140, 229, 156, 229, color=TEAL, width=1.3, head=4.5)
    canvas.circle(169, 229, 12, fill=WHITE, stroke=RED, line_width=0.8)
    canvas.text(169, 225.5, "-", size=15, color=RED, bold=True, align="center")
    arrow(canvas, 182, 229, 197, 229, color=TEAL, width=1.3, head=4.5)

    canvas.text(239, 266, "Missed departure surge", size=8.2, color=RED,
                bold=True, align="center")
    canvas.rounded_rect(199, 204, 80, 48, 5, fill=PALE_RED,
                        stroke=RED, line_width=0.8)
    draw_series_scaled(
        canvas, 207, 213, 64, 27, event_trace,
        minimum=residual_min, maximum=residual_max,
        color=RED, line_width=1.7,
    )
    canvas.line(207, 226, 271, 226, color=MID_GRAY, width=0.6, dash=(2, 2))
    arrow(canvas, 282, 229, 298, 229, color=TEAL, width=1.3, head=4.5)

    canvas.text(352, 266, "Error memory", size=8.5, color=INK,
                bold=True, align="center")
    for row, (dy, color) in enumerate(((0, BLUE), (15, GOLD), (30, GREEN))):
        y = 200 + dy
        canvas.rounded_rect(300, y, 105, 13, 3, fill=WHITE,
                            stroke=color, line_width=0.75)
        canvas.circle(308, y + 6.5, 2.7, fill=color)
        draw_series_scaled(
            canvas, 318, y + 2.5, 49, 8, memory_residuals[row],
            minimum=residual_min, maximum=residual_max,
            color=color, line_width=0.8,
        )
        canvas.text(397, y + 4, f"event {3 - row}", size=6.2,
                    color=GRAY, align="right")
    canvas.text(352, 190, "store only after the surge is observed", size=6.9,
                color=TEAL, align="center")
    arrow(canvas, 408, 229, 424, 229, color=TEAL, width=1.3, head=4.5)

    canvas.text(493, 266, "Candidate policies", size=8.5, color=INK,
                bold=True, align="center")
    portfolio = (
        ("residual retrieval", PALE_GREEN, GREEN, 1.3),
        ("seasonal / overlap / bias", PALE_GOLD, GOLD, 0.7),
        ("identity (no change)", WHITE, MID_GRAY, 0.7),
    )
    for index, (label, fill, color, line_width) in enumerate(portfolio):
        y = 236 - index * 18
        canvas.rounded_rect(428, y, 130, 15, 4, fill=fill,
                            stroke=color, line_width=line_width)
        canvas.text(493, y + 4.6, label, size=7.0, color=color,
                    bold=True, align="center")
    arrow(canvas, 561, 229, 577, 229, color=TEAL, width=1.3, head=4.5)

    canvas.rounded_rect(579, 201, 111, 56, 6, fill=PALE_GREEN,
                        stroke=GREEN, line_width=1.0)
    canvas.text(634.5, 266, "Locked policy", size=8.5, color=GREEN,
                bold=True, align="center")
    canvas.text(634.5, 235, "residual retrieval", size=8.0,
                color=INK, align="center")
    canvas.text(634.5, 216, "chosen before test", size=7.4,
                color=GRAY, align="center")

    section_frame(
        canvas,
        14,
        47,
        692,
        113,
        "2. FORECASTING: APPLY THE LOCKED POLICY",
        286,
    )

    canvas.text(78, 137, "Today: event ends", size=8.3, color=INK,
                bold=True, align="center")
    draw_stadium_window(
        24,
        70,
        108,
        58,
        query_context,
        ((query_base_trace, BLUE, 1.5, (3, 2)),),
        highlight_cue=True,
    )

    arrow(canvas, 135, 99, 151, 99, color=TEAL, width=1.3, head=4.5)
    canvas.text(190.5, 137, "Describe event state", size=8.0, color=PURPLE,
                bold=True, align="center")
    canvas.rounded_rect(153, 72, 75, 54, 5, fill=WHITE,
                        stroke=PURPLE, line_width=0.8)
    for row, width in enumerate((49, 35, 57, 42)):
        canvas.rect(163, 110 - row * 9, width, 4,
                    fill=(199 / 255, 174 / 255, 232 / 255))
    arrow(canvas, 231, 99, 247, 99, color=TEAL, width=1.3, head=4.5)

    canvas.text(290, 137, "Retrieve missed surges", size=8.0, color=INK,
                bold=True, align="center")
    for row, (dy, color) in enumerate(((0, BLUE), (17, GOLD), (34, GREEN))):
        y = 72 + dy
        canvas.rounded_rect(249, y, 82, 14, 3, fill=WHITE,
                            stroke=color, line_width=1.0)
        canvas.circle(257, y + 7, 2.8, fill=color)
        draw_series_scaled(
            canvas, 265, y + 2.5, 58, 9, memory_residuals[row],
            minimum=residual_min, maximum=residual_max,
            color=color, line_width=0.85,
        )
    arrow(canvas, 334, 99, 350, 99, color=TEAL, width=1.3, head=4.5)

    canvas.text(407, 137, "Combine agreeing errors", size=8.0, color=INK,
                bold=True, align="center")
    canvas.rounded_rect(352, 72, 110, 54, 5, fill=PALE_PURPLE,
                        stroke=PURPLE, line_width=0.8)
    for row, (width, color) in enumerate(
        ((35, BLUE), (24, GOLD), (15, GREEN))
    ):
        y = 110 - row * 10
        canvas.circle(365, y + 1, 2.8, fill=color)
        canvas.rect(372, y - 1, width, 4, fill=color)
    canvas.line(416, 82, 448, 82, color=MID_GRAY, width=3.2)
    canvas.line(416, 82, 439, 82, color=GREEN, width=3.2)
    canvas.circle(439, 82, 3.5, fill=WHITE, stroke=GREEN, line_width=0.8)
    arrow(canvas, 465, 99, 481, 99, color=TEAL, width=1.3, head=4.5)

    canvas.text(521, 137, "Add missed surge", size=8.3, color=RED,
                bold=True, align="center")
    canvas.rounded_rect(483, 72, 76, 54, 5, fill=PALE_RED,
                        stroke=RED, line_width=0.8)
    draw_series_scaled(
        canvas, 493, 84, 56, 21, event_trace,
        minimum=residual_min, maximum=residual_max,
        color=RED, line_width=1.8,
    )
    canvas.line(493, 94, 549, 94, color=MID_GRAY, width=0.6, dash=(2, 2))

    arrow(canvas, 562, 99, 574, 99, color=TEAL, width=1.2, head=4)
    canvas.circle(586, 99, 12, fill=WHITE, stroke=INK, line_width=0.8)
    canvas.text(586, 95, "+", size=15, color=TEAL, bold=True,
                align="center")
    arrow(canvas, 599, 99, 611, 99, color=TEAL, width=1.2, head=4)

    canvas.text(651, 137, "Corrected matches actual", size=7.8, color=GREEN,
                bold=True, align="center")
    canvas.rounded_rect(613, 70, 74, 54, 5, fill=PALE_GREEN,
                        stroke=GREEN, line_width=1.0)
    for values, color, width, dash in (
        (query_base_trace, BLUE, 1.1, (3, 2)),
        (query_truth_trace, INK, 2.3, ()),
        (corrected_trace, GREEN, 1.8, (4, 2)),
    ):
        draw_series_scaled(
            canvas, 619, 78, 62, 38, values,
            minimum=traffic_min, maximum=traffic_max,
            color=color, line_width=width, dash=dash,
        )
    draw_check(698, 97)

    canvas.line(135, 62, 559, 62, color=MID_GRAY, width=0.8, dash=(4, 2))
    arrow(canvas, 559, 62, 610, 79, color=MID_GRAY, width=0.8, head=3.5)
    canvas.text(347, 53, "If identity wins: keep the model forecast",
                size=7.4, color=GRAY, align="center")

    pill(
        canvas, 156, 14, 408, 24,
        "THE BACKBONE IS NEVER UPDATED; THE LOCKED POLICY ONLY ADDS A CORRECTION",
        fill=PALE_GREEN, stroke=GREEN, color=INK, size=8.0,
    )
    canvas.save(path)


def residual_retrieval_example(path: Path) -> None:
    """Illustrate why model-error retrieval complements a base forecast."""
    real_case = load_real_case()
    series = real_case["series"]
    canvas = PdfCanvas(720, 196)
    canvas.rounded_rect(
        4, 4, 712, 188, 10,
        fill=CANVAS, stroke=TEAL, line_width=1.4,
    )
    panel_y = 11
    panel_height = 174
    panel_width = 228
    panel_xs = (10, 246, 482)

    for x, fill, header, icon, title in (
        (panel_xs[0], PALE_BLUE, HEADER_BLUE, "1",
         "(a) Match observed context"),
        (panel_xs[1], PALE_GOLD, HEADER_GOLD, "2",
         "(b) Retrieve model errors"),
        (panel_xs[2], PALE_GREEN, HEADER_GREEN, "3",
         "(c) Correct conservatively"),
    ):
        card(
            canvas,
            x, panel_y, panel_width, panel_height, title,
            fill=fill, header=header, icon=icon,
            title_size=9.8, header_height=27,
        )
    arrow(canvas, 239, 96, 244, 96, color=TEAL, width=1.2, head=3.5)
    arrow(canvas, 475, 96, 480, 96, color=TEAL, width=1.2, head=3.5)

    # Panel A: the query and retrieved memory contexts have similar shapes.
    x = panel_xs[0]
    plot_x = x + 17
    plot_y = 65
    plot_width = 194
    plot_height = 70
    canvas.text(
        x + panel_width / 2,
        148,
        "Descriptor search finds close historical shapes",
        size=8.5,
        color=GRAY,
        align="center",
    )
    canvas.line(plot_x, plot_y, plot_x, plot_y + plot_height,
                color=LIGHT_GRAY, width=0.6)
    canvas.line(plot_x, plot_y, plot_x + plot_width, plot_y,
                color=LIGHT_GRAY, width=0.6)
    query = compact_series(series["query_context_normalized"], 32)
    neighbor_1 = compact_series(
        series["neighbor_contexts_normalized"][0], 32
    )
    neighbor_2 = compact_series(
        series["neighbor_contexts_normalized"][1], 32
    )
    context_min, context_max = shared_range(
        (query, neighbor_1, neighbor_2)
    )
    for values, color, width, dash in (
        (neighbor_1, GOLD, 1.2, (3, 2)),
        (neighbor_2, GREEN, 1.2, (3, 2)),
        (query, BLUE, 2.2, ()),
    ):
        draw_series_scaled(
            canvas,
            plot_x,
            plot_y,
            plot_width,
            plot_height,
            values,
            minimum=context_min,
            maximum=context_max,
            color=color,
            line_width=width,
            dash=dash,
        )
    legend_y = 43
    for legend_x, color, label, dash in (
        (x + 19, BLUE, "query", ()),
        (x + 80, GOLD, "neighbor 1", (3, 2)),
        (x + 160, GREEN, "neighbor 2", (3, 2)),
    ):
        canvas.line(legend_x, legend_y + 2, legend_x + 14, legend_y + 2,
                    color=color, width=1.5, dash=dash)
        canvas.text(legend_x + 18, legend_y, label, size=7.8, color=INK)
    canvas.text(
        x + panel_width / 2,
        27,
        "Similarity identifies candidates;",
        size=8.0,
        color=GRAY,
        align="center",
    )
    canvas.text(
        x + panel_width / 2,
        16,
        "it does not prove their futures.",
        size=8.0,
        color=GRAY,
        align="center",
    )

    # Panel B: raw analog futures retain level variation, while residuals
    # isolate a shared underprediction pattern of the trained forecaster.
    x = panel_xs[1]
    raw_x = x + 17
    raw_y = 105
    raw_width = 194
    raw_height = 35
    canvas.text(
        x + panel_width / 2,
        148,
        "Raw analog futures mix level and dynamics",
        size=8.5,
        color=GRAY,
        align="center",
    )
    canvas.line(raw_x, raw_y, raw_x + raw_width, raw_y,
                color=LIGHT_GRAY, width=0.6)
    raw_future_values = tuple(
        compact_series(values, 32)
        for values in series["neighbor_futures"][:3]
    )
    future_min, future_max = shared_range(raw_future_values)
    for values, color in zip(raw_future_values, (BLUE, GOLD, MID_GRAY)):
        draw_series_scaled(
            canvas,
            raw_x,
            raw_y,
            raw_width,
            raw_height,
            values,
            minimum=future_min,
            maximum=future_max,
            color=color,
            line_width=1.1,
        )

    residual_x = x + 17
    residual_y = 34
    residual_width = 194
    residual_height = 43
    canvas.text(
        x + panel_width / 2,
        85,
        "Residuals expose model-specific error",
        size=8.5,
        color=GREEN,
        bold=True,
        align="center",
    )
    residual_values = tuple(
        compact_series(values, 32)
        for values in series["neighbor_residuals"][:3]
    )
    retrieved_correction = compact_series(
        series["retrieved_correction"], 32
    )
    residual_min, residual_max = shared_range(
        residual_values + (retrieved_correction, (0.0,))
    )
    zero_y = (
        residual_y
        + residual_height
        * (0.0 - residual_min)
        / (residual_max - residual_min)
    )
    canvas.line(residual_x, zero_y, residual_x + residual_width, zero_y,
                color=LIGHT_GRAY, width=0.6, dash=(2, 2))
    for values, color in zip(residual_values, (BLUE, GOLD, MID_GRAY)):
        draw_series_scaled(
            canvas,
            residual_x,
            residual_y,
            residual_width,
            residual_height,
            values,
            minimum=residual_min,
            maximum=residual_max,
            color=color,
            line_width=1.0,
        )
    draw_series_scaled(
        canvas,
        residual_x,
        residual_y,
        residual_width,
        residual_height,
        retrieved_correction,
        minimum=residual_min,
        maximum=residual_max,
        color=GREEN,
        line_width=2.3,
    )
    canvas.text(
        x + panel_width / 2,
        20,
        "green: applied residual correction",
        size=8.2,
        color=GREEN,
        bold=True,
        align="center",
    )

    # Panel C: add only the reliable component to the query's base forecast.
    x = panel_xs[2]
    forecast_x = x + 17
    forecast_y = 64
    forecast_width = 194
    forecast_height = 76
    canvas.text(
        x + panel_width / 2,
        148,
        "Add the agreed error pattern to the base forecast",
        size=8.5,
        color=GRAY,
        align="center",
    )
    canvas.line(forecast_x, forecast_y, forecast_x + forecast_width, forecast_y,
                color=LIGHT_GRAY, width=0.6)
    base = compact_series(series["query_base_forecast"], 32)
    truth = compact_series(series["query_truth"], 32)
    corrected = compact_series(series["query_corrected_forecast"], 32)
    forecast_min, forecast_max = shared_range((base, truth, corrected))
    for values, color, width, dash in (
        (base, GRAY, 1.4, (4, 2)),
        (truth, INK, 1.4, ()),
        (corrected, GREEN, 2.4, ()),
    ):
        draw_series_scaled(
            canvas,
            forecast_x,
            forecast_y,
            forecast_width,
            forecast_height,
            values,
            minimum=forecast_min,
            maximum=forecast_max,
            color=color,
            line_width=width,
            dash=dash,
        )
    legend_y = 47
    for legend_x, color, label, dash in (
        (x + 18, GRAY, "base", (4, 2)),
        (x + 82, GREEN, "corrected", ()),
        (x + 168, INK, "truth", ()),
    ):
        canvas.line(legend_x, legend_y + 2, legend_x + 14, legend_y + 2,
                    color=color, width=1.6, dash=dash)
        canvas.text(legend_x + 18, legend_y, label, size=7.8, color=INK)
    canvas.text(
        x + panel_width / 2,
        33,
        "reliable residual: add correction",
        size=8.2,
        color=GREEN,
        bold=True,
        align="center",
    )
    canvas.text(
        x + panel_width / 2,
        24,
        "disagreement: shrink toward",
        size=8.2,
        color=GRAY,
        align="center",
    )
    canvas.text(
        x + panel_width / 2,
        15,
        "the identity forecast",
        size=7.8,
        color=GRAY,
        align="center",
    )
    canvas.save(path)


def _bar_panel(
    canvas: PdfCanvas,
    *,
    y: float,
    title: str,
    official: Sequence[float],
    post_test: Sequence[float],
    official_intervals: Sequence[tuple[float, float]] | None = None,
    post_test_intervals: Sequence[tuple[float, float]] | None = None,
    maximum: float = 5.0,
    fill: tuple[float, float, float] = WHITE,
    header: tuple[float, float, float] = HEADER_BLUE,
) -> None:
    datasets = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")
    x0 = 67
    x1 = 332
    plot_width = x1 - x0
    panel_height = 108

    card(
        canvas, 10, y, 340, panel_height, title,
        fill=fill, header=header, icon="%",
        title_size=9.7, header_height=24,
    )

    grid_bottom = y + 15
    grid_top = y + panel_height - 32
    for tick in range(int(maximum) + 1):
        tick_x = x0 + plot_width * tick / maximum
        canvas.line(tick_x, grid_bottom, tick_x, grid_top, color=LIGHT_GRAY,
                    width=0.55)
        canvas.text(tick_x, y + 4, str(tick), size=7.5, color=GRAY,
                    align="center")

    row_top = y + panel_height - 40
    for index, dataset in enumerate(datasets):
        row_y = row_top - index * 17
        canvas.text(63, row_y - 2, dataset, size=8.2, color=INK, bold=True,
                    align="right")
        for offset, value, color, intervals in (
            (1.5, official[index], BLUE, official_intervals),
            (-5.5, post_test[index], GREEN, post_test_intervals),
        ):
            bar_width = plot_width * value / maximum
            canvas.rect(x0, row_y + offset, bar_width, 5.5, fill=color)
            canvas.text(x0 + bar_width + 3, row_y + offset + 0.2,
                        f"{value:.2f}", size=7.2, color=color, bold=True)
            if intervals is not None:
                lower, upper = intervals[index]
                line_y = row_y + offset + 2.75
                lower_x = x0 + plot_width * lower / maximum
                upper_x = x0 + plot_width * upper / maximum
                canvas.line(lower_x, line_y, upper_x, line_y,
                            color=INK, width=0.8)
                canvas.line(lower_x, line_y - 2, lower_x, line_y + 2,
                            color=INK, width=0.8)
                canvas.line(upper_x, line_y - 2, upper_x, line_y + 2,
                            color=INK, width=0.8)


def ett_gains(path: Path) -> None:
    canvas = PdfCanvas(360, 275)
    canvas.rounded_rect(
        4, 4, 352, 267, 10,
        fill=CANVAS, stroke=TEAL, line_width=1.35,
    )
    canvas.rounded_rect(
        63, 249, 234, 20, 10,
        fill=WHITE, stroke=INK, line_width=0.75,
    )
    canvas.rect(76, 256, 11, 6, fill=BLUE)
    canvas.text(91, 254.5, "Official test", size=8.3, color=INK, bold=True)
    canvas.rect(172, 256, 11, 6, fill=GREEN)
    canvas.text(187, 254.5, "Subsequent post-test", size=8.3,
                color=INK, bold=True)

    _bar_panel(
        canvas,
        y=140,
        title="MSE reduction (%)",
        official=(1.86, 2.30, 4.44, 2.36),
        post_test=(2.10, 4.51, 3.67, 2.79),
        official_intervals=(
            (0.201, 3.450),
            (0.511, 3.803),
            (1.361, 7.686),
            (1.470, 3.425),
        ),
        post_test_intervals=(
            (0.688, 3.390),
            (2.482, 6.407),
            (1.180, 6.894),
            (1.970, 3.724),
        ),
        maximum=8.0,
        fill=PALE_BLUE,
        header=HEADER_BLUE,
    )
    _bar_panel(
        canvas,
        y=20,
        title="MAE reduction (%)",
        official=(2.84, 2.55, 2.46, 1.09),
        post_test=(3.01, 3.41, 2.28, 1.83),
        fill=PALE_GREEN,
        header=HEADER_GREEN,
    )
    canvas.save(path)


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).resolve().parents[1] / "latex" / "figures"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Figure output directory (default: {default_output})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "intro_motivation.pdf": intro_motivation,
        "intro_motivation_stadium.pdf": intro_motivation_stadium,
        "method_overview.pdf": method_overview,
        "residual_retrieval_example.pdf": residual_retrieval_example,
        "ett_gains.pdf": ett_gains,
    }
    for filename, generator in outputs.items():
        destination = output_dir / filename
        generator(destination)
        print(destination)


if __name__ == "__main__":
    main()
