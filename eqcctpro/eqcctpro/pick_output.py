"""
Station pick output serialization (XML, per-station ASCII summary, or legacy CSV).
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from xml.sax.saxutils import escape

PICK_RESULT_COLUMNS = (
    "file_name",
    "network",
    "station",
    "instrument_type",
    "station_lat",
    "station_lon",
    "station_elv",
    "p_arrival_time",
    "p_probability",
    "s_arrival_time",
    "s_probability",
)

SUPPORTED_PICK_OUTPUT_FORMATS = frozenset({"xml", "ascii", "csv"})

# One header row + one TSV data row; times in P_Phase / S_Phase are semicolon-separated.
ASCII_SUMMARY_COLUMNS = (
    "Station_name",
    "Time_of_the_picks",
    "P_Phase",
    "S_Phase",
    "Model_name",
    "Detection_Confidence_Threshold",
)


def resolve_picker_model_label(
    model_type: str | None,
    seisbench_parent_model: str | None,
    seisbench_child_model: str | None,
) -> str:
    mt = (model_type or "eqcct").lower()
    if mt == "eqcct":
        return "EQCCT"
    parent = (seisbench_parent_model or "").strip()
    child = (seisbench_child_model or "").strip()
    if parent and child:
        return f"{parent}/{child}"
    return parent or child or "SeisBench"


def format_detection_confidence_threshold_summary(args: dict) -> str:
    """
    Single string for the ASCII summary column; compatible with EQCCT and SeisBench args dicts.
    """
    mt = str(args.get("model_type") or "eqcct").lower()
    p_t = args.get("P_threshold")
    s_t = args.get("S_threshold")
    det = args.get("Detection_threshold")
    if mt == "eqcct":
        return f"P_Threshold={p_t};S_Threshold={s_t}"
    return f"Detection_Threshold={det};P_Threshold={p_t};S_Threshold={s_t}"


def format_pick_time_cell(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")


def _unique_sorted_join(values: list[str]) -> str:
    if not values:
        return ""
    return ";".join(sorted(set(values)))


def write_ascii_station_summary(
    path: str,
    *,
    station_name: str,
    time_of_the_picks_minutes: float | None,
    p_phases: list[str],
    s_phases: list[str],
    model_name: str,
    detection_confidence_threshold: str,
) -> None:
    """Write one TSV station summary (header + one row). *time_of_the_picks_minutes* is the full analysis span in minutes."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    t_cell = "" if time_of_the_picks_minutes is None else f"{float(time_of_the_picks_minutes):.12g}"
    row = [
        str(station_name).strip(),
        t_cell,
        _unique_sorted_join(p_phases),
        _unique_sorted_join(s_phases),
        model_name,
        detection_confidence_threshold,
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        w.writerow(list(ASCII_SUMMARY_COLUMNS))
        w.writerow(row)


def normalize_pick_output_format(fmt) -> str:
    if fmt is None:
        return "xml"
    s = str(fmt).lower().strip()
    if s not in SUPPORTED_PICK_OUTPUT_FORMATS:
        raise ValueError(
            f"pick_output_format must be one of {sorted(SUPPORTED_PICK_OUTPUT_FORMATS)}, got {fmt!r}"
        )
    return s


def prediction_results_filename(fmt: str) -> str:
    fmt = normalize_pick_output_format(fmt)
    reverse = {"xml": ".xml", "ascii": ".ascii", "csv": ".csv"}
    return f"X_prediction_results{reverse[fmt]}"


def prediction_results_path(save_dir: str, fmt: str) -> str:
    return os.path.join(save_dir, prediction_results_filename(fmt))


class PickOutputSink:
    """
    Writes a single station result file for XML or legacy CSV row-per-window layout.
    ``pick_output_format='ascii'`` uses :func:`write_ascii_station_summary` instead.
    """

    def __init__(self, path: str, fmt: str):
        self.path = path
        self.fmt = normalize_pick_output_format(fmt)
        if self.fmt == "ascii":
            raise ValueError(
                "ASCII output is a one-row-per-station summary; use write_ascii_station_summary() "
                "from the prediction workers, not PickOutputSink."
            )
        self._f = open(path, "w", encoding="utf-8")
        self._csv_writer = None
        if self.fmt == "csv":
            self._csv_writer = csv.writer(
                self._f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
        elif self.fmt == "xml":
            self._f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            self._f.write('<eqcctpro_picks version="1.0">\n')

    def write_header(self):
        if self._csv_writer:
            self._csv_writer.writerow(list(PICK_RESULT_COLUMNS))

    def write_pick_row(self, row):
        if len(row) != len(PICK_RESULT_COLUMNS):
            raise ValueError(
                f"pick row must have {len(PICK_RESULT_COLUMNS)} fields, got {len(row)}"
            )
        if self._csv_writer:
            self._csv_writer.writerow(row)
            return
        self._f.write(" <pick>\n")
        for name, val in zip(PICK_RESULT_COLUMNS, row):
            self._f.write(f"  <{name}>{escape(str(val))}</{name}>\n")
        self._f.write(" </pick>\n")

    def flush(self):
        self._f.flush()

    def close(self):
        if self.fmt == "xml" and self._csv_writer is None:
            self._f.write("</eqcctpro_picks>\n")
        self._f.close()
