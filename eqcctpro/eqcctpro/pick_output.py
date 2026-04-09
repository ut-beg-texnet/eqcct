"""
Station pick output serialization (XML, run-level ASCII summary table, or legacy CSV).
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

# Per-station file when the run-level table is ``ascii`` (XML or CSV only; not ``ascii``).
ASCII_STATION_PICK_FORMATS = frozenset({"xml", "csv"})

# One run-level file: header + one row per station; monospace column alignment.
ASCII_RUN_SUMMARY_COLUMNS = (
    "Station_name",
    "Analysis_time_window",
    "N_P_picks",
    "N_S_picks",
    "Model_name",
    "Detection_Confidence_Threshold",
)

SUMMARY_RESULTS_ASCII = "summary_results.ascii"


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


def _parse_pick_time_sort_key(time_str: str):
    """Parse pick time string for chronological sorting; returns datetime or None."""
    s = str(time_str).strip()
    if not s:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def write_station_pick_log(
    out_dir: str,
    station_name: str,
    events: list[tuple[str, str]],
) -> None:
    """
    Write ``<station>_outputs/<station>.log`` under *out_dir* (same directory as
    ``X_prediction_results.xml`` / ``.csv`` for that station): first line is the
    station id, then one line per pick as ``YYYY-MM-DDTHH:MM:SS.ffffffP`` or ``...S``
    (ISO-like ``T``, microsecond width, phase suffix), sorted by time then P before
    S at equal times.
    """
    sta = str(station_name).strip()
    save_dir = os.path.join(str(out_dir), f"{sta}_outputs")
    path = os.path.join(save_dir, f"{sta}.log")
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    phase_rank = {"P": 0, "S": 1}

    def sort_key(item: tuple[str, str]):
        t_str, ph = item
        ph_up = str(ph).strip().upper()[:1]
        dt = _parse_pick_time_sort_key(t_str)
        if dt is None:
            return datetime.min, phase_rank.get(ph_up, 9)
        return dt, phase_rank.get(ph_up, 9)

    ordered = sorted(events, key=sort_key)
    lines = [sta]
    for t_str, ph in ordered:
        ph_up = str(ph).strip().upper()[:1]
        if ph_up not in ("P", "S"):
            continue
        dt = _parse_pick_time_sort_key(t_str)
        if dt is None:
            continue
        lines.append(dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + ph_up)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def ascii_summary_results_path(
    output_dir: str,
    *,
    timechunk_id: str | None = None,
    total_timechunks=None,
) -> str:
    """
    Per-run ASCII summary path. Multiple configured timechunks use one file per chunk
    id so runs do not overwrite each other; a single-chunk (or unknown) layout uses
    ``summary_results.ascii`` only.
    """
    try:
        ntc = int(total_timechunks) if total_timechunks is not None else None
    except (TypeError, ValueError):
        ntc = None
    if timechunk_id and ntc is not None and ntc > 1:
        safe = "".join(
            c if (c.isalnum() or c in "._-") else "_" for c in str(timechunk_id)
        )
        return os.path.join(output_dir, f"summary_results_{safe}.ascii")
    return os.path.join(output_dir, SUMMARY_RESULTS_ASCII)


def build_ascii_summary_row_tuple(
    station_name: str,
    args: dict,
    *,
    p_phases: list[str],
    s_phases: list[str],
) -> tuple[str, ...]:
    """One aligned row for :func:`write_ascii_run_summary` (station task builds this)."""
    window = str(args.get("analysis_time_window_str") or "").strip()
    return (
        str(station_name).strip(),
        window,
        str(len(p_phases)),
        str(len(s_phases)),
        str(args.get("picker_model_label") or ""),
        str(args.get("detection_confidence_threshold") or ""),
    )


def read_ascii_run_summary_data_rows(path: str) -> list[tuple[str, ...]]:
    """Parse existing ``summary_results*.ascii`` data rows (skip header)."""
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    if not lines:
        return []
    n = len(ASCII_RUN_SUMMARY_COLUMNS)
    rows: list[tuple[str, ...]] = []
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split(" | ")]
        if len(parts) >= n:
            rows.append(tuple(parts[:n]))
    return rows


def merge_ascii_summary_rows(
    path: str, new_rows: list[tuple[str, ...]]
) -> list[tuple[str, ...]]:
    """
    Merge *new_rows* into any existing summary at *path* by station name (first column).
    Stations only in *new_rows* replace or add; stations absent from *new_rows* are kept.
    """
    new_rows = list(new_rows or [])
    new_rows.sort(key=lambda r: str(r[0]).upper())
    if not os.path.isfile(path):
        return new_rows
    old = read_ascii_run_summary_data_rows(path)
    if not new_rows:
        return sorted(old, key=lambda r: str(r[0]).upper())
    by = {str(r[0]).strip(): r for r in old}
    for r in new_rows:
        by[str(r[0]).strip()] = r
    return sorted(by.values(), key=lambda r: str(r[0]).upper())


def write_ascii_run_summary(path: str, rows: list[tuple[str, ...]]) -> None:
    """
    Write one UTF-8 summary table: header + one row per station.
    Columns are space-padded to the same width so values line up under headers.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    headers = list(ASCII_RUN_SUMMARY_COLUMNS)
    str_matrix = [headers] + [[str(c) for c in r] for r in rows]
    ncols = len(headers)
    widths = [
        max(len(str_matrix[i][j]) for i in range(len(str_matrix)))
        for j in range(ncols)
    ]
    lines = [
        " | ".join(str_matrix[i][j].ljust(widths[j]) for j in range(ncols))
        for i in range(len(str_matrix))
    ]
    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def normalize_pick_output_format(fmt) -> str:
    if fmt is None:
        return "xml"
    s = str(fmt).lower().strip()
    if s not in SUPPORTED_PICK_OUTPUT_FORMATS:
        raise ValueError(
            f"pick_output_format must be one of {sorted(SUPPORTED_PICK_OUTPUT_FORMATS)}, got {fmt!r}"
        )
    return s


def normalize_ascii_station_pick_format(fmt) -> str:
    """Format for per-station files when ``pick_output_format`` is ``ascii`` (``xml`` or ``csv`` only)."""
    if fmt is None:
        return "xml"
    s = str(fmt).lower().strip()
    if s not in ASCII_STATION_PICK_FORMATS:
        raise ValueError(
            "ascii_station_pick_format must be 'xml' or 'csv' "
            f"(per-station picks while using run-level ASCII summary), got {fmt!r}"
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
    When ``pick_output_format='ascii'`` at the driver, workers use this sink with
    ``fmt`` from ``ascii_station_pick_format`` (``xml`` or ``csv``) so each station gets
    ``X_prediction_results.xml`` or ``X_prediction_results.csv`` in that station's
    ``_outputs`` directory, while the driver writes run-level ``summary_results.ascii``
    and :func:`write_station_pick_log` writes ``<station>.log`` beside those files
    (see :func:`write_ascii_run_summary`).
    """

    def __init__(self, path: str, fmt: str):
        self.path = path
        self.fmt = normalize_pick_output_format(fmt)
        if self.fmt == "ascii":
            raise ValueError(
                "ASCII output is aggregated into summary_results.ascii by the driver; "
                "prediction workers must not use PickOutputSink for ascii."
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
