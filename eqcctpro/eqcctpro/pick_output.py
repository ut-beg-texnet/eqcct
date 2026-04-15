"""
Station pick output serialization (XML, run-level ASCII summary table, or legacy CSV).
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from xml.sax.saxutils import escape

from eqcctpro.tools import looks_like_timechunk_id

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
    "Expected_Header_Samples",
    "Decoded_Samples",
    "MSEED_errors",
)

# Tag keys from :func:`~eqcctpro.tools._read_mseed_best_effort` reports (and ``decode_failed``).
# Shown in ``MSEED_errors`` column; full text in docs: ``docs/summary_results_mseed_columns.md``.
MSEED_ERROR_TAG_GLOSSARY: dict[str, str] = {
    "per_channel_longest_prefix": (
        "Physical-record demux path used longest-prefix Steim decode for one channel blob "
        "(marginal Steim / tail corruption)."
    ),
    "whole_file_longest_prefix": (
        "Whole-file strict decode failed; largest decodable physical-record prefix was used "
        "(data may be shorter than headers imply)."
    ),
    "loose_obspy_read": (
        "Fallback generic obspy.read() succeeded after strict miniSEED paths returned empty."
    ),
    "loose_check_compression": (
        "Fallback obspy.read(..., check_compression=False) was used."
    ),
    "decoded_shorter_than_header": (
        "Sum of decoded samples is less than head-only header npts (partial decode vs. declared length)."
    ),
    "decode_failed": (
        "No samples decoded from this miniSEED file (strict and recovery paths exhausted)."
    ),
}


def aggregate_station_mseed_for_summary(file_reports: list[dict]) -> dict[str, str]:
    """
    Build three display strings for the ASCII summary row from per-file ``report`` dicts
    (same structure as :func:`~eqcctpro.tools.read_station_waveform_file` miniSEED reports).
    """
    if not file_reports:
        return {
            "expected_header_samples": "",
            "decoded_samples": "",
            "mseed_errors": "no miniSEED files aggregated for this station",
        }
    dec_sum = 0
    exp_sum = 0
    exp_known = 0
    tags_acc: set[str] = set()
    any_failed = False
    for rep in file_reports:
        dec_sum += int(rep.get("decoded_samples") or 0)
        ex = rep.get("expected_header_samples")
        if ex is not None:
            exp_sum += int(ex)
            exp_known += 1
        q = (rep.get("quality") or "").upper()
        if q == "FAILED":
            any_failed = True
        for t in rep.get("recovery_tags") or []:
            if t:
                tags_acc.add(str(t))
    n = len(file_reports)
    if exp_known == n:
        exp_str = str(exp_sum)
    elif exp_known > 0:
        exp_str = f"{exp_sum} (partial; {exp_known}/{n} files with header npts)"
    else:
        exp_str = "?"
    dec_str = str(dec_sum)
    if not tags_acc and not any_failed:
        err_str = "OK"
    else:
        parts = []
        for code in sorted(tags_acc):
            gloss = MSEED_ERROR_TAG_GLOSSARY.get(code, code)
            parts.append(f"{code}: {gloss}")
        if any_failed and "decode_failed" not in tags_acc:
            parts.append(
                "decode_failed: "
                + MSEED_ERROR_TAG_GLOSSARY["decode_failed"]
            )
        err_str = " | ".join(parts)
    return {
        "expected_header_samples": exp_str,
        "decoded_samples": dec_str,
        "mseed_errors": err_str,
    }


def lookup_station_mseed_summary(args: dict | None, station_name: str) -> dict[str, str] | None:
    """Return precomputed per-station miniSEED stats from ``args['station_mseed_stats']`` if present."""
    if not isinstance(args, dict):
        return None
    sms = args.get("station_mseed_stats")
    if not isinstance(sms, dict):
        return None
    key = str(station_name).strip()
    row = sms.get(key)
    if row is None:
        row = sms.get(key.upper())
    return row if isinstance(row, dict) else None

SUMMARY_RESULTS_ASCII = "summary_results.ascii"

MSEED_ERROR_REFERENCE_BASENAME = "mseed_error_tags_reference.txt"

# Aggregated across timechunk subdirectories under a common picks root (``output_dir``).
EXECUTIVE_PICKS_SUMMARY_BASENAME = "executive_summary.ascii"
EXECUTIVE_PICKS_SUMMARY_COLUMNS = (
    "Station_name",
    "Analysis_time_window",
    "N_P_picks",
    "N_S_picks",
    "Model_name",
    "Detection_Confidence_Threshold",
    "Problematic Time Windows",
)


def write_mseed_error_reference_beside_summary(summary_results_path: str) -> None:
    """
    Write a short tag glossary next to ``summary_results.ascii`` in the same directory
    (once per directory; no-op if file already exists).
    """
    parent = os.path.dirname(os.path.abspath(summary_results_path))
    if not parent:
        return
    ref_path = os.path.join(parent, MSEED_ERROR_REFERENCE_BASENAME)
    if os.path.isfile(ref_path):
        return
    lines = [
        "EQCCTPro — meanings for tags in the MSEED_errors column of summary_results.ascii",
        "(same machine-readable keys as driver logs; full notes in docs/summary_results_mseed_columns.md).",
        "",
    ]
    for code, text in sorted(MSEED_ERROR_TAG_GLOSSARY.items()):
        lines.append(f"{code}")
        lines.append(f"  {text}")
        lines.append("")
    try:
        with open(ref_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines).rstrip() + "\n")
    except OSError:
        pass


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
    mseed_stats: dict[str, str] | None = None,
) -> tuple[str, ...]:
    """One aligned row for :func:`write_ascii_run_summary` (station task builds this)."""
    window = str(args.get("analysis_time_window_str") or "").strip()
    ms = mseed_stats if mseed_stats is not None else lookup_station_mseed_summary(
        args, station_name
    )
    if ms:
        exp_s = str(ms.get("expected_header_samples") or "")
        dec_s = str(ms.get("decoded_samples") or "")
        err_s = str(ms.get("mseed_errors") or "")
    else:
        exp_s, dec_s, err_s = "", "", "not available (no driver preload stats)"
    return (
        str(station_name).strip(),
        window,
        str(len(p_phases)),
        str(len(s_phases)),
        str(args.get("picker_model_label") or ""),
        str(args.get("detection_confidence_threshold") or ""),
        exp_s,
        dec_s,
        err_s,
    )


def split_ascii_table_row(line: str, n_columns: int) -> list[str]:
    """
    Split one row from a padded `` ... | ... | ... `` table.

    Uses at most *n_columns* - 1 splits on ``' | '`` so cell text may contain ``|``
    without breaking (last column keeps any embedded separators).
    """
    if n_columns <= 1:
        return [line.strip()]
    parts = line.split(" | ", n_columns - 1)
    parts = [p.strip() for p in parts]
    if len(parts) < n_columns:
        parts.extend([""] * (n_columns - len(parts)))
    return parts[:n_columns]


def read_ascii_run_summary_data_rows(path: str) -> list[tuple[str, ...]]:
    """Parse existing ``summary_results*.ascii`` data rows (skip header)."""
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    if not lines:
        return []
    target = len(ASCII_RUN_SUMMARY_COLUMNS)
    rows: list[tuple[str, ...]] = []
    for ln in lines[1:]:
        parts = split_ascii_table_row(ln, target)
        rows.append(tuple(parts[:target]))
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


def write_padded_ascii_table(
    path: str,
    headers: tuple[str, ...] | list[str],
    rows: list[tuple[str, ...]],
) -> None:
    """Write UTF-8 table with ``' | '`` separators and padded columns (same as run summaries)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    headers = list(headers)
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


def write_ascii_run_summary(path: str, rows: list[tuple[str, ...]]) -> None:
    """
    Write one UTF-8 summary table: header + one row per station.
    Columns are space-padded to the same width so values line up under headers.
    """
    write_padded_ascii_table(path, ASCII_RUN_SUMMARY_COLUMNS, rows)


def _mseed_cell_problematic_for_executive(mseed_errors: str) -> bool:
    """True if the per-chunk MSEED_errors cell indicates a decode/quality issue worth flagging."""
    s = (mseed_errors or "").strip()
    if not s or s.upper() == "OK":
        return False
    if "not available (no driver preload stats)" in s.lower():
        return False
    return True


def discover_timechunk_summary_files(picks_root_dir: str) -> list[tuple[str, str]]:
    """
    Return sorted ``(timechunk_id, path_to_summary_results)`` for subdirs that look like
    timechunk ids and contain ``summary_results.ascii`` or ``summary_results_<id>.ascii``.
    """
    root = os.path.abspath(picks_root_dir)
    if not os.path.isdir(root):
        return []
    out: list[tuple[str, str]] = []
    try:
        names = os.listdir(root)
    except OSError:
        return []
    for name in names:
        if not looks_like_timechunk_id(name):
            continue
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)
        candidates = (
            os.path.join(d, SUMMARY_RESULTS_ASCII),
            os.path.join(d, f"summary_results_{safe}.ascii"),
        )
        for p in candidates:
            if os.path.isfile(p):
                out.append((name, p))
                break
    out.sort(key=lambda x: x[0])
    return out


def build_executive_picks_summary_rows(picks_root_dir: str) -> list[tuple[str, ...]]:
    """
    One row per station: totals of P/S picks across all timechunk summaries under *picks_root_dir*,
    plus a list of timechunk ids where that station had non-OK miniSEED quality in the chunk table.
    """
    chunks = discover_timechunk_summary_files(picks_root_dir)
    if not chunks:
        return []

    by_station: dict[str, dict] = {}
    idx_mseed = 8  # ``MSEED_errors`` in :data:`ASCII_RUN_SUMMARY_COLUMNS`

    for tcid, path in chunks:
        rows = read_ascii_run_summary_data_rows(path)
        for r in rows:
            if len(r) < 6:
                continue
            sta = str(r[0]).strip()
            if not sta:
                continue
            bucket = by_station.setdefault(
                sta,
                {
                    "np": 0,
                    "ns": 0,
                    "window": "",
                    "model": "",
                    "det": "",
                    "bad_chunks": [],
                },
            )
            try:
                bucket["np"] += int(str(r[2]).strip() or 0)
            except ValueError:
                pass
            try:
                bucket["ns"] += int(str(r[3]).strip() or 0)
            except ValueError:
                pass
            if not bucket["window"] and len(r) > 1:
                bucket["window"] = str(r[1]).strip()
            if not bucket["model"] and len(r) > 4:
                bucket["model"] = str(r[4]).strip()
            if not bucket["det"] and len(r) > 5:
                bucket["det"] = str(r[5]).strip()
            mseed = str(r[idx_mseed]).strip() if len(r) > idx_mseed else ""
            if _mseed_cell_problematic_for_executive(mseed):
                bucket["bad_chunks"].append(tcid)

    exec_rows: list[tuple[str, ...]] = []
    # Cap list length so the table stays usable when many days have miniSEED recovery tags.
    try:
        max_ids = int(os.environ.get("EQCCTPRO_EXECUTIVE_PROBLEMATIC_CHUNK_CAP", "80"))
    except ValueError:
        max_ids = 80
    max_ids = max(8, min(500, max_ids))

    for sta in sorted(by_station.keys(), key=lambda s: str(s).upper()):
        b = by_station[sta]
        bad = b["bad_chunks"]
        if not bad:
            bad_cell = "-"
        elif len(bad) <= max_ids:
            bad_cell = "; ".join(bad)
        else:
            head = "; ".join(bad[:max_ids])
            bad_cell = f"{head}; (+{len(bad) - max_ids} more timechunk(s))"
        exec_rows.append(
            (
                sta,
                b["window"] or "-",
                str(b["np"]),
                str(b["ns"]),
                b["model"] or "-",
                b["det"] or "-",
                bad_cell,
            )
        )
    return exec_rows


def format_total_trial_duration_dhms(seconds: float | None) -> str:
    """
    Wall-clock string for the **Analysis_time_window** column on the final **Total Trial Time** row:
    ``0 Days 5 Hrs 12 Min 30 Sec``.
    """
    if seconds is None:
        return "not recorded"
    try:
        s = int(round(float(seconds)))
    except (TypeError, ValueError):
        return "not recorded"
    if s < 0:
        s = 0
    days, r = divmod(s, 86400)
    hours, r = divmod(r, 3600)
    mins, secs = divmod(r, 60)
    return f"{days} Days {hours} Hrs {mins} Min {secs} Sec"


def write_executive_picks_summary_file(
    path: str,
    station_rows: list[tuple[str, ...]],
    *,
    total_trial_time_seconds: float | None = None,
) -> None:
    """
    Write ``executive_summary.ascii``: header, one row per station, then optional **Total Trial Time**
    row at the bottom (same column layout; duration in **Analysis_time_window**).
    """
    ncols = len(EXECUTIVE_PICKS_SUMMARY_COLUMNS)
    headers = list(EXECUTIVE_PICKS_SUMMARY_COLUMNS)
    str_matrix: list[list[str]] = [headers]
    str_matrix.extend([[str(c) for c in r] for r in station_rows])
    if total_trial_time_seconds is not None:
        dur = format_total_trial_duration_dhms(total_trial_time_seconds)
        spacer_row = [""] * ncols
        str_matrix.append(spacer_row)
        trial_row = ["Total Trial Time", dur] + ["-"] * (ncols - 2)
        str_matrix.append(trial_row)
    widths = [
        max(len(str_matrix[i][j]) for i in range(len(str_matrix)))
        for j in range(ncols)
    ]
    lines = [
        " | ".join(str_matrix[i][j].ljust(widths[j]) for j in range(ncols))
        for i in range(len(str_matrix))
    ]
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines) + "\n")


def refresh_executive_picks_summary(
    picks_root_dir: str,
    *,
    total_trial_time_seconds: float | None = None,
) -> str | None:
    """
    Write ``executive_summary.ascii`` next to timechunk subdirectories (same directory as chunk folders).

    When *total_trial_time_seconds* is set (typically the full ``mseed_predictor`` wall time from
    ``trial_start_time`` to process end), a **Total Trial Time** row is appended after all station
    rows, with days / hrs / min / sec in the **Analysis_time_window** column. Per-chunk refreshes
    omit this by passing ``None``; the final refresh at the end of the trial should pass the
    measured seconds.

    Returns the path written, or ``None`` if nothing was written (no chunk summaries found).
    """
    rows = build_executive_picks_summary_rows(picks_root_dir)
    if not rows:
        return None
    out = os.path.join(os.path.abspath(picks_root_dir), EXECUTIVE_PICKS_SUMMARY_BASENAME)
    write_executive_picks_summary_file(out, rows, total_trial_time_seconds=total_trial_time_seconds)
    return out


def refresh_executive_picks_summary_from_chunk_output_dir(
    chunk_output_dir: str,
    *,
    total_trial_time_seconds: float | None = None,
) -> str | None:
    """
    If *chunk_output_dir* is a timechunk picks folder (e.g. ``.../picks/20251001T..._20251002T...``),
    recompute the aggregate ``executive_summary.ascii`` in the parent ``picks`` directory.
    """
    base = os.path.basename(os.path.abspath(chunk_output_dir))
    if not looks_like_timechunk_id(base):
        return None
    picks_root = os.path.dirname(os.path.abspath(chunk_output_dir))
    return refresh_executive_picks_summary(
        picks_root, total_trial_time_seconds=total_trial_time_seconds
    )


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
