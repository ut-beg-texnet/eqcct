"""
EQCCT predictor for scmlpick — integrated with eqcctpro ModelActor pool.

Legacy path: loads TensorFlow EQCCT weights inside each Ray picker task (slow).
Preferred path: ``model_actor`` is an eqcctpro ``ModelActor`` handle; inference
runs via ``predict_from_arrays`` on a persistent actor (weights loaded once).

Requires: ``eqcctpro`` installed (same env as scmlpick).
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import time
import glob
import obspy
import numpy as np
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.threading.set_inter_op_parallelism_threads(8)

import warnings
warnings.filterwarnings("ignore")
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import logging
import ray
from datetime import datetime, timedelta

from eqcctpro.eqcct_tf_models import load_eqcct_model, PreLoadGeneratorTest

_logger = logging.getLogger(__name__)


def _bandpass_hz_from_scmlpick_bindings(stations_filters, net_sta):
    """
    Bandpass corners (Hz) from scmlpick ``df_filters`` (list of ``{key, filter}`` or DataFrame
    with ``sta``, ``hp``, ``lp``). Prefers **exact** ``key == net_sta``, then station-code suffix
    match. Returns None if there is nothing to apply (caller may use a broadband default).
    """
    if stations_filters is None:
        return None
    net, sta = net_sta.split(".", 1)
    if isinstance(stations_filters, list):
        if not stations_filters:
            return None
        for d in stations_filters:
            if d.get("key") == net_sta:
                filt = d["filter"]
                return tuple(
                    sorted(float(x) for x in filt[filt.index("(") + 1 : filt.index(")")].split(",")[1:3])
                )
        for d in stations_filters:
            k = d.get("key", "")
            if isinstance(k, str) and k.split(".", 1)[-1] == sta:
                filt = d["filter"]
                return tuple(
                    sorted(float(x) for x in filt[filt.index("(") + 1 : filt.index(")")].split(",")[1:3])
                )
        return None
    try:
        r = stations_filters[stations_filters.sta == sta].iloc[0]
        return float(r["hp"]), float(r["lp"])
    except Exception:
        return None


def mseed_predictor(
    stream=None,
    filterShift=2.5,
    wfinfo=None,
    files=None,
    P_threshold=0.1,
    S_threshold=0.1,
    normalization_mode="std",
    overlap=0.3,
    batch_size=500,
    overwrite=False,
    p_model=None,
    s_model=None,
    numCPUs=1,
    gpu_id=None,
    gpu_limit=None,
    stations_filters=None,
    playback=False,
    model_actor=None,
    inference_mode="model_actor",
    ripper_gpu_memory_limit_mb=None,
):
    """Run picking for one station × time window.

    Parameters
    ----------
    model_actor : optional ray.actor.ActorHandle
        eqcctpro ``ModelActor``. When set, EQCCT weights stay on the actor and
        this call only prepares windows and collects ``predict_from_arrays``.
    inference_mode : str
        When ``model_actor`` is None, ``ripper`` enables per-task GPU setup
        (TensorFlow env) if ``gpu_id >= 0``; ``model_actor`` skips that hook.
    ripper_gpu_memory_limit_mb : optional int
        Soft VRAM cap per Ripper task (passed to ``eqcctpro.tools.tf_environ``).
    """
    net, sta, t0, t1 = wfinfo
    net_sta = f"{net}.{sta}"

    args = {
        "filterShift": filterShift,
        "P_threshold": P_threshold,
        "S_threshold": S_threshold,
        "normalization_mode": normalization_mode,
        "overlap": overlap,
        "batch_size": batch_size,
        "gpu_id": gpu_id,
        "gpu_limit": gpu_limit,
        "p_model": p_model,
        "s_model": s_model,
        "stations_filters": stations_filters,
        "station": net_sta,
        "files": files,
        "t_ini": t0,
        "t_end": t1,
        "playback": playback,
        "ripper_use_gpu": (
            (inference_mode or "").lower() in ("ripper", "task", "per-task", "legacy")
            and gpu_id is not None
            and int(gpu_id) >= 0
        ),
        "ripper_gpu_memory_limit_mb": ripper_gpu_memory_limit_mb,
    }

    if playback:
        task = [stream, net_sta, args]
    else:
        task = [stream.select(network=net, station=sta), net_sta, args]

    if model_actor is not None:
        return parallel_predict_with_actor(task, model_actor)
    return parallel_predict(task)


def parallel_predict(predict_args):
    """Load EQCCT in-process (legacy Ripper-style for one task)."""
    stream, st, args = predict_args
    if args.get("ripper_use_gpu"):
        from eqcctpro.tools import tf_environ

        tf_environ(
            gpu_id=0,
            vram_limit_mb=args.get("ripper_gpu_memory_limit_mb"),
            gpus_to_use=None,
            intra_threads=2,
            inter_threads=2,
            log_device=False,
            logger=None,
            skip_tf=False,
        )
    model = load_eqcct_model(args["p_model"], args["s_model"])
    start_Predicting = time.time()
    meta, data_set = _readnparray(stream, args, st)
    batch_sizes = [args["batch_size"], 500, 250, 100, 50, 10, 5, 1]
    i = 0
    while True:
        params_pred = {"batch_size": batch_sizes[i], "norm_mode": args["normalization_mode"]}
        pred_generator = PreLoadGeneratorTest(meta["trace_start_time"], data_set, **params_pred)
        predP, predS = model.predict(pred_generator, verbose=0)

        detection_memory = []
        prob_memory = []
        picks = []
        for ix in range(len(predP)):
            Ppicks, Pprob = _picker(args, predP[ix, :, 0])
            Spicks, Sprob = _picker(args, predS[ix, :, 0], "S_threshold")
            pick, detection_memory, prob_memory = _output_dict_prediction(
                meta, Ppicks, Pprob, Spicks, Sprob, detection_memory, prob_memory,
                ix, len(predP), len(predS),
            )
            picks.append(pick)

        end_Predicting = time.time()
        delta = end_Predicting - start_Predicting
        info = f"[{datetime.now()}] {st}: Finished the prediction in {round(delta,2)}s."
        return {"picks": picks, "info": info}


def parallel_predict_with_actor(predict_args, model_actor):
    """Use persistent eqcctpro ModelActor (no per-task TF load)."""
    stream, st, args = predict_args
    start_Predicting = time.time()
    meta, data_set = _readnparray(stream, args, st)
    predP, predS = ray.get(
        model_actor.predict_from_arrays.remote(
            meta["trace_start_time"],
            data_set,
            args["batch_size"],
            args["normalization_mode"],
        )
    )
    detection_memory = []
    prob_memory = []
    picks = []
    for ix in range(len(predP)):
        Ppicks, Pprob = _picker(args, predP[ix, :, 0])
        Spicks, Sprob = _picker(args, predS[ix, :, 0], "S_threshold")
        pick, detection_memory, prob_memory = _output_dict_prediction(
            meta, Ppicks, Pprob, Spicks, Sprob, detection_memory, prob_memory,
            ix, len(predP), len(predS),
        )
        picks.append(pick)

    end_Predicting = time.time()
    delta = end_Predicting - start_Predicting
    info = f"[{datetime.now()}] {st}: Finished the prediction in {round(delta,2)}s. [ModelActor]"
    return {"picks": picks, "info": info}


def _obspy_time_to_datetime(t):
    if t is None:
        return None
    if hasattr(t, "datetime"):
        return t.datetime
    return t


def _prepared_stream_to_seisbench_3c(st, sta_code):
    """Build E,N,Z stream from ``prepare_station_chunk`` output (bandpass/resample already applied)."""
    t0 = max(tr.stats.starttime for tr in st)
    t1 = min(tr.stats.endtime for tr in st)
    st = obspy.Stream(traces=[tr.copy() for tr in st])
    st.trim(t0, t1, pad=False)
    by_last = {}
    for tr in st:
        by_last.setdefault(tr.stats.channel[-1], []).append(tr)

    def _best_trace(letter):
        lst = by_last.get(letter, [])
        return lst[0] if lst else None

    trE = _best_trace("E") or _best_trace("1")
    trN = _best_trace("N") or _best_trace("2")
    trZ = _best_trace("Z")
    missing = []
    if trZ is None:
        missing.append("Z")
    if trE is None:
        missing.append("E (or 1)")
    if trN is None:
        missing.append("N (or 2)")
    if missing:
        chans = [tr.stats.channel for tr in st]
        raise ValueError(
            f"Missing 3C components for {sta_code}: {', '.join(missing)}. Channels: {chans}"
        )
    out = obspy.Stream(traces=[trE.copy(), trN.copy(), trZ.copy()])
    out[0].stats.channel = out[0].stats.channel[:-1] + "E"
    out[1].stats.channel = out[1].stats.channel[:-1] + "N"
    out[2].stats.channel = out[2].stats.channel[:-1] + "Z"
    return out


def _classify_output_to_scmlpick_picks(classify_output, stream3c):
    """Match ``_output_dict_prediction`` dict shape for ``scPhase``."""
    picks_raw = classify_output.picks if hasattr(classify_output, "picks") else []
    p_picks = [p for p in picks_raw if getattr(p, "phase", "P").upper() == "P"]
    s_picks = [p for p in picks_raw if getattr(p, "phase", "P").upper() == "S"]

    tr0 = stream3c[0]
    st_Z = stream3c.select(channel="*Z")
    channel_out = st_Z[0].stats.channel if len(st_Z) else stream3c[0].stats.channel
    trace_name = f"{tr0.stats.network}.{tr0.stats.station}.{tr0.stats.location}.{channel_out}"
    network_name = "{:<2}".format(tr0.stats.network)
    station_name = "{:<4}".format(tr0.stats.station)
    instrument_type = "{:<2}".format("  ")
    coord = getattr(tr0.stats, "coordinates", None) or {}
    station_lat = float(coord.get("latitude", 0.0) or 0.0)
    station_lon = float(coord.get("longitude", 0.0) or 0.0)
    station_elv = float(coord.get("elevation", 0.0) or 0.0)

    out_rows = []
    used_s = set()
    for p in p_picks:
        p_time = getattr(p, "peak_time", getattr(p, "start_time", getattr(p, "time", None)))
        p_prob = float(getattr(p, "peak_value", getattr(p, "score", getattr(p, "value", 0.0))))
        if p_time is None:
            continue
        match_s = None
        for s in s_picks:
            s_time = getattr(s, "peak_time", getattr(s, "start_time", getattr(s, "time", None)))
            if s in used_s or s_time is None:
                continue
            try:
                gap = float(s_time - p_time)
            except Exception:
                continue
            if 0 < gap < 30:
                match_s = s
                used_s.add(s)
                break

        if match_s:
            ms_time = getattr(
                match_s, "peak_time", getattr(match_s, "start_time", getattr(match_s, "time", None))
            )
            ms_prob = float(
                getattr(
                    match_s, "peak_value", getattr(match_s, "score", getattr(match_s, "value", 0.0))
                )
            )
            s_dt = _obspy_time_to_datetime(ms_time)
            s_prob = ms_prob
        else:
            s_dt = None
            s_prob = None

        out_rows.append(
            {
                "trace_name": trace_name,
                "network_name": network_name,
                "station_name": station_name,
                "instrument_type": instrument_type,
                "station_lat": station_lat,
                "station_lon": station_lon,
                "station_elv": station_elv,
                "PdateTime": _obspy_time_to_datetime(p_time),
                "p_prob": np.float64(p_prob),
                "SdateTime": s_dt,
                "s_prob": np.float64(s_prob) if s_prob is not None else None,
            }
        )

    return out_rows


def mseed_predictor_seisbench(
    stream=None,
    filterShift=2.5,
    wfinfo=None,
    files=None,
    stations_filters=None,
    playback=False,
    parent_model="PhaseNet",
    child_model="original",
    P_threshold=0.3,
    S_threshold=0.3,
    Detection_threshold=0.3,
    model_actor=None,
    inference_mode="model_actor",
    gpu_id=None,
    ripper_gpu_memory_limit_mb=None,
):
    """One station × window via SeisBench ``classify``; output compatible with ``scPhase``."""
    net, sta, t0, t1 = wfinfo
    net_sta = f"{net}.{sta}"

    # Bandpass from bindings only (exact NET.STA first, then suffix). Do not rely on a hidden
    # fallback inside prepare_station_chunk when a BW(...) string is present for this station.
    resolved_bp = _bandpass_hz_from_scmlpick_bindings(stations_filters, net_sta)
    if resolved_bp is None:
        resolved_bp = (1.0, 45.0)
        if stations_filters is not None and (
            (isinstance(stations_filters, list) and len(stations_filters) > 0)
            or (hasattr(stations_filters, "empty") and not stations_filters.empty)
        ):
            _logger.warning(
                "SeisBench %s: no matching profiles.*.filter in bindings; using broadband %.1f–%.1f Hz.",
                net_sta,
                resolved_bp[0],
                resolved_bp[1],
            )

    if playback:
        st = prepare_station_chunk(
            files, net_sta, t0, t1, stations_filters=None, default_band=resolved_bp
        )
    else:
        st = prepare_station_chunk(
            stream, net_sta, t0, t1, stations_filters=None, default_band=resolved_bp
        )

    if not st or len(st) == 0:
        return {
            "picks": [],
            "info": f"[{datetime.now()}] {net_sta}: SeisBench — empty stream after prepare_station_chunk.",
        }

    try:
        stream3c = _prepared_stream_to_seisbench_3c(st, sta)
    except Exception as exc:
        return {
            "picks": [],
            "info": f"[{datetime.now()}] {net_sta}: SeisBench 3C prep failed: {exc}",
        }

    mode = (inference_mode or "model_actor").lower()

    if model_actor is not None and mode != "ripper":
        classify_output = ray.get(
            model_actor.classify.remote(
                stream3c,
                P_threshold=P_threshold,
                S_threshold=S_threshold,
                Detection_threshold=Detection_threshold,
                strict=False,
                flexible_horizontal_components=True,
            )
        )
    else:
        use_gpu = gpu_id is not None and int(gpu_id) >= 0
        from eqcctpro.seisbench_models import SeisBenchModels

        wrapper = SeisBenchModels(parent_model, child_model, validate_pretrained=False)
        wrapper.load_model()
        import torch

        device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        if use_gpu and torch.cuda.is_available():
            try:
                if hasattr(wrapper.model, "to"):
                    wrapper.model.to(device)
            except Exception:
                pass
        classify_output = wrapper.classify(
            stream3c,
            P_threshold=P_threshold,
            S_threshold=S_threshold,
            Detection_threshold=Detection_threshold,
            strict=False,
            flexible_horizontal_components=True,
        )
        del wrapper
        if use_gpu and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    pick_dicts = _classify_output_to_scmlpick_picks(classify_output, stream3c)
    info = (
        f"[{datetime.now()}] {net_sta}: SeisBench {parent_model}/{child_model} — "
        f"{len(pick_dicts)} pick row(s)."
    )
    return {"picks": pick_dicts, "info": info}


def get_stream_filtered(net_sta: str,start,end,files):

    net, sta  = net_sta.split(".")
    st = obspy.Stream()
    for fn in files:
        try:
            temp_st = obspy.read(fn)
        except Exception as e:
            print(f"[READ ERROR] {fn}: {e!r}")
            continue

        try:
            temp_st.merge(fill_value=0)
        except Exception as e:
            print(f"[MERGE ERROR] {fn}: {e!r}")
            try:
                temp_st.merge(fill_value=0)
            except Exception as e2:
                print(f"[MERGE RETRY FAILED] {fn}: {e2!r}")

        # Select only the requested net/station before adding
        temp_st = temp_st.select(network=net, station=sta)
        if temp_st:
            st += temp_st
    # try:
    st.trim(start, end, pad=True, fill_value=0)
    # except Exception as e:
    #     print(f"[TRIM ERROR] Station={net}.{sta} Window=[{start} .. {end}] -> {e!r}")

    if len(st) == 0:
        print(f"Empty stream for {net}.{sta} in window [{start} .. {end}]")
        return None

    print(f"Stream ready: {net}.{sta} in window [{start} .. {end}] with {len(st)} traces")
    return st

def prepare_station_chunk(st_or_files, net_sta, t0, t1, stations_filters=None,
                          default_band=(1.0, 45.0)):

    net, sta = net_sta.split(".")

    # -------- REAL-TIME --------
    if isinstance(st_or_files, obspy.Stream):
        st = st_or_files.select(network=net, station=sta).copy()
        if not st:
            return None

        # 1) Merge + detrend(linear) + demean
        st.merge(fill_value=0)
        st.detrend("linear")
        st.detrend("demean")

        # 2) Fixed 5-second taper (exact legacy behavior)
        try:
            max_percentage = 5.0 / (st[0].stats.delta * st[0].stats.npts)
            st.taper(max_percentage=max_percentage, type="cosine")
        except Exception:
            pass

        # 3) Bandpass per station: bindings (exact NET.STA first) or default_band
        fmin, fmax = default_band
        if stations_filters is not None:
            resolved = _bandpass_hz_from_scmlpick_bindings(stations_filters, net_sta)
            if resolved is not None:
                fmin, fmax = resolved
        st.filter(type="bandpass", freqmin=fmin, freqmax=fmax, corners=2, zerophase=True)

        # 4) Resample to 100 Hz if required (interpolate first, fallback to resample)
        if any(tr.stats.sampling_rate != 100.0 for tr in st):
            try:
                st.interpolate(100.0, method="linear")
            except Exception:
                for tr in st:
                    tr.resample(100.0)

        # 5) Final trim to common [min, max] (legacy)
        st.trim(min(tr.stats.starttime for tr in st),
                max(tr.stats.endtime for tr in st),
                pad=True, fill_value=0)
        return st

    # -------- PLAYBACK: robust load/clean/filter + exact trim to [t0, t1] --------
    st = obspy.Stream()

    # Load and select only the target net.sta
    for fn in st_or_files:
        try:
            tmp = obspy.read(fn)
        except Exception:
            continue
        tmp = tmp.select(network=net, station=sta)
        if tmp:
            st += tmp

    if not st:
        return None

    # Clean masked arrays and non-finite values before merge/filter
    for tr in st:
        data = tr.data
        try:
            if np.ma.isMaskedArray(data):
                data = data.filled(0)
        except Exception:
            pass
        data = np.asarray(data)
        if not np.isfinite(data).all():
            bad = ~np.isfinite(data)
            if bad.any():
                data = data.copy()
                data[bad] = 0
        tr.data = data

    # Merge with fill_value=0 (bridge gaps with zeros)
    try:
        st.merge(method=1, fill_value=0)  # method=1 keeps gaps simple; fill with zeros
    except Exception:
        st.merge(fill_value=0)

    # Detrend + demean
    try:
        st.detrend("linear")
        st.detrend("demean")
    except Exception:
        pass

    # Fixed 5-second taper (stable across window lengths)
    try:
        max_percentage = 5.0 / (st[0].stats.delta * st[0].stats.npts)
        st.taper(max_percentage=max_percentage, type="cosine")
    except Exception:
        pass

    # Station-specific bandpass: bindings (exact NET.STA first) or default_band
    fmin, fmax = default_band
    if stations_filters is not None:
        resolved = _bandpass_hz_from_scmlpick_bindings(stations_filters, net_sta)
        if resolved is not None:
            fmin, fmax = resolved

    # Apply bandpass (fallback per-trace if needed)
    try:
        st.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=2, zerophase=True)
    except Exception:
        new_st = obspy.Stream()
        for tr in st:
            try:
                tr2 = tr.copy()
                tr2.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=2, zerophase=True)
                new_st += tr2
            except Exception:
                new_st += tr.copy()
        st = new_st

    # Resample to 100 Hz if required
    if any(abs(tr.stats.sampling_rate - 100.0) > 1e-6 for tr in st):
        try:
            st.interpolate(100.0, method="linear")
        except Exception:
            for tr in st:
                tr.resample(100.0)

    # Final exact trim to [t0, t1] (no extended window in playback)
    st.trim(t0, t1, pad=True, fill_value=0)

    # Sanitize to float32 and ensure finite values
    for tr in st:
        arr = np.asarray(tr.data)
        if not np.isfinite(arr).all():
            bad = ~np.isfinite(arr)
            if bad.any():
                arr = arr.copy()
                arr[bad] = 0
        tr.data = arr.astype(np.float32, copy=False)

    return st


def _readnparray(stream, args, st_name):
    # 1) Station-prepared Stream
    if args["playback"]:
        st = prepare_station_chunk(
            args["files"], args["station"], args["t_ini"], args["t_end"],
            stations_filters=args.get("stations_filters")
        )
    else:
        st = prepare_station_chunk(
            stream, args["station"], args["t_ini"], args["t_end"],
            stations_filters=args.get("stations_filters")
        )
    if not st:
        raise RuntimeError("Empty stream: no traces loaded.")

    span_start = min(tr.stats.starttime for tr in st)
    span_end   = max(tr.stats.endtime   for tr in st)

    if not args["playback"]:
        span_start = span_start + args['filterShift']

    # Map station components
    components = {tr.stats.channel[-1]: tr for tr in st}
    st_Z = st.select(channel="*Z")
    channel_out = st_Z[0].stats.channel if len(st_Z) > 0 else st[0].stats.channel

    meta = {
        "start_time": span_start,
        "end_time":   span_end,
        "trace_name": f"{st_name}.{st[0].stats.location}.{channel_out}"
    }

    data_set = {}
    st_times = []

    # Preferred component mapping per column
    components_list = [
        ['E', '1'],  # Column 0
        ['N', '2'],  # Column 1
        ['Z']        # Column 2
    ]

    # 3) Window construction
    if args["playback"]:
        # 60 s windows with overlap
        step_sec = int(60 - (args.get('overlap', 0.0) * 60))
        step_sec = max(1, step_sec)

        current = span_start
        # Ensure monotonically increasing windows
        while current < span_end:
            window_end = current + 60
            st_times.append(str(current).replace('T', ' ').replace('Z', ''))

            npz_data = np.zeros((6000, 3))
            for col_idx, comp_options in enumerate(components_list):
                for comp in comp_options:
                    if comp in components:
                        tr = components[comp].copy().slice(current, window_end, nearest_sample=False)
                        data = tr.data[:6000]
                        if len(data) < 6000:
                            data = np.pad(data, (0, 6000 - len(data)), 'constant')
                        npz_data[:, col_idx] = data
                        break

            key = str(current).replace('T', ' ').replace('Z', '')
            data_set[key] = npz_data
            current += step_sec

    else:
        # Real-time: single window.
        # If we have at least 60 s, prefer the last 60 s; otherwise use the full span.
        if (span_end - span_start) >= 60:
            win_start = span_end - 60
            win_end   = span_end
        else:
            win_start = span_start
            win_end   = span_end

        st_times.append(str(win_start).replace('T', ' ').replace('Z', ''))
        npz_data = np.zeros((6000, 3))
        for col_idx, comp_options in enumerate(components_list):
            for comp in comp_options:
                if comp in components:
                    tr = components[comp].copy().slice(win_start, win_end, nearest_sample=False)
                    data = tr.data[:6000]
                    if len(data) < 6000:
                        data = np.pad(data, (0, 6000 - len(data)), 'constant')
                    npz_data[:, col_idx] = data
                    break

        key = str(win_start).replace('T', ' ').replace('Z', '')
        data_set[key] = npz_data

    meta["trace_start_time"] = st_times

    # 4) Populate basic metadata
    try:
        meta.update({
            "receiver_code": st[0].stats.station,
            "instrument_type": 0,
            "network_code": 0,
            "receiver_latitude": 0,
            "receiver_longitude": 0,
            "receiver_elevation_m": 0
        })
    except Exception:
        meta.update({
            "receiver_code": st_name,
            "instrument_type": 0,
            "network_code": 0,
            "receiver_latitude": 0,
            "receiver_longitude": 0,
            "receiver_elevation_m": 0
        })

    return meta, data_set

def _output_writter_prediction(meta, csvPr, Ppicks, Pprob, Spicks, Sprob, detection_memory,prob_memory,predict_writer, idx, cq, cqq):

    """ 
    
    Writes the detection & picking results into a CSV file.

    Parameters
    ----------
    dataset: hdf5 obj
        Dataset object of the trace.

    predict_writer: obj
        For writing out the detection/picking results in the CSV file. 
       
    csvPr: obj
        For writing out the detection/picking results in the CSV file.  

    matches: dic
        It contains the information for the detected and picked event.  
  
    snr: list of two floats
        Estimated signal to noise ratios for picked P and S phases.   
    
    detection_memory : list
        Keep the track of detected events.          
        
    Returns
    -------   
    detection_memory : list
        Keep the track of detected events.  
        
        
    """      

    station_name = meta["receiver_code"]
    station_lat = meta["receiver_latitude"]
    station_lon = meta["receiver_longitude"]
    station_elv = meta["receiver_elevation_m"]
    start_time = meta["trace_start_time"][idx]
    station_name = "{:<4}".format(station_name)
    network_name = meta["network_code"]
    network_name = "{:<2}".format(network_name)
    instrument_type = meta["instrument_type"]
    instrument_type = "{:<2}".format(instrument_type)  

    try:
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S.%f')
    except Exception:
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        
    def _date_convertor(r):  
        if isinstance(r, str):
            mls = r.split('.')
            if len(mls) == 1:
                new_t = datetime.strptime(r, '%Y-%m-%d %H:%M:%S')
            else:
                new_t = datetime.strptime(r, '%Y-%m-%d %H:%M:%S.%f')
        else:
            new_t = r
            
        return new_t
    
    p_time = []
    p_prob = []
    PdateTime = []
    if Ppicks[0]!=None: 
#for iP in range(len(Ppicks)):
#if Ppicks[iP]!=None: 
        p_time.append(start_time+timedelta(seconds= Ppicks[0]/100))
        p_prob.append(Pprob[0])
        PdateTime.append(_date_convertor(start_time+timedelta(seconds= Ppicks[0]/100)))
        detection_memory.append(p_time) 
        prob_memory.append(p_prob)  
    else:          
        p_time.append(None)
        p_prob.append(None)
        PdateTime.append(None)

    s_time = []
    s_prob = []    
    SdateTime=[]
    if Spicks[0]!=None: 
#for iS in range(len(Spicks)):
#if Spicks[iS]!=None: 
        s_time.append(start_time+timedelta(seconds= Spicks[0]/100))
        s_prob.append(Sprob[0])
        SdateTime.append(_date_convertor(start_time+timedelta(seconds= Spicks[0]/100)))
    else:
        s_time.append(None)
        s_prob.append(None)
        SdateTime.append(None)

    SdateTime = np.array(SdateTime)
    s_prob = np.array(s_prob)
    
    p_prob = np.array(p_prob)
    PdateTime = np.array(PdateTime)
        
    predict_writer.writerow([meta["trace_name"], 
                     network_name,
                     station_name, 
                     instrument_type,
                     station_lat, 
                     station_lon,
                     station_elv,
                     PdateTime[0], 
                     p_prob[0],
                     SdateTime[0], 
                     s_prob[0]
                     ]) 



    csvPr.flush()                


    return detection_memory,prob_memory  

def _output_dict_prediction(meta, Ppicks, Pprob, Spicks, Sprob, detection_memory,prob_memory, idx, cq, cqq):

    """ 
    
    Writes the detection & picking results into a dictionary.

    Parameters
    ----------
    dataset: hdf5 obj
        Dataset object of the trace.

    matches: dic
        It contains the information for the detected and picked event.  
  
    snr: list of two floats
        Estimated signal to noise ratios for picked P and S phases.   
    
    detection_memory : list
        Keep the track of detected events.          
        
    Returns
    -------   
    detection_memory : list
        Keep the track of detected events.  
        
        
    """      

    station_name = meta["receiver_code"]
    station_lat = meta["receiver_latitude"]
    station_lon = meta["receiver_longitude"]
    station_elv = meta["receiver_elevation_m"]
    start_time = meta["trace_start_time"][idx]
    station_name = "{:<4}".format(station_name)
    network_name = meta["network_code"]
    network_name = "{:<2}".format(network_name)
    instrument_type = meta["instrument_type"]
    instrument_type = "{:<2}".format(instrument_type)  

    try:
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S.%f')
    except Exception:
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        
    def _date_convertor(r):  
        if isinstance(r, str):
            mls = r.split('.')
            if len(mls) == 1:
                new_t = datetime.strptime(r, '%Y-%m-%d %H:%M:%S')
            else:
                new_t = datetime.strptime(r, '%Y-%m-%d %H:%M:%S.%f')
        else:
            new_t = r
            
        return new_t
    
    p_time = []
    p_prob = []
    PdateTime = []
    if Ppicks[0]!=None: 
#for iP in range(len(Ppicks)):
#if Ppicks[iP]!=None: 
        p_time.append(start_time+timedelta(seconds= Ppicks[0]/100))
        p_prob.append(Pprob[0])
        PdateTime.append(_date_convertor(start_time+timedelta(seconds= Ppicks[0]/100)))
        detection_memory.append(p_time) 
        prob_memory.append(p_prob)  
    else:          
        p_time.append(None)
        p_prob.append(None)
        PdateTime.append(None)

    s_time = []
    s_prob = []    
    SdateTime=[]
    if Spicks[0]!=None: 
#for iS in range(len(Spicks)):
#if Spicks[iS]!=None: 
        s_time.append(start_time+timedelta(seconds= Spicks[0]/100))
        s_prob.append(Sprob[0])
        SdateTime.append(_date_convertor(start_time+timedelta(seconds= Spicks[0]/100)))
    else:
        s_time.append(None)
        s_prob.append(None)
        SdateTime.append(None)

    SdateTime = np.array(SdateTime)
    s_prob = np.array(s_prob)
    
    p_prob = np.array(p_prob)
    PdateTime = np.array(PdateTime)
    
    pick = {
        "trace_name": meta["trace_name"],
        "network_name": network_name,
        "station_name": station_name,
        "instrument_type": instrument_type,
        "station_lat": station_lat,
        "station_lon": station_lon,
        "station_elv": station_elv,
        "PdateTime": PdateTime[0],
        "p_prob": p_prob[0],
        "SdateTime": SdateTime[0],
        "s_prob": s_prob[0]
    }          

    return pick,detection_memory,prob_memory  


def _get_snr(data, pat, window=200):
    
    """ 
    
    Estimates SNR.
    
    Parameters
    ----------
    data : numpy array
        3 component data.    
        
    pat: positive integer
        Sample point where a specific phase arrives. 
        
    window: positive integer, default=200
        The length of the window for calculating the SNR (in the sample).         
        
    Returns
   --------   
    snr : {float, None}
       Estimated SNR in db. 
       
        
    """      
    import math
    snr = None
    if pat:
        try:
            if int(pat) >= window and (int(pat)+window) < len(data):
                nw1 = data[int(pat)-window : int(pat)];
                sw1 = data[int(pat) : int(pat)+window];
                snr = round(10*math.log10((np.percentile(sw1,95)/np.percentile(nw1,95))**2), 1)           
            elif int(pat) < window and (int(pat)+window) < len(data):
                window = int(pat)
                nw1 = data[int(pat)-window : int(pat)];
                sw1 = data[int(pat) : int(pat)+window];
                snr = round(10*math.log10((np.percentile(sw1,95)/np.percentile(nw1,95))**2), 1)
            elif (int(pat)+window) > len(data):
                window = len(data)-int(pat)
                nw1 = data[int(pat)-window : int(pat)];
                sw1 = data[int(pat) : int(pat)+window];
                snr = round(10*math.log10((np.percentile(sw1,95)/np.percentile(nw1,95))**2), 1)    
        except Exception:
            pass
    return snr 


def _detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False):

    """
    
    Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
        
    mph : {None, number}, default=None
        detect peaks that are greater than minimum peak height.
        
    mpd : int, default=1
        detect peaks that are at least separated by minimum peak distance (in number of data).
        
    threshold : int, default=0
        detect peaks (valleys) that are greater (smaller) than `threshold in relation to their immediate neighbors.
        
    edge : str, default=rising
        for a flat peak, keep only the rising edge ('rising'), only the falling edge ('falling'), both edges ('both'), or don't detect a flat peak (None).
        
    kpsh : bool, default=False
        keep peaks with same height even if they are closer than `mpd`.
        
    valley : bool, default=False
        if True (1), detect valleys (local minima) instead of peaks.

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Modified from 
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind


def _picker(args, yh3, thr_type='P_threshold'):
    """ 
    Performs detection and picking.

    Parameters
    ----------
    args : dic
        A dictionary containing all of the input parameters.  
        
    yh1 : 1D array
         probability. 

    Returns
    --------    
    Ppickall: Pick.
    Pproball: Pick Probability.                           
                
    """
    P_PICKall=[]
    Ppickall=[]
    Pproball = []
    perrorall=[]

    sP_arr = _detect_peaks(yh3, mph=args[thr_type], mpd=1)

    P_PICKS = []
    pick_errors = []
    if len(sP_arr) > 0:
        P_uncertainty = None  

        for pick in range(len(sP_arr)):        
            sauto = sP_arr[pick]


            if sauto: 
                P_prob = np.round(yh3[int(sauto)], 3) 
                P_PICKS.append([sauto,P_prob, P_uncertainty]) 

    so=[]
    si=[]
    P_PICKS = np.array(P_PICKS)
    P_PICKall.append(P_PICKS)
    for ij in P_PICKS:
        so.append(ij[1])
        si.append(ij[0])
    try:
        so = np.array(so)
        inds = np.argmax(so)
        swave = si[inds]
        Ppickall.append((swave))
        Pproball.append((np.max(so)))
    except:
        Ppickall.append(None)
        Pproball.append(None)

    #print(np.shape(Ppickall))
    #Ppickall = np.array(Ppickall)
    #Pproball = np.array(Pproball)
    
    return Ppickall, Pproball


def _resampling(st):
    'perform resampling on Obspy stream objects'
    
    need_resampling = [tr for tr in st if tr.stats.sampling_rate != 100.0]
    if len(need_resampling) > 0:
       # print('resampling ...', flush=True)    
        for indx, tr in enumerate(need_resampling):
            if tr.stats.delta < 0.01:
                tr.filter('lowpass',freq=45,zerophase=True)
            tr.resample(100)
            tr.stats.sampling_rate = 100
            tr.stats.delta = 0.01
            tr.data.dtype = 'int32'
            st.remove(tr)                    
            st.append(tr) 
    return st 


def _normalize(data, mode = 'max'):  
    """ 
    
    Normalize 3D arrays.
    
    Parameters
    ----------
    data : 3D numpy array
        3 component traces. 
        
    mode : str, default='std'
        Mode of normalization. 'max' or 'std'     
        
    Returns
    -------  
    data : 3D numpy array
        normalized data. 
            
    """  
       
    data -= np.mean(data, axis=0, keepdims=True)
    if mode == 'max':
        max_data = np.max(data, axis=0, keepdims=True)
        assert(max_data.shape[-1] == data.shape[-1])
        max_data[max_data == 0] = 1
        data /= max_data              

    elif mode == 'std':               
        std_data = np.std(data, axis=0, keepdims=True)
        assert(std_data.shape[-1] == data.shape[-1])
        std_data[std_data == 0] = 1
        data /= std_data
    return data
