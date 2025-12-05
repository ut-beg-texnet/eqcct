#!/usr/bin/env python3
import os
import math
import argparse
import numpy as np
from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client

# optional seisbench import
try:
    import seisbench.models as sbm
except ImportError:
    sbm = None

def ensure_dir(path):
    """Ensures that a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)

def to_list(val):
    """
    Convert a comma-separated string to a list.
    If val is "*" or empty, return None (meaning "all").
    """
    if val is None:
        return None
    s = val.strip()
    if s == "" or s == "*":
        return None
    return [v.strip() for v in s.split(",") if v.strip()]

def split_windows(t_start, t_end, chunk_min):
    """
    Yield successive (start, end) UTCDateTime pairs of chunk_min minutes.
    The last chunk may be shorter.
    """
    chunk_sec = chunk_min * 60
    cur = t_start
    while cur < t_end:
        nxt = cur + chunk_sec
        if nxt > t_end:
            nxt = t_end
        yield cur, nxt
        cur = nxt

def get_window_count(t_start, t_end, chunk_min):
    """
    Return how many chunk_min-minute windows fit between t_start and t_end
    (ceiling of the division).
    """
    total_sec = t_end - t_start
    chunk_sec = chunk_min * 60
    return math.ceil(total_sec / chunk_sec)

def parse_stream_selector(selector_string):
    """Parses a single stream selector like 'NET.STA.LOC.CHA' into a tuple."""
    parts = selector_string.split('.')
    if len(parts) != 4:
        raise ValueError(f"Invalid stream selector format: '{selector_string}'. Expected NNN.SSS.LLL.CCC.")
    network, station, location, channel = parts
    return network, to_list(station), to_list(location), to_list(channel)


def main():
    p = argparse.ArgumentParser(
        description="Download FDSN waveforms in equal-time chunks."
    )
    p.add_argument("--start", help="Start time, e.g. 2024-12-03T00:00:00Z", required=False)
    p.add_argument("--end", help="End time,   e.g. 2024-12-03T02:00:00Z", required=False)
    p.add_argument("--streams", help="Comma-separated stream selectors, e.g., 'A1.*.*.HH?,ZA.*.*.HH?,GT.*.*.BH?'", required=True)
    p.add_argument("--host", default="http://localhost:8080", help="FDSNWS base URL")
    p.add_argument("--output", default=".", help="Base output directory")
    p.add_argument("--chunk", type=int, default=None,
                   help="Chunk size in minutes. Splits start–end into N windows.")
    p.add_argument("--denoise", action="store_true",
                   help="If set, apply seisbench.DeepDenoiser to each chunk.")
    args = p.parse_args()

    if not args.start:
        args.start = input("Start time (ISO): ").strip()
    if not args.end:
        args.end = input("End time (ISO): ").strip()

    # parse times
    try:
        t_start = UTCDateTime(args.start)
        t_end = UTCDateTime(args.end)
    except Exception as e:
        print(f"Error parsing times: {e}")
        return

    # parse stream selectors
    stream_selectors = [s.strip() for s in args.streams.split(',')]
    parsed_selectors = []
    for selector in stream_selectors:
        try:
            net, stations, locations, channels = parse_stream_selector(selector)
            parsed_selectors.append((net, stations, locations, channels))
        except ValueError as e:
            print(f"Error: {e}")
            return

    # build windows
    if args.chunk:
        windows = list(split_windows(t_start, t_end, args.chunk))
        wcount = get_window_count(t_start, t_end, args.chunk)
        print(f"Total time: {t_start} → {t_end}  ({t_end - t_start} s)")
        print(f"Chunk size: {args.chunk} minute(s) → {wcount} window(s)\n")
    else:
        windows = [(t_start, t_end)]
        wcount = 1

    # prepare client
    try:
        client = Client(args.host)
    except Exception as e:
        print(f"Could not create FDSN client: {e}")
        return

    # prepare denoiser
    denoiser = None
    if args.denoise:
        if sbm is None:
            print("seisbench not installed; cannot denoise.")
            return
        print("Loading seisbench DeepDenoiser…")
        denoiser = sbm.DeepDenoiser.from_pretrained("urban")

    # prepare output base
    base_dir = os.path.abspath(args.output)
    ensure_dir(base_dir)

    # loop over windows
    for idx, (win_start, win_end) in enumerate(windows, 1):
        start_str = win_start.strftime("%Y%m%dT%H%M%SZ")
        end_str = win_end.strftime("%Y%m%dT%H%M%SZ")
        time_dir = os.path.join(base_dir, f"{start_str}_{end_str}")
        ensure_dir(time_dir)

        print(f"\n=== WINDOW {idx}/{wcount}: {start_str} → {end_str} ===")

        # download all traces for this window
        streams_by_key = {}
        for net, stations, locations, channels in parsed_selectors:
            sta_iter = stations if stations is not None else ["*"]
            loc_iter = locations if locations is not None else ["*"]
            cha_iter = channels if channels is not None else ["*"]

            print(f"  Fetching for: {net}.{','.join(sta_iter)}.{','.join(loc_iter)}.{','.join(cha_iter)}")

            for sta in sta_iter:
                for loc in loc_iter:
                    for cha in cha_iter:
                        try:
                            st = client.get_waveforms(net, sta, loc, cha,
                                                      win_start, win_end)
                            if not st:
                                continue
                            for tr in st:
                                key = (tr.stats.network,
                                       tr.stats.station,
                                       tr.stats.location,
                                       tr.stats.channel)
                                streams_by_key.setdefault(key, []).append(tr)
                        except Exception as e:
                            print(f"  Warning: can't fetch {net}.{sta}.{loc}.{cha}: {e}")

        if not streams_by_key:
            print("  No data downloaded for this window.")
            continue

        # write out each station/channel bundle
        for (net, sta, loc, cha), traces in streams_by_key.items():
            station_dir = os.path.join(time_dir, sta)
            ensure_dir(station_dir)

            if not traces:
                continue

            st_out = Stream(traces)

            # optionally denoise
            if denoiser:
                st_out = denoiser.annotate(st_out)

            if len(st_out) == 0:
                continue

            # cast to int32
            for tr in st_out:
                tr.data = tr.data.astype(np.int32)

            # strip any DeepDenoiser_ prefix
            for tr in st_out:
                tr.stats.channel = tr.stats.channel.replace("DeepDenoiser_", "")

            # pick a channel name for the filename (all traces share same channel)
            out_chan = st_out[0].stats.channel
            loc_str = loc if loc not in (None, "") else ""
            fname = f"{net}.{sta}.{loc_str}.{out_chan}__{start_str}__{end_str}.mseed"
            fpath = os.path.join(station_dir, fname)

            try:
                st_out.write(fpath, format="MSEED",
                             reclen=512, encoding="STEIM2", byteorder=">")
                print(f"  Wrote {fpath}  ({len(st_out)} traces)")
            except Exception as e:
                print(f"  Error writing {fpath}: {e}")

    print("\nAll windows processed. Done.")

if __name__ == "__main__":
    main()
