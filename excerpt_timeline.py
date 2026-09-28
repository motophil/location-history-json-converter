#!/usr/bin/env python3
"""
Extract a time-bounded excerpt from a Google Timeline.json file, so you can
share a small sample without handing over your whole location history.

Keeps the same top-level structure (semanticSegments / rawSignals) as the
original file, so it can be fed straight into timeline_to_gpx.py or used as
a sample for further work.

Usage:
    python excerpt_timeline.py Timeline.json excerpt.json \
        --start 2024-01-05-14-30 --end 2024-01-05-18-00

Start/end format is YYYY-DD-MM-HH-mm (year, day, month, hour, minute), e.g.
2024-05-01-14-30 means 2024, day 05, month 01 (5 Jan 2024), 14:30.
"""

import argparse
import json
import sys
from datetime import datetime, timezone


def parse_arg_time(s):
    """Parse a YYYY-MM-DD-HH-mm string into a timezone-aware UTC datetime."""
    try:
        dt = datetime.strptime(s.strip(), "%Y-%m-%d-%H-%M")
    except ValueError as e:
        raise ValueError(
            f"Could not parse {s!r} as YYYY-MM-DD-HH-mm (e.g. 2024-05-01-14-30 "
            f"for 5 Jan 2024, 14:30): {e}"
        )
    return dt.replace(tzinfo=timezone.utc)


def parse_iso_time(t):
    """Parse an ISO 8601 timestamp string (as found in Timeline.json) to UTC."""
    if not t or not isinstance(t, str):
        return None
    s = t.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def overlaps(seg_start, seg_end, start, end):
    """True if [seg_start, seg_end] overlaps [start, end]. Missing bounds pass."""
    if seg_end is not None and start is not None and seg_end < start:
        return False
    if seg_start is not None and end is not None and seg_start > end:
        return False
    return True


def excerpt_semantic_segments(segments, start, end, trim_paths):
    kept = []
    for seg in segments:
        seg_start = parse_iso_time(seg.get("startTime"))
        seg_end = parse_iso_time(seg.get("endTime")) or seg_start
        if not overlaps(seg_start, seg_end, start, end):
            continue

        seg = dict(seg)  # shallow copy so we don't mutate the original
        if trim_paths and "timelinePath" in seg:
            trimmed = []
            for tp in seg["timelinePath"]:
                t = parse_iso_time(tp.get("time"))
                if t is None or (start is None or t >= start) and (end is None or t <= end):
                    trimmed.append(tp)
            seg["timelinePath"] = trimmed
        kept.append(seg)
    return kept


def excerpt_raw_signals(signals, start, end):
    kept = []
    for signal in signals:
        pos = signal.get("position")
        if not pos:
            continue
        t = parse_iso_time(pos.get("timestamp"))
        if t is None:
            continue
        if (start is None or t >= start) and (end is None or t <= end):
            kept.append(signal)
    return kept


def excerpt_locations(locations, start, end):
    """Old Takeout 'locations' array format."""
    kept = []
    for loc in locations:
        t = None
        if "timestamp" in loc:
            t = parse_iso_time(loc["timestamp"])
        elif "timestampMs" in loc:
            try:
                t = datetime.fromtimestamp(int(loc["timestampMs"]) / 1000, tz=timezone.utc)
            except (ValueError, TypeError):
                t = None
        if t is None:
            continue
        if (start is None or t >= start) and (end is None or t <= end):
            kept.append(loc)
    return kept


def main():
    parser = argparse.ArgumentParser(
        description="Extract a time-bounded excerpt from a Google Timeline.json file."
    )
    parser.add_argument("input", help="Path to the full Timeline.json (or Takeout Records.json)")
    parser.add_argument("output", help="Path to write the excerpt JSON")
    parser.add_argument(
        "--start", required=True, help="Start time, format YYYY-DD-MM-HH-mm (e.g. 2024-05-01-14-30)"
    )
    parser.add_argument(
        "--end", required=True, help="End time, format YYYY-DD-MM-HH-mm (e.g. 2024-05-01-18-00)"
    )
    parser.add_argument(
        "--no-trim-paths",
        action="store_true",
        help="Keep every timelinePath point of an overlapping segment, instead of trimming "
        "timelinePath points to just those inside the window",
    )
    parser.add_argument(
        "--include-profile",
        action="store_true",
        help="Also include the 'userLocationProfile' section (frequent places/trips), which "
        "isn't time-scoped and is omitted by default to keep the excerpt small",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the output JSON (default is compact, to keep the file small)",
    )

    args = parser.parse_args()

    try:
        start = parse_arg_time(args.start)
        end = parse_arg_time(args.end)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if end < start:
        print("Error: --end is before --start.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error reading {args.input}: {e}", file=sys.stderr)
        sys.exit(1)

    out = {}
    counts = {}

    if "semanticSegments" in data:
        kept = excerpt_semantic_segments(
            data["semanticSegments"], start, end, trim_paths=not args.no_trim_paths
        )
        out["semanticSegments"] = kept
        counts["semanticSegments"] = f"{len(kept)} / {len(data['semanticSegments'])}"

    if "rawSignals" in data:
        kept = excerpt_raw_signals(data["rawSignals"], start, end)
        out["rawSignals"] = kept
        counts["rawSignals"] = f"{len(kept)} / {len(data['rawSignals'])}"

    if "locations" in data:
        kept = excerpt_locations(data["locations"], start, end)
        out["locations"] = kept
        counts["locations"] = f"{len(kept)} / {len(data['locations'])}"

    if args.include_profile and "userLocationProfile" in data:
        out["userLocationProfile"] = data["userLocationProfile"]

    if not counts:
        print(
            "Warning: input file has none of 'semanticSegments', 'rawSignals', or 'locations'.",
            file=sys.stderr,
        )

    with open(args.output, "w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(out, f, indent=2)
        else:
            json.dump(out, f, separators=(",", ":"))

    print(f"Excerpt window: {start.isoformat()} to {end.isoformat()}")
    for key, val in counts.items():
        print(f"  {key}: kept {val}")
    print(f"Wrote excerpt to {args.output}")


if __name__ == "__main__":
    main()
