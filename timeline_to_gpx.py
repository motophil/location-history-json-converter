#!/usr/bin/env python3
"""
Convert Google Timeline.json to a GPX track.

Supports two Google location-history formats:
  1. The modern on-device "Timeline.json" export (Settings > Location >
     Timeline > Export), which has top-level "semanticSegments" and/or
     "rawSignals" keys, with coordinates as "geo:lat,lng" strings.
  2. The older Google Takeout "Records.json" / "Location History.json"
     format, with a top-level "locations" array using latitudeE7/longitudeE7.

Usage:
    python timeline_to_gpx.py Timeline.json output.gpx \
        --start 2024-01-01 --end 2024-01-31 \
        --min-accuracy 50

Run with -h for all options.
"""

import argparse
import heapq
import json
import math
import sys
from datetime import datetime, timezone
from xml.sax.saxutils import escape


def parse_date(s, end_of_day=False):
    """Parse a date/day string into a timezone-aware UTC datetime."""
    s = s.strip()
    fmts = ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ"]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if fmt == "%Y-%m-%d" and end_of_day:
                dt = dt.replace(hour=23, minute=59, second=59)
            return dt
        except ValueError:
            continue
    raise ValueError(f"Could not parse date: {s!r}. Use YYYY-MM-DD or ISO 8601.")


def parse_geo_string(geo):
    """Parse a coordinate value into (lat, lon).

    Handles the plain string form ('geo:37.4219999,-122.0862462') as well as
    nested-object forms occasionally seen in real exports, e.g.
    {'latLng': 'geo:37.42,-122.09'} or {'lat': 37.42, 'lng': -122.09}.
    """
    if geo is None:
        return None

    # Some fields (e.g. topCandidate.placeLocation) are objects, not strings.
    if isinstance(geo, dict):
        for key in ("latLng", "LatLng", "point", "location"):
            if key in geo:
                return parse_geo_string(geo[key])
        if "lat" in geo and ("lng" in geo or "lon" in geo):
            try:
                return float(geo["lat"]), float(geo.get("lng", geo.get("lon")))
            except (ValueError, TypeError):
                return None
        return None

    if not isinstance(geo, str):
        return None

    geo = geo.strip()
    if geo.startswith("geo:"):
        geo = geo[4:]
    # Some geo strings carry a third component (e.g. altitude: "lat,lon,alt")
    # or stray whitespace/degree symbols; be tolerant and take the first two
    # numeric-looking fields.
    parts = [p.strip().rstrip("°") for p in geo.split(",")]
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except (ValueError, TypeError):
        return None


def parse_iso_time(t):
    if not t:
        return None
    t = t.strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


class Point:
    __slots__ = ("lat", "lon", "time", "accuracy", "altitude", "source", "fix")

    def __init__(self, lat, lon, time, accuracy=None, altitude=None, source="", fix=None):
        self.lat = lat
        self.lon = lon
        self.time = time
        self.accuracy = accuracy
        self.altitude = altitude
        self.source = source
        # How the position was obtained, when known: "GPS", "WIFI", "WIFI_ONLY", "CELL"
        self.fix = fix


def extract_points_new_format(data, verbose=False):
    """Extract points from the modern Timeline.json (semanticSegments + rawSignals)."""
    points = []
    stats = {
        "rawSignal_seen": 0, "rawSignal_ok": 0, "rawSignal_bad_latlng": 0, "rawSignal_bad_time": 0,
        "timelinePath_seen": 0, "timelinePath_ok": 0, "timelinePath_bad_latlng": 0, "timelinePath_bad_time": 0,
        "visit_seen": 0, "visit_ok": 0, "visit_bad_latlng": 0, "visit_bad_time": 0,
        "activity_seen": 0, "activity_ok": 0, "activity_bad_latlng": 0, "activity_bad_time": 0,
    }

    # rawSignals: raw GPS pings — the most useful ones for accuracy filtering.
    for signal in data.get("rawSignals", []):
        pos = signal.get("position")
        if not pos:
            continue
        stats["rawSignal_seen"] += 1
        latlng = parse_geo_string(pos.get("LatLng") or pos.get("latLng"))
        if not latlng:
            stats["rawSignal_bad_latlng"] += 1
            continue
        t = parse_iso_time(pos.get("timestamp"))
        if not t:
            stats["rawSignal_bad_time"] += 1
            continue
        accuracy = pos.get("accuracyMeters")
        altitude = pos.get("altitudeMeters")
        points.append(Point(latlng[0], latlng[1], t, accuracy, altitude, "rawSignal",
                            fix=pos.get("source")))
        stats["rawSignal_ok"] += 1

    # semanticSegments: timelinePath points (usually no accuracy figure),
    # plus visit/activity start-end points as a fallback.
    for seg in data.get("semanticSegments", []):
        for tp in seg.get("timelinePath", []):
            stats["timelinePath_seen"] += 1
            latlng = parse_geo_string(tp.get("point"))
            if not latlng:
                stats["timelinePath_bad_latlng"] += 1
                continue
            t = parse_iso_time(tp.get("time"))
            if not t:
                stats["timelinePath_bad_time"] += 1
                continue
            points.append(Point(latlng[0], latlng[1], t, None, None, "timelinePath"))
            stats["timelinePath_ok"] += 1

        visit = seg.get("visit")
        if visit:
            stats["visit_seen"] += 1
            candidate = visit.get("topCandidate", {})
            latlng = parse_geo_string(candidate.get("placeLocation"))
            t = parse_iso_time(seg.get("startTime"))
            if not latlng:
                stats["visit_bad_latlng"] += 1
            elif not t:
                stats["visit_bad_time"] += 1
            else:
                points.append(Point(latlng[0], latlng[1], t, None, None, "visit"))
                stats["visit_ok"] += 1

        activity = seg.get("activity")
        if activity:
            for key in ("start", "end"):
                stats["activity_seen"] += 1
                latlng = parse_geo_string(activity.get(key))
                t = parse_iso_time(seg.get("startTime") if key == "start" else seg.get("endTime"))
                if not latlng:
                    stats["activity_bad_latlng"] += 1
                elif not t:
                    stats["activity_bad_time"] += 1
                else:
                    points.append(Point(latlng[0], latlng[1], t, None, None, f"activity_{key}"))
                    stats["activity_ok"] += 1

    if verbose or not points:
        print("--- extraction breakdown ---", file=sys.stderr)
        for key, val in stats.items():
            if val:
                print(f"  {key}: {val}", file=sys.stderr)
        print("----------------------------", file=sys.stderr)

    return points


def extract_points_old_format(data):
    """Extract points from the classic Takeout 'locations' array format."""
    points = []
    for loc in data.get("locations", []):
        try:
            lat = loc["latitudeE7"] / 1e7
            lon = loc["longitudeE7"] / 1e7
        except KeyError:
            continue

        # timestamp can be 'timestamp' (ISO string) or 'timestampMs' (legacy)
        t = None
        if "timestamp" in loc:
            t = parse_iso_time(loc["timestamp"])
        elif "timestampMs" in loc:
            try:
                ms = int(loc["timestampMs"])
                t = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
            except (ValueError, TypeError):
                t = None
        if not t:
            continue

        accuracy = loc.get("accuracy")
        altitude = loc.get("altitude")
        points.append(Point(lat, lon, t, accuracy, altitude, "location"))

    return points


def extract_visit_windows(data):
    """Return (start, end) datetimes for every 'visit' (stay) segment.

    Google already tells us when you were dwelling somewhere. That is far more
    reliable than inferring a stay from gaps between points, because stray
    fixes keep trickling in during a stay (e.g. overnight at a hotel) and
    never leave a gap.
    """
    windows = []
    for seg in data.get("semanticSegments", []):
        if not seg.get("visit"):
            continue
        s = parse_iso_time(seg.get("startTime"))
        e = parse_iso_time(seg.get("endTime"))
        if s and e and e > s:
            windows.append((s, e))
    windows.sort()
    return windows


def load_points_and_windows(path, verbose=False):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    windows = []
    if "semanticSegments" in data or "rawSignals" in data:
        points = extract_points_new_format(data, verbose=verbose)
        windows = extract_visit_windows(data)
    elif "locations" in data:
        points = extract_points_old_format(data)
    else:
        raise ValueError(
            "Unrecognized file structure: expected 'semanticSegments'/'rawSignals' "
            "(modern Timeline.json) or 'locations' (Takeout Records.json)."
        )

    return points, windows


def load_points(path, verbose=False):
    return load_points_and_windows(path, verbose=verbose)[0]


def filter_points(points, start, end, min_accuracy, require_accuracy):
    filtered = []
    for p in points:
        if start and p.time < start:
            continue
        if end and p.time > end:
            continue
        if min_accuracy is not None:
            if p.accuracy is None:
                if require_accuracy:
                    continue
                # no accuracy info available: keep it, since we can't judge it
            elif p.accuracy > min_accuracy:
                continue
        filtered.append(p)
    return filtered


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points, in kilometers."""
    from math import radians, sin, cos, asin, sqrt

    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(a))


SEGMENT_ESTIMATE_SOURCES = ("activity_start", "activity_end", "visit")
REAL_READING_SOURCES = ("rawSignal", "timelinePath", "location")


def drop_segment_estimates(points, window_minutes=5.0):
    """Treat activity start/end and visit anchors as fallback-only.

    Those points are Google's *estimate* of where a segment began or ended.
    They carry no accuracy value and can be tens of kilometers away from the
    real GPS fix recorded moments before or after. Because the discrepancy
    plays out over a minute or two, it looks like physically possible (if
    absurdly fast) travel, so it produces a spike out and back. Whenever a
    real reading exists within window_minutes, trust that and drop the
    estimate; keep the estimate only when it is all we have.
    """
    window_s = window_minutes * 60.0
    real_times = sorted(p.time for p in points if p.source in REAL_READING_SOURCES)

    import bisect

    kept = []
    for p in points:
        if p.source not in SEGMENT_ESTIMATE_SOURCES:
            kept.append(p)
            continue
        i = bisect.bisect_left(real_times, p.time)
        near = False
        for j in (i - 1, i):
            if 0 <= j < len(real_times) and abs((real_times[j] - p.time).total_seconds()) <= window_s:
                near = True
                break
        if not near:
            kept.append(p)
    return kept


def deduplicate_points(points, time_window_s=120.0, dist_threshold_m=150.0):
    """Collapse near-duplicate readings that come from different sources.

    Google's Timeline.json frequently repeats the same underlying GPS fix
    across sources — e.g. a semanticSegments 'timelinePath' point that is a
    coarser, independently-timestamped resampling of a 'rawSignals' fix a
    few seconds away. When all sources are merged and sorted by time, these
    near-duplicates can land next to each other with a timestamp that
    doesn't quite match their true moment, producing a spurious tiny-time /
    large(or zero)-distance jump that has nothing to do with real movement.
    This removes that noise before track-splitting sees it, keeping the more
    informative reading (the one with an accuracy value, i.e. a raw signal)
    whenever two points are close in both time and space.
    """
    points = sorted(points, key=lambda p: p.time)
    kept = []
    for p in points:
        merged = False
        for i in range(len(kept) - 1, -1, -1):
            q = kept[i]
            if (p.time - q.time).total_seconds() > time_window_s:
                break  # kept is time-ordered, so nothing earlier can match either
            if haversine_km(p.lat, p.lon, q.lat, q.lon) * 1000.0 <= dist_threshold_m:
                # Prefer the reading that carries an accuracy value, and of two
                # such readings the more accurate one.
                if p.accuracy is not None and (q.accuracy is None or p.accuracy < q.accuracy):
                    kept[i] = p
                merged = True
                break
        if not merged:
            kept.append(p)

    kept.sort(key=lambda p: p.time)
    return kept


NETWORK_FIXES = ("WIFI", "WIFI_ONLY", "CELL")


def remove_spikes(points, max_window_minutes=15.0, min_spike_distance_m=1000.0,
                  return_ratio=0.25, max_run=3, network_window_minutes=60.0,
                  network_max_run=8):
    """Remove out-and-back outlier fixes (e.g. mis-located WiFi fixes).

    A bad fix can pass every other check: its accuracy value may look fine,
    and because the jump out and back plays out over minutes, the implied
    acceleration is perfectly plausible. What gives it away is the shape: the
    path leaves A, visits far-away points, then returns to (almost) A at C, so
    the run in between is a detour the surrounding readings don't support.

    For a run of consecutive points between A (last kept point) and C (the
    next point), the run is dropped when:
      - C is close to A compared with how far the run strays (return_ratio), and
      - every point of the run is at least min_spike_distance_m from both A
        and C, and
      - A to C fits in the time window.
    Ordinary runs may have up to max_run points and span max_window_minutes.
    Runs made only of network-derived fixes (WiFi/cell), which are known to be
    occasionally mis-located by kilometers, get the longer network_window_minutes
    and up to network_max_run points.
    Legitimate out-and-back trips that exceed these limits are untouched.
    """
    points = sorted(points, key=lambda p: p.time)
    n = len(points)
    if n < 3:
        return points

    def dist(a, b):
        return haversine_km(a.lat, a.lon, b.lat, b.lon) * 1000.0

    kept = [points[0]]
    i = 1
    while i < n:
        a = kept[-1]
        dropped = False
        for m in range(1, network_max_run + 1):
            if i + m >= n:
                break
            run = points[i:i + m]
            network_only = all(r.fix in NETWORK_FIXES for r in run)
            if m > max_run and not network_only:
                break
            window = network_window_minutes if network_only else max_window_minutes
            c = points[i + m]
            if (c.time - a.time).total_seconds() > window * 60.0:
                continue
            far = all(dist(a, r) >= min_spike_distance_m and dist(r, c) >= min_spike_distance_m
                      for r in run)
            if not far:
                continue
            shortest = min(min(dist(a, r), dist(r, c)) for r in run)
            if dist(a, c) <= return_ratio * shortest:
                i += m  # drop the run; A stays the reference
                dropped = True
                break
        if not dropped:
            kept.append(points[i])
            i += 1
    return kept


def split_into_tracks(points, max_gap_minutes=15.0, max_acceleration_mps2=10.0,
                       speed_window=5, noise_floor_m=50.0, visit_windows=None):
    """Split a chronological point list into separate trips.

    A new track starts whenever any of these holds:
      - the time since the previous point exceeds max_gap_minutes (a real
        pause — you were stationary or untracked for a while), or
      - one point is inside a long stay (a Google 'visit' lasting at least
        max_gap_minutes) and the other is not, or they are in different stays.
        This catches e.g. a night at a hotel, where stray fixes keep arriving
        every few minutes and so no time gap ever appears, or
      - reaching the new point's implied speed from the track's own recent
        speeds would require physically implausible acceleration.

    The acceleration check is relative to the track's *own* recent speeds
    and normalized by how much time actually elapsed, not a fixed speed
    ceiling. That means sustained fast travel (a flight, a train) is fine as
    long as the speed changes gradually relative to elapsed time — a plane
    accelerating over minutes is physically plausible even though its speed
    is high. Only a jump that would require impossible acceleration (a GPS
    glitch, a mis-geocoded point) triggers a split. Tiny movements within
    the points' own accuracy radius are treated as noise/no movement so
    stationary GPS jitter doesn't cause spurious splits.
    """
    points = sorted(points, key=lambda p: p.time)
    if not points:
        return []

    min_stay_s = max_gap_minutes * 60.0
    stays = [(s, e) for (s, e) in (visit_windows or []) if (e - s).total_seconds() >= min_stay_s]

    def stay_id(p):
        for idx, (s, e) in enumerate(stays):
            if s <= p.time <= e:
                return idx
        return None

    tracks = [[points[0]]]
    recent_speeds = []  # rolling window of recent speeds (m/s) within the current track
    prev_stay = stay_id(points[0]) if stays else None

    for prev, cur in zip(points, points[1:]):
        dt_s = (cur.time - prev.time).total_seconds()
        gap_minutes = dt_s / 60.0
        new_track = False
        v_cur = None

        cur_stay = stay_id(cur) if stays else None
        if cur_stay != prev_stay:
            new_track = True
        elif dt_s <= 0:
            # Duplicate or out-of-order timestamp; keep it with the current track.
            pass
        elif gap_minutes > max_gap_minutes:
            new_track = True
        else:
            dist_m = haversine_km(prev.lat, prev.lon, cur.lat, cur.lon) * 1000.0
            noise_floor = noise_floor_m
            if prev.accuracy:
                noise_floor = max(noise_floor, prev.accuracy)
            if cur.accuracy:
                noise_floor = max(noise_floor, cur.accuracy)
            v_cur = 0.0 if dist_m <= noise_floor else dist_m / dt_s  # meters/second

            if recent_speeds:
                reference_speed = sorted(recent_speeds)[len(recent_speeds) // 2]  # median
                acceleration = abs(v_cur - reference_speed) / dt_s
                if acceleration > max_acceleration_mps2:
                    new_track = True

        prev_stay = cur_stay

        if new_track:
            tracks.append([cur])
            recent_speeds = []
        else:
            tracks[-1].append(cur)
            if v_cur is not None:
                recent_speeds.append(v_cur)
                if len(recent_speeds) > speed_window:
                    recent_speeds.pop(0)

    return tracks


def render_gpx(tracks, base_name="Google Timeline export"):
    """Build the GPX 1.1 XML text for one or more tracks, without writing to disk.

    tracks: a list of point-lists (each an already-chronological trip), or a
    single flat list of points (treated as one track) for convenience.
    """
    if tracks and isinstance(tracks[0], Point):
        tracks = [tracks]  # a flat list of points was passed in

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        '<gpx version="1.1" creator="timeline_to_gpx.py" '
        'xmlns="http://www.topografix.com/GPX/1/1" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:schemaLocation="http://www.topografix.com/GPX/1/1 '
        'http://www.topografix.com/GPX/1/1/gpx.xsd">'
    )

    multi = len(tracks) > 1
    for i, track_points in enumerate(tracks, start=1):
        track_points = sorted(track_points, key=lambda p: p.time)
        if multi:
            start_str = track_points[0].time.strftime("%Y-%m-%d %H:%M")
            end_str = track_points[-1].time.strftime("%Y-%m-%d %H:%M")
            name = f"{base_name} — Track {i}: {start_str} to {end_str}"
        else:
            name = base_name

        lines.append(f"  <trk><name>{escape(name)}</name><trkseg>")
        for p in track_points:
            time_str = p.time.strftime("%Y-%m-%dT%H:%M:%SZ")
            lines.append(f'    <trkpt lat="{p.lat:.7f}" lon="{p.lon:.7f}">')
            if p.altitude is not None:
                lines.append(f"      <ele>{p.altitude}</ele>")
            lines.append(f"      <time>{time_str}</time>")
            if p.accuracy is not None:
                lines.append(f"      <extensions><accuracy>{p.accuracy}</accuracy></extensions>")
            lines.append("    </trkpt>")
        lines.append("  </trkseg></trk>")

    lines.append("</gpx>")

    return "\n".join(lines)


def write_gpx(tracks, out_path, base_name="Google Timeline export"):
    """Render one or more tracks to GPX and write them to out_path."""
    text = render_gpx(tracks, base_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)


def _local_xy(lat, lon, lat0):
    """Project (lat, lon) to local flat-earth meters, centered near lat0."""
    r = 6371000.0
    return math.radians(lon) * math.cos(math.radians(lat0)) * r, math.radians(lat) * r


def _triangle_area_m2(a, b, c):
    """Shoelace-formula area (m^2) of the triangle a-b-c, given as local-meter (x, y) pairs."""
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2.0


def _point_area(track, prev, nxt, i):
    """Effective area (Visvalingam-Whyatt) of dropping point i given its current neighbors."""
    p, c = prev[i], nxt[i]
    if p == -1 or c == -1:
        return float("inf")  # endpoints are never dropped
    lat0 = track[i].lat
    a = _local_xy(track[p].lat, track[p].lon, lat0)
    b = _local_xy(track[i].lat, track[i].lon, lat0)
    c_xy = _local_xy(track[c].lat, track[c].lon, lat0)
    return _triangle_area_m2(a, b, c_xy)


def simplify_tracks(tracks, n_remove):
    """Drop n_remove points total across all tracks (Visvalingam-Whyatt).

    At each step, whichever remaining interior point changes its track's shape
    the least — the smallest-area triangle formed with its current neighbors —
    is dropped next, and its neighbors' areas are recomputed. Track endpoints
    are never dropped, so every track keeps its start and end.
    """
    if n_remove <= 0:
        return tracks

    heap = []  # (area, track_idx, point_idx, version)
    prevs, nexts, alives, versions = [], [], [], []
    for t_idx, track in enumerate(tracks):
        n = len(track)
        prev = list(range(-1, n - 1))
        nxt = list(range(1, n + 1))
        nxt[-1] = -1
        prevs.append(prev)
        nexts.append(nxt)
        alives.append([True] * n)
        versions.append([0] * n)
        for i in range(1, n - 1):
            heapq.heappush(heap, (_point_area(track, prev, nxt, i), t_idx, i, 0))

    removed_flags = [[False] * len(t) for t in tracks]
    removed = 0
    while heap and removed < n_remove:
        area, t_idx, i, ver = heapq.heappop(heap)
        alive, version = alives[t_idx], versions[t_idx]
        if not alive[i] or ver != version[i]:
            continue  # stale entry left over from a neighbor update

        track, prev, nxt = tracks[t_idx], prevs[t_idx], nexts[t_idx]
        p, c = prev[i], nxt[i]
        alive[i] = False
        removed_flags[t_idx][i] = True
        nxt[p] = c
        prev[c] = p
        removed += 1

        n = len(track)
        for j in (p, c):
            if j in (0, n - 1):
                continue  # endpoints are never re-scored or dropped
            version[j] += 1
            heapq.heappush(heap, (_point_area(track, prev, nxt, j), t_idx, j, version[j]))

    return [
        [pt for idx, pt in enumerate(track) if not removed_flags[t_idx][idx]]
        for t_idx, track in enumerate(tracks)
    ]


def enforce_max_gpx_size(tracks, base_name, max_bytes, verbose=False):
    """Shrink tracks, if needed, so the rendered GPX fits within max_bytes.

    Repeatedly estimates how many points to drop from the rendered size, drops
    that many via simplify_tracks(), and re-measures, since actual bytes-per-
    point varies slightly (e.g. altitude/accuracy tags aren't on every point).
    Stops once the file fits or every track is down to its minimum of 2 points.
    """
    text = render_gpx(tracks, base_name)
    size = len(text.encode("utf-8"))
    if size <= max_bytes:
        return tracks

    total_points = sum(len(t) for t in tracks)
    removable = sum(max(0, len(t) - 2) for t in tracks)

    for _ in range(20):
        if size <= max_bytes or removable <= 0:
            break
        avg_bytes_per_point = size / total_points
        excess = size - max_bytes
        n_remove = min(removable, max(1, math.ceil(excess / avg_bytes_per_point * 1.05)))

        tracks = simplify_tracks(tracks, n_remove)
        total_points = sum(len(t) for t in tracks)
        removable = sum(max(0, len(t) - 2) for t in tracks)

        text = render_gpx(tracks, base_name)
        size = len(text.encode("utf-8"))

    if verbose:
        print(f"Simplified to {total_points} point(s) to target {max_bytes} bytes "
              f"(final size: {size} bytes)", file=sys.stderr)
    if size > max_bytes:
        print(f"Warning: could not shrink the GPX below {max_bytes} bytes "
              f"(reached {size} bytes with every track at its minimum of 2 points)", file=sys.stderr)

    return tracks


def main():
    parser = argparse.ArgumentParser(
        description="Convert Google Timeline.json to a GPX track, with time-range and accuracy filtering."
    )
    parser.add_argument("input", help="Path to Timeline.json (or Takeout Records.json)")
    parser.add_argument("output", help="Path to write the .gpx file")
    parser.add_argument("-s", "--start", help="Start date/time, e.g. 2024-01-01 (inclusive)")
    parser.add_argument("-e", "--end", help="End date/time, e.g. 2024-01-31 (inclusive)")
    parser.add_argument(
        "-a", "--min-accuracy",
        type=float,
        default=200.0,
        help="Discard points whose accuracy is worse (a larger number of meters) than this. "
        "Default: 200. Pass a negative number to disable accuracy filtering entirely.",
    )
    parser.add_argument(
        "-r", "--require-accuracy",
        action="store_true",
        help="When accuracy filtering is active, also discard points that have no accuracy "
        "value at all (by default such points are kept, since most 'semanticSegments' points "
        "carry none)",
    )
    parser.add_argument(
        "-n", "--name", default="Google Timeline export", help="Track name stored inside the GPX file"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Always print a breakdown of how many points were extracted/skipped per source",
    )
    parser.add_argument(
        "-g", "--max-gap-minutes",
        type=float,
        default=15.0,
        help="Start a new track after a pause of at least this many minutes (default: 15). "
        "Also the minimum length of a Google 'visit' (stay) that forces a track break.",
    )
    parser.add_argument(
        "-E", "--estimate-window-minutes",
        type=float,
        default=5.0,
        help="Google's activity start/end and visit anchor points are only estimates. Drop them "
        "when a real reading (rawSignal/timelinePath) exists within this many minutes "
        "(default: 5). Set to 0 to keep all of them.",
    )
    parser.add_argument(
        "-A", "--max-acceleration",
        type=float,
        default=30.0,
        dest="max_acceleration_mps2",
        help="Start a new track when reaching a point's speed from the track's own recent "
        "speeds would require more than this much acceleration, in m/s^2 (default: 30, "
        "roughly 3g). This replaces any fixed speed ceiling, so sustained fast travel (a "
        "flight, a train) that changes speed gradually relative to elapsed time won't "
        "trigger a split — only implausible jumps will.",
    )
    parser.add_argument(
        "-m", "--min-track-points",
        type=int,
        default=2,
        help="Drop resulting tracks with fewer than this many points (default: 2); this also "
        "cleans up single-point GPS glitches isolated by the acceleration check",
    )
    parser.add_argument(
        "-p", "--spike-distance-m",
        type=float,
        default=1000.0,
        help="Remove isolated out-and-back outlier fixes (e.g. a mis-located WiFi fix) that "
        "stray at least this many meters from the surrounding path and then return within "
        "--max-gap-minutes. Default: 1000. Set to 0 to disable.",
    )
    parser.add_argument(
        "-M", "--max-gpx-size",
        type=float,
        default=4.9,
        dest="max_gpx_size_mb",
        help="Target maximum size of the output .gpx file, in megabytes (default: 4.9). If "
        "exceeded, track points are progressively dropped as a last step \u2014 always whichever "
        "point changes its track's shape the least (Visvalingam-Whyatt) \u2014 until the file fits "
        "or every track is down to its minimum of 2 points. Set to 0 to disable.",
    )

    args = parser.parse_args()

    start = parse_date(args.start) if args.start else None
    end = parse_date(args.end, end_of_day=True) if args.end else None
    min_accuracy = None if args.min_accuracy is not None and args.min_accuracy < 0 else args.min_accuracy

    try:
        points, visit_windows = load_points_and_windows(args.input, verbose=args.verbose)
    except (json.JSONDecodeError, ValueError, OSError) as e:
        print(f"Error reading {args.input}: {e}", file=sys.stderr)
        sys.exit(1)

    if not points:
        print("No location points found in the input file.", file=sys.stderr)
        sys.exit(1)

    # 1. Restrict to the requested time range only (accuracy comes later).
    filtered = filter_points(points, start, end, None, False)

    if not filtered:
        print("No points left in the requested time range.", file=sys.stderr)
        sys.exit(1)

    # 2. De-duplicate BEFORE the accuracy filter. A timelinePath point is often a
    #    resampled copy of a raw fix but carries no accuracy value, so it would
    #    sail through the accuracy filter even when its raw twin (say a 800 m
    #    cell-tower fix 20 km off) is rejected. Merging first lets the copy
    #    inherit the raw fix's accuracy, so both are filtered together.
    before_dedup = len(filtered)
    filtered = deduplicate_points(filtered)
    removed = before_dedup - len(filtered)
    if removed and args.verbose:
        print(f"De-duplicated {removed} near-identical cross-source point(s)", file=sys.stderr)

    # 3. Accuracy filter.
    filtered = filter_points(filtered, None, None, min_accuracy, args.require_accuracy)

    if not filtered:
        print("No points left after filtering — check your date range / accuracy threshold.", file=sys.stderr)
        sys.exit(1)

    # 4. Google's activity/visit anchors are only fallbacks for real readings.
    if args.estimate_window_minutes > 0:
        before_est = len(filtered)
        filtered = drop_segment_estimates(filtered, args.estimate_window_minutes)
        if before_est != len(filtered) and args.verbose:
            print(f"Dropped {before_est - len(filtered)} estimated segment anchor point(s) "
                  f"that had real readings nearby", file=sys.stderr)

    if args.spike_distance_m > 0:
        before_spike = len(filtered)
        filtered = remove_spikes(filtered, args.max_gap_minutes, args.spike_distance_m)
        if before_spike != len(filtered) and args.verbose:
            print(f"Removed {before_spike - len(filtered)} out-and-back outlier point(s)", file=sys.stderr)

    tracks = split_into_tracks(
        filtered, args.max_gap_minutes, args.max_acceleration_mps2,
        visit_windows=visit_windows,
    )
    before = len(tracks)
    tracks = [t for t in tracks if len(t) >= args.min_track_points]
    dropped = before - len(tracks)
    if dropped:
        print(f"Dropped {dropped} tiny track(s) with fewer than {args.min_track_points} points", file=sys.stderr)

    if not tracks:
        print("No tracks left after splitting/filtering — try lowering --min-track-points.", file=sys.stderr)
        sys.exit(1)

    # 5. Last step: shrink the file, if needed, to fit within --max-gpx-size.
    if args.max_gpx_size_mb > 0:
        max_bytes = int(args.max_gpx_size_mb * 1_000_000)
        tracks = enforce_max_gpx_size(tracks, args.name, max_bytes, verbose=args.verbose)

    write_gpx(tracks, args.output, args.name)
    total_written = sum(len(t) for t in tracks)
    print(f"Wrote {len(tracks)} track(s), {total_written} points (of {len(points)} total) to {args.output}")


if __name__ == "__main__":
    main()




