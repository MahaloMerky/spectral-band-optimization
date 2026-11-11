"""Record and summarise how long exact-search work takes."""

import csv
import math
from datetime import datetime, timezone
from pathlib import Path

SEARCH_LOG_FIELDS = (
    "recorded_at", "k", "n_bands", "chunk", "subsets", "seconds", "subsets_per_s",
    "device", "batch_size", "stats_fingerprint", "out_dir",
)


def timestamp():
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_rows(path, rows, fields):
    """Append dict rows to a CSV log, writing the header if the file is new.

    Refuses to append to a file whose header differs, so one log never mixes
    two layouts.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fields)
    is_new = not path.exists() or path.stat().st_size == 0
    if not is_new:
        with path.open(newline="") as file:
            header = next(csv.reader(file), [])
        if header != fields:
            raise ValueError(f"{path} has columns {header}, expected {fields}. Use another log file.")
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with Path(path).open(newline="") as file:
        return list(csv.DictReader(file))


def summarize_search_log(rows):
    """Summarise exact-search chunk timings per (k, n_bands, device).

    A chunk that was run more than once (for example after its file was deleted)
    counts only with its latest row, so reruns do not double the totals. Rows
    from different statistics files (folds) are pooled, because throughput
    depends on k, n_bands and the device, not on the data.

    Returns a list of dicts, sorted by k, n_bands and device, with the number of
    chunks and subsets timed, their total seconds, the mean subsets per second,
    the size of the full search space, and the projected time to search it.
    """
    latest = {}
    for row in rows:
        key = (row["stats_fingerprint"], int(row["k"]), int(row["n_bands"]),
               row["device"], int(row["chunk"]))
        latest[key] = row

    groups = {}
    for (_, k, n_bands, device, _), row in latest.items():
        group = groups.setdefault((k, n_bands, device), {"chunks": 0, "subsets": 0, "seconds": 0.0})
        group["chunks"] += 1
        group["subsets"] += int(row["subsets"])
        group["seconds"] += float(row["seconds"])

    summaries = []
    for (k, n_bands, device), group in sorted(groups.items()):
        rate = group["subsets"] / group["seconds"] if group["seconds"] > 0 else math.nan
        search_space = math.comb(n_bands, k)
        summaries.append({
            "k": k,
            "n_bands": n_bands,
            "device": device,
            "chunks": group["chunks"],
            "subsets": group["subsets"],
            "seconds": group["seconds"],
            "subsets_per_s": rate,
            "search_space": search_space,
            "projected_seconds": search_space / rate if rate > 0 else math.nan,
        })
    return summaries


def format_duration(seconds):
    """Format seconds as a short duration such as "42.0 s", "3.5 min" or "1.2 d"."""
    if math.isnan(seconds):
        return "n/a"
    for unit, size in (("d", 86400), ("h", 3600), ("min", 60)):
        if seconds >= size:
            return f"{seconds / size:.1f} {unit}"
    return f"{seconds:.1f} s"
