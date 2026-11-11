"""Summarise exact-search timings by number of bands k, band count and device.

Reads the timing log that exact_search.py appends to (by default timings.csv
next to the --out directory) and prints, for each k: chunks and subsets timed,
total compute time, subsets per second, and the projected time of the full
search. Use --csv to save the table, for example for the paper's RQ4 results:
    python scripts/timing_report.py results/exact/timings.csv --csv results/timing_summary.csv
"""

import argparse
import csv
import sys
from pathlib import Path

from bandopt.timing import format_duration, read_rows, summarize_search_log


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("logs", type=Path, nargs="+", help="One or more timings.csv files")
    parser.add_argument("--csv", type=Path, help="Also write the summary to this CSV file")
    args = parser.parse_args()

    rows = []
    for path in args.logs:
        if not path.exists():
            sys.exit(f"{path} does not exist")
        rows.extend(read_rows(path))
    if not rows:
        sys.exit("No timings recorded yet")
    summaries = summarize_search_log(rows)

    print(
        f"{'k':>2} {'bands':>5} {'chunks':>6} {'subsets':>16} {'compute':>10} "
        f"{'subsets/s':>14} {'full search':>12}  device"
    )
    for s in summaries:
        print(
            f"{s['k']:>2} {s['n_bands']:>5} {s['chunks']:>6} {s['subsets']:>16,} "
            f"{format_duration(s['seconds']):>10} {s['subsets_per_s']:>14,.0f} "
            f"{format_duration(s['projected_seconds']):>12}  {s['device']}"
        )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
