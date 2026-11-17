"""Download the inputs listed in a scene config: EMIT files, WorldCover tiles, solar spectrum.

For every [[scene]] in the config (see configs/scenes.example.toml) this fetches
the three EMIT files named there (L1B radiance, L1B observation geometry, L2A
mask) into the folders given by their paths. It then fetches every ESA
WorldCover 2021 tile that overlaps any scene into the folder of the first
`worldcover_tiles` pattern, and the TSIS-1 HSRS solar spectrum if it is missing.

Files that already exist with the expected size are skipped, so an interrupted
download is resumed by running the same command again. EMIT downloads need an
Earthdata login saved in ~/.netrc or ~/_netrc (see data/README.md).

    python scripts/download_data.py --config configs/scenes.toml --dry-run
    python scripts/download_data.py --config configs/scenes.toml
"""

import argparse
import math
import os
import re
import sys
import time
import tomllib
from pathlib import Path

import requests

WORLDCOVER_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)
SOLAR_URL = (
    "https://lasp.colorado.edu/lisird/resources/lasp/hsrs/v2/"
    "hybrid_reference_spectrum_p1nm_resolution_c2022-11-30_with_unc.nc"
)
STAMP = re.compile(r"EMIT_L1B_RAD_001_(\d{8}T\d{6}_\d+_\d+)\.nc$")


def resolve(base, path):
    path = Path(path)
    return path if path.is_absolute() else base / path


def granule_files(short_name, granule_name):
    """Return the granule's metadata and {file name: (url, size in bytes)}."""
    import earthaccess

    found = earthaccess.search_data(short_name=short_name, granule_name=granule_name, count=1)
    if not found:
        sys.exit(f"{granule_name} was not found in {short_name}")
    granule = found[0]
    sizes = {
        item["Name"]: item.get("SizeInBytes")
        for item in granule["umm"]["DataGranule"]["ArchiveAndDistributionInformation"]
    }
    urls = {url.rsplit("/", 1)[-1]: url for url in granule.data_links()}
    return granule, {name: (url, sizes.get(name)) for name, url in urls.items()}


def footprint(granule):
    """Return (west, south, east, north) of a granule's spatial extent."""
    geometry = granule["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]
    lons, lats = [], []
    for polygon in geometry.get("GPolygons", []):
        for point in polygon["Boundary"]["Points"]:
            lons.append(point["Longitude"])
            lats.append(point["Latitude"])
    for box in geometry.get("BoundingRectangles", []):
        lons += [box["WestBoundingCoordinate"], box["EastBoundingCoordinate"]]
        lats += [box["SouthBoundingCoordinate"], box["NorthBoundingCoordinate"]]
    if not lons:
        sys.exit(f"{granule['umm']['GranuleUR']} has no footprint in its metadata")
    return min(lons), min(lats), max(lons), max(lats)


def worldcover_tiles(west, south, east, north):
    """Names of the 3x3 degree WorldCover tiles, such as N33W114, covering a box."""
    tiles = []
    for lat in range(3 * math.floor(south / 3), 3 * math.floor(north / 3) + 1, 3):
        for lon in range(3 * math.floor(west / 3), 3 * math.floor(east / 3) + 1, 3):
            tiles.append(
                f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
            )
    return tiles


RETRIES = 6


def download(url, path, session, expected_size=None):
    """Download url to path, retrying server errors and dropped connections.

    Returns False on a 404. Waits 5, 10, 20, ... seconds between attempts.
    """
    for attempt in range(1, RETRIES + 1):
        try:
            return _download_once(url, path, session, expected_size)
        except (requests.ConnectionError, requests.Timeout,
                requests.exceptions.ChunkedEncodingError, ServerError) as error:
            if attempt == RETRIES:
                raise
            wait = 5 * 2 ** (attempt - 1)
            print(f"  {error}; retrying in {wait} s ({attempt}/{RETRIES - 1})", flush=True)
            time.sleep(wait)


class ServerError(Exception):
    """A 5xx response, which is usually temporary."""


def _download_once(url, path, session, expected_size):
    partial = path.with_name(path.name + ".partial")
    with session.get(url, stream=True, timeout=60) as response:
        if response.status_code == 404:
            return False
        if response.status_code >= 500:
            raise ServerError(f"{response.status_code} {response.reason}")
        response.raise_for_status()
        total = expected_size or int(response.headers.get("Content-Length", 0)) or None
        done = 0
        with partial.open("wb") as file:
            for block in response.iter_content(chunk_size=1 << 20):
                file.write(block)
                done += len(block)
                if total and sys.stderr.isatty():
                    sys.stderr.write(f"\r  {path.name}: {100 * done / total:5.1f}% of {total / 1e9:.2f} GB")
                    sys.stderr.flush()
        if total and sys.stderr.isatty():
            sys.stderr.write("\n")
    if expected_size and partial.stat().st_size != expected_size:
        partial.unlink()
        raise requests.exceptions.ChunkedEncodingError(
            f"{path.name}: got {done} bytes, expected {expected_size}"
        )
    os.replace(partial, path)
    return True


def is_complete(path, expected_size):
    return path.exists() and (expected_size is None or path.stat().st_size == expected_size)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", help="Only these scene ids")
    parser.add_argument("--dry-run", action="store_true", help="List what would be downloaded")
    args = parser.parse_args()

    with args.config.open("rb") as file:
        config = tomllib.load(file)
    base = args.config.parent
    scenes = [s for s in config["scene"] if not args.scenes or s["id"] in args.scenes]
    if args.scenes and len(scenes) != len(set(args.scenes)):
        sys.exit(f"Unknown scene id in {args.scenes}")

    import earthaccess

    if not args.dry_run:
        auth = earthaccess.login(strategy="netrc")
        if not auth.authenticated:
            sys.exit("Earthdata login failed. Save it first (see data/README.md).")
        session = earthaccess.get_requests_https_session()
    plain = requests.Session()

    jobs, tiles = [], set()
    for scene in scenes:
        radiance = resolve(base, scene["radiance"])
        match = STAMP.search(radiance.name)
        if not match:
            sys.exit(f"{scene['id']}: cannot read the granule from {radiance.name}")
        stamp = match.group(1)
        l1b, l1b_files = granule_files("EMITL1BRAD", f"EMIT_L1B_RAD_001_{stamp}")
        _, l2a_files = granule_files("EMITL2ARFL", f"EMIT_L2A_RFL_001_{stamp}")
        for key, files in (("radiance", l1b_files), ("observation", l1b_files), ("mask", l2a_files)):
            path = resolve(base, scene[key])
            if path.name not in files:
                sys.exit(f"{scene['id']}: {path.name} is not part of granule {stamp}")
            url, size = files[path.name]
            jobs.append((scene["id"], path, url, size))
        tiles.update(worldcover_tiles(*footprint(l1b)))

    worldcover_dir = resolve(base, config["worldcover_tiles"][0]).parent
    solar = resolve(base, config["solar_spectrum"])

    pending = [job for job in jobs if not is_complete(job[1], job[3])]
    pending_tiles = [
        t for t in sorted(tiles)
        if not (worldcover_dir / Path(WORLDCOVER_URL.format(tile=t)).name).exists()
    ]
    size = sum(job[3] or 0 for job in pending)
    print(
        f"{len(scenes)} scenes: {len(pending)} of {len(jobs)} EMIT files to fetch "
        f"({size / 1e9:.1f} GB), {len(pending_tiles)} of {len(tiles)} WorldCover tiles, "
        f"solar spectrum {'present' if solar.exists() else 'missing'}"
    )
    if args.dry_run:
        for scene_id, path, _, job_size in pending:
            print(f"  {scene_id}: {path} ({(job_size or 0) / 1e9:.2f} GB)")
        print(f"  WorldCover tiles: {' '.join(pending_tiles) or 'none'}")
        return

    for i, (scene_id, path, url, job_size) in enumerate(pending, 1):
        print(f"[{i}/{len(pending)}] {scene_id}: {path.name}", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not download(url, path, session, job_size):
            sys.exit(f"{url} returned 404")

    worldcover_dir.mkdir(parents=True, exist_ok=True)
    for i, tile in enumerate(pending_tiles, 1):
        url = WORLDCOVER_URL.format(tile=tile)
        print(f"[{i}/{len(pending_tiles)}] WorldCover {tile}", flush=True)
        if not download(url, worldcover_dir / Path(url).name, plain):
            print(f"  {tile} does not exist (open ocean), skipped")

    if not solar.exists():
        print("Solar spectrum", flush=True)
        solar.parent.mkdir(parents=True, exist_ok=True)
        download(SOLAR_URL, solar, plain)

    print("All files present.")


if __name__ == "__main__":
    main()
