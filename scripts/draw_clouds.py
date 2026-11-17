"""Draw liquid- and ice-cloud polygons on an EMIT scene and save them as GeoJSON.

Opens one scene from the config on its map grid and writes the polygons in the
format build_pixel_table.py reads: GeoJSON in longitude/latitude with a "label"
of "liquid_cloud" or "ice_cloud". Polygons already saved for the scene are
loaded, so you can stop and continue later.

    python scripts/draw_clouds.py --config configs/scenes.toml --scene india_storm

Views (press v to switch):
  cloud phase   red = 0.65 um, green = 1.6 um, blue = 2.2 um reflectance.
                Liquid cloud reflects at all three and looks WHITE. Ice absorbs
                at 1.6 and 2.2 um and looks RED / PINK. Vegetation looks green.
  true colour   ordinary RGB.

Controls:
  left click        add a corner to the current polygon
  right click       remove the last corner
  l                 finish the polygon as liquid cloud (drawn in blue)
  i                 finish the polygon as ice cloud (drawn in red)
  Esc               throw away the polygon you are drawing
  Backspace/Delete  delete the last finished polygon
  m                 show or hide the L2A mask outlines (cyan = cloud, magenta = cirrus)
  s                 save now (the file is also saved when you close the window)
  Zoom and pan with the toolbar buttons; clicks don't add corners while one is on.

Only pixels inside a polygon AND flagged by L2A become cloud samples (liquid: the
cloud flag, cyan; ice: the cloud or cirrus flag, magenta),
so draw generously around a cloud of one phase rather than tracing its edge.
Keep polygons of different phases apart: overlapping areas are discarded.
"""

import argparse
import json
import sys
import tomllib
from pathlib import Path

import numpy as np

from bandopt import emit
from bandopt.toa import (
    W_PER_M2_TO_UW_PER_CM2,
    band_solar_irradiance,
    load_solar_spectrum,
    radiance_to_toa_reflectance,
)

PHASE_NM = (650.0, 1610.0, 2200.0)
TRUE_COLOUR_NM = (640.0, 550.0, 460.0)
LABEL_COLOURS = {"liquid_cloud": "#3b82f6", "ice_cloud": "#ef4444"}


def resolve(base, path):
    path = Path(path)
    return path if path.is_absolute() else base / path


def scene_paths(config_path, scene_id):
    """Return (radiance, observation, mask, solar spectrum, output GeoJSON) paths."""
    with open(config_path, "rb") as file:
        config = tomllib.load(file)
    base = Path(config_path).parent
    for scene in config["scene"]:
        if scene["id"] == scene_id:
            out = scene.get("clouds", f"../data/labels/{scene_id}_clouds.geojson")
            return (
                resolve(base, scene["radiance"]),
                resolve(base, scene["observation"]),
                resolve(base, scene["mask"]),
                resolve(base, config["solar_spectrum"]),
                resolve(base, out),
            )
    ids = [scene["id"] for scene in config["scene"]]
    sys.exit(f"No scene {scene_id!r} in {config_path}. Scenes: {', '.join(ids)}")


def read_reflectance_bands(radiance_path, observation_path, solar_path, wanted_nm):
    """TOA reflectance of the bands nearest wanted_nm, shape (downtrack, crosstrack, bands)."""
    import netCDF4

    wavelengths, fwhm = emit.read_wavelengths(radiance_path)
    index = [int(np.abs(wavelengths - nm).argmin()) for nm in wanted_nm]
    with netCDF4.Dataset(radiance_path) as dataset:
        variable = dataset.variables["radiance"]
        radiance = np.stack([variable[:, :, i] for i in index], axis=-1).astype(np.float32)
    zenith, distance = emit.read_geometry(observation_path)
    solar_wavelengths, solar_irradiance = load_solar_spectrum(solar_path)
    irradiance = band_solar_irradiance(
        solar_wavelengths, solar_irradiance * W_PER_M2_TO_UW_PER_CM2,
        wavelengths[index], fwhm[index],
    )
    return radiance_to_toa_reflectance(radiance, irradiance, zenith, distance)


def to_display(reflectance, top):
    """Scale reflectance to 0-1 with one common range for all channels.

    A common range keeps the channels' relative brightness, which is what makes
    the phase colours meaningful.
    """
    image = np.clip(np.nan_to_num(reflectance) / top, 0, 1) ** 0.75
    return image


def pixel_to_lonlat(geotransform, cols, rows):
    x0, dx, rx, y0, ry, dy = geotransform
    cols, rows = np.asarray(cols, float), np.asarray(rows, float)
    return x0 + cols * dx + rows * rx, y0 + cols * ry + rows * dy


def lonlat_to_pixel(geotransform, lons, lats):
    x0, dx, rx, y0, ry, dy = geotransform
    inverse = np.linalg.inv(np.array([[dx, rx], [ry, dy]]))
    offsets = np.stack([np.asarray(lons, float) - x0, np.asarray(lats, float) - y0])
    cols, rows = inverse @ offsets
    return cols, rows


def polygons_to_geojson(polygons, geotransform):
    """polygons: list of (label, [(col, row), ...]) in grid pixel coordinates."""
    features = []
    for label, corners in polygons:
        cols, rows = zip(*corners)
        lons, lats = pixel_to_lonlat(geotransform, cols, rows)
        ring = [[float(x), float(y)] for x, y in zip(lons, lats)]
        ring.append(ring[0])
        features.append({
            "type": "Feature",
            "properties": {"label": label},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })
    return {"type": "FeatureCollection", "features": features}


def geojson_to_polygons(collection, geotransform):
    polygons = []
    for feature in collection["features"]:
        geometry = feature["geometry"]
        rings = (
            [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        )
        for rings_of_one in rings:
            lons, lats = zip(*rings_of_one[0][:-1])
            cols, rows = lonlat_to_pixel(geotransform, lons, lats)
            polygons.append((feature["properties"]["label"], list(zip(cols, rows))))
    return polygons


def save(path, polygons, geotransform):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(polygons_to_geojson(polygons, geotransform), indent=1))


def masked_cells(corners, cloud):
    """Number of L2A-cloud cells whose centre lies inside the polygon."""
    from matplotlib.path import Path as MplPath

    cols, rows = np.array(corners).T
    c0, c1 = max(int(cols.min()), 0), min(int(np.ceil(cols.max())) + 1, cloud.shape[1])
    r0, r1 = max(int(rows.min()), 0), min(int(np.ceil(rows.max())) + 1, cloud.shape[0])
    if c0 >= c1 or r0 >= r1:
        return 0
    rr, cc = np.mgrid[r0:r1, c0:c1] + 0.5
    inside = MplPath(corners).contains_points(np.column_stack([cc.ravel(), rr.ravel()]))
    return int(cloud[r0:r1, c0:c1].ravel()[inside].sum())


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene", required=True, help="Scene id from the config")
    parser.add_argument("--out", type=Path, help="GeoJSON to write (default: from the config)")
    args = parser.parse_args()

    radiance_path, observation_path, mask_path, solar_path, out = scene_paths(
        args.config, args.scene
    )
    out = args.out or out
    print(f"Loading {args.scene} ...", flush=True)
    grid = emit.read_grid(radiance_path)
    reflectance = read_reflectance_bands(
        radiance_path, observation_path, solar_path, PHASE_NM + TRUE_COLOUR_NM
    )
    on_grid = grid.to_grid(np.nan_to_num(reflectance).astype(np.float32))
    background = np.where(grid.valid[..., None], 0.0, 0.15)
    views = {
        "cloud phase": to_display(on_grid[..., :3], 0.9) + background,
        "true colour": to_display(on_grid[..., 3:], 0.6) + background,
    }
    cloud, _ = emit.read_cloud_flags(mask_path)
    cloud = grid.to_grid(cloud) & grid.valid
    cirrus = grid.to_grid(emit.read_cirrus_flag(mask_path)) & grid.valid
    flags = {"liquid_cloud": cloud, "ice_cloud": cloud | cirrus}

    polygons = []
    if out.exists():
        polygons = geojson_to_polygons(json.loads(out.read_text()), grid.geotransform)
        print(f"Loaded {len(polygons)} polygons from {out}")

    import matplotlib.pyplot as plt

    my_keys = {"l", "i", "s", "m", "v", "escape", "backspace", "delete"}
    for name in [k for k in plt.rcParams if k.startswith("keymap.")]:
        plt.rcParams[name] = [key for key in plt.rcParams[name] if key not in my_keys]

    height, width = grid.shape
    fig, ax = plt.subplots(figsize=(11, 10))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02)
    state = {"view": "cloud phase", "current": [], "dirty": False, "mask": True}
    image = ax.imshow(views[state["view"]], extent=(0, width, height, 0), interpolation="nearest")
    centres = (np.arange(width) + 0.5, np.arange(height) + 0.5)
    mask_outlines = [
        ax.contour(*centres, cirrus.astype(float), levels=[0.5], colors="magenta", linewidths=0.4),
        ax.contour(*centres, cloud.astype(float), levels=[0.5], colors="cyan", linewidths=0.4),
    ]
    ax.set_xticks([])
    ax.set_yticks([])
    drawn = []
    (current_line,) = ax.plot([], [], "-o", color="yellow", markersize=3, linewidth=1)

    def status():
        counts = {}
        for label, corners in polygons:
            n, cells = counts.get(label, (0, 0))
            counts[label] = (n + 1, cells + masked_cells(corners, flags[label]))
        parts = [
            f"{label.split('_')[0]}: {n} polygon(s), {cells:,} usable pixels"
            for label, (n, cells) in sorted(counts.items())
        ] or ["no polygons yet"]
        unsaved = "  (unsaved)" if state["dirty"] else ""
        ax.set_title(
            f"{args.scene} | {state['view']} view (v) | " + " | ".join(parts) + unsaved
            + "\nclick corners, then l = liquid, i = ice | right click undo | Backspace delete | s save",
            fontsize=9,
        )

    def redraw():
        for artist in drawn:
            artist.remove()
        drawn.clear()
        for label, corners in polygons:
            xs, ys = zip(*(list(corners) + [corners[0]]))
            drawn.extend(ax.plot(xs, ys, color=LABEL_COLOURS[label], linewidth=1.5))
        if state["current"]:
            current_line.set_data(*zip(*state["current"]))
        else:
            current_line.set_data([], [])
        status()
        fig.canvas.draw_idle()

    def do_save():
        save(out, polygons, grid.geotransform)
        state["dirty"] = False
        print(f"Saved {len(polygons)} polygons to {out}", flush=True)
        redraw()

    def on_click(event):
        toolbar = fig.canvas.toolbar
        if event.inaxes is not ax or (toolbar is not None and toolbar.mode):
            return
        if event.button == 1:
            state["current"].append((event.xdata, event.ydata))
        elif event.button == 3 and state["current"]:
            state["current"].pop()
        redraw()

    def on_key(event):
        key = event.key
        if key in ("l", "i"):
            if len(state["current"]) < 3:
                print("A polygon needs at least 3 corners.")
                return
            polygons.append(("liquid_cloud" if key == "l" else "ice_cloud", state["current"]))
            state["current"] = []
            state["dirty"] = True
        elif key == "escape":
            state["current"] = []
        elif key in ("backspace", "delete") and polygons:
            polygons.pop()
            state["dirty"] = True
        elif key == "v":
            state["view"] = "true colour" if state["view"] == "cloud phase" else "cloud phase"
            image.set_data(views[state["view"]])
        elif key == "m":
            state["mask"] = not state["mask"]
            for outline in mask_outlines:
                outline.set_visible(state["mask"])
        elif key == "s":
            do_save()
            return
        redraw()

    def on_close(_):
        if state["dirty"]:
            do_save()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)
    redraw()
    plt.show()


if __name__ == "__main__":
    main()
