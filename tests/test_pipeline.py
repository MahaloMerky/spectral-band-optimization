"""Tests for the EMIT-to-pixel-table pipeline, using small synthetic granules."""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
rasterio = pytest.importorskip("rasterio")

from bandopt.emit import Grid  # noqa: E402
from bandopt.labels import CLASSES  # noqa: E402

SCRIPTS = Path(__file__).parents[1] / "scripts"
WAVELENGTHS = [500.0, 800.0, 1600.0, 2200.0]
CELL = 0.001
X0, Y0 = 10.0, 20.0
RHO = 0.3
OBS_BANDS = [
    "Path length (sensor-to-ground in meters)",
    "To-sensor azimuth (0 to 360 degrees CW from N)",
    "To-sensor zenith (0 to 90 degrees from zenith)",
    "To-sun azimuth (0 to 360 degrees CW from N)",
    "To-sun zenith (0 to 90 degrees from zenith)",
    "Solar phase (degrees between to-sensor and to-sun vectors in principal plane)",
    "Slope (local surface slope as derived from DEM in degrees)",
    "Aspect (local surface aspect 0 to 360 degrees clockwise from N)",
    "Cosine(i) (apparent local illumination factor based on DEM slope and aspect and to sun vector)",
    "UTC Time (decimal hours for mid-line pixels)",
    "Earth-sun distance (AU)",
]
MASK_BANDS = [
    "Cloud flag", "Cirrus flag", "Water flag", "Spacecraft Flag",
    "Dilated Cloud Flag", "AOD550", "H2O (g cm-2)", "Aggregate Flag",
]


def write_product(path, name, data, band_var, band_names, glt=None):
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("downtrack", data.shape[0])
        ds.createDimension("crosstrack", data.shape[1])
        ds.createDimension("bands", data.shape[2])
        ds.createVariable(name, "f4", ("downtrack", "crosstrack", "bands"))[:] = data
        ds.setncattr("geotransform", np.array([X0, CELL, 0.0, Y0, 0.0, -CELL]))
        bands = ds.createGroup("sensor_band_parameters")
        if band_names is not None:
            variable = bands.createVariable(band_var, str, ("bands",))
            variable[:] = np.array(band_names, dtype=object)
        else:
            bands.createVariable("wavelengths", "f4", ("bands",))[:] = WAVELENGTHS
            bands.createVariable("fwhm", "f4", ("bands",))[:] = [8.0] * len(WAVELENGTHS)
        if glt is not None:
            location = ds.createGroup("location")
            location.createDimension("ortho_y", glt[0].shape[0])
            location.createDimension("ortho_x", glt[0].shape[1])
            location.createVariable("glt_x", "i4", ("ortho_y", "ortho_x"))[:] = glt[0]
            location.createVariable("glt_y", "i4", ("ortho_y", "ortho_x"))[:] = glt[1]


@pytest.fixture
def scene_files(tmp_path):
    """A 6 x 5 pixel granule on a 6 x 6 grid whose last column repeats column 4.

    Grid columns 0-1 are tree cover, column 2 is half tree and half built-up,
    columns 3-5 are built-up. Sensor row 0 is cloud: liquid-cloud polygon over
    columns 0-2, ice over 3-5. Sensor pixel (1, 0) has a dilated cloud flag and
    pixel (5, 4) has a fill value.
    """
    rows, cols = 6, 5
    radiance = np.full((rows, cols, 4), RHO * 150.0 / math.pi, dtype=np.float32)
    radiance[5, 4, 2] = -9999.0
    grid_cols = np.minimum(np.arange(6), 4)
    glt_x = np.broadcast_to(grid_cols + 1, (6, 6)).astype(np.int32)
    glt_y = np.broadcast_to(np.arange(6)[:, None] + 1, (6, 6)).astype(np.int32)
    write_product(tmp_path / "rad.nc", "radiance", radiance, None, None, (glt_x, glt_y))

    obs = np.zeros((rows, cols, len(OBS_BANDS)), dtype=np.float32)
    obs[:, :, 4] = 0.0  # Sun overhead
    obs[:, :, 10] = 1.0  # 1 AU
    write_product(tmp_path / "obs.nc", "obs", obs, "observation_bands", OBS_BANDS)

    mask = np.zeros((rows, cols, len(MASK_BANDS)), dtype=np.float32)
    mask[0, :, 0] = 1  # Cloud flag
    mask[1, 0, 4] = 1  # Dilated cloud flag
    write_product(tmp_path / "mask.nc", "mask", mask, "mask_bands", MASK_BANDS)

    # WorldCover at 0.0001 degrees: tree west of 10.0025, built-up east of it.
    res = 0.0001
    west, north = 9.99, 20.01
    width, height = 300, 300
    lon = west + (np.arange(width) + 0.5) * res
    codes = np.where(lon < X0 + 2.5 * CELL, 10, 50).astype(np.uint8)
    codes = np.broadcast_to(codes, (height, width))
    from rasterio.transform import from_origin

    with rasterio.open(
        tmp_path / "wc.tif", "w", driver="GTiff", width=width, height=height, count=1,
        dtype="uint8", crs="EPSG:4326", transform=from_origin(west, north, res, res),
    ) as tif:
        tif.write(codes, 1)

    def box(x_min, x_max):
        y_min, y_max = Y0 - CELL, Y0
        return {"type": "Polygon", "coordinates": [[
            [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]
        ]]}

    clouds = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"label": "liquid_cloud"},
         "geometry": box(X0, X0 + 3 * CELL)},
        {"type": "Feature", "properties": {"label": "ice_cloud"},
         "geometry": box(X0 + 3 * CELL, X0 + 6 * CELL)},
    ]}
    (tmp_path / "clouds.geojson").write_text(json.dumps(clouds))

    solar = np.column_stack([np.arange(300.0, 2600.0), np.full(2300, 1.5)])
    np.savetxt(tmp_path / "solar.csv", solar, delimiter=",")

    (tmp_path / "scenes.toml").write_text("""
solar_spectrum = "solar.csv"
worldcover_tiles = ["wc*.tif"]
[[scene]]
id = "test_scene"
radiance = "rad.nc"
observation = "obs.nc"
mask = "mask.nc"
clouds = "clouds.geojson"
""")
    return tmp_path


def run_build(config, out, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "build_pixel_table.py"),
         "--config", str(config), "--out", str(out), *map(str, args)],
        capture_output=True, text=True,
    )


def test_build_pixel_table_labels_and_reflectance(scene_files):
    out = scene_files / "pixels.npz"
    done = run_build(scene_files / "scenes.toml", out)
    assert done.returncode == 0, done.stderr

    table = np.load(out)
    counts = {name: int((table["labels"] == i).sum()) for i, name in enumerate(CLASSES)}
    # Rows 1-5 x columns 0-1 minus the dilated-cloud pixel; column 2 is mixed.
    assert counts["vegetation"] == 9
    # Rows 1-5 x columns 3-4 minus the fill-value pixel; grid column 5 repeats 4.
    assert counts["urban"] == 9
    assert counts["liquid_cloud"] == 3
    assert counts["ice_cloud"] == 2
    assert counts["bare_soil"] == counts["sand"] == counts["water"] == 0

    np.testing.assert_allclose(table["spectra"], RHO, rtol=1e-4)
    assert table["wavelengths_nm"].tolist() == WAVELENGTHS
    assert set(table["scene_ids"]) == {"test_scene"}
    assert table["class_names"].tolist() == list(CLASSES)
    assert "1 dropped as non-finite" in done.stdout

    summary = out.with_suffix(".counts.csv").read_text().splitlines()
    assert summary[0] == "scene_id," + ",".join(CLASSES)
    assert summary[-1] == "total,9,9,0,0,0,3,2"


def test_scene_override_and_sampling_cap(scene_files):
    config = scene_files / "scenes.toml"
    config.write_text(config.read_text() + '\n[scene.worldcover_classes]\n50 = "ignore"\n')
    out = scene_files / "pixels.npz"
    done = run_build(config, out, "--max-per-class", 4)
    assert done.returncode == 0, done.stderr
    labels = np.load(out)["labels"].tolist()
    assert labels.count(CLASSES.index("urban")) == 0
    assert labels.count(CLASSES.index("vegetation")) == 4
    assert "WARNING: no pixels" in done.stdout


def test_mismatched_granules_are_rejected(scene_files):
    small = np.zeros((3, 5, len(MASK_BANDS)), dtype=np.float32)
    write_product(scene_files / "mask.nc", "mask", small, "mask_bands", MASK_BANDS)
    done = run_build(scene_files / "scenes.toml", scene_files / "pixels.npz")
    assert done.returncode != 0
    assert "same granule" in done.stderr


def test_conflicting_labels_for_one_sensor_pixel_are_dropped():
    sys.path.insert(0, str(SCRIPTS))
    from build_pixel_table import sensor_pixels

    # Three grid cells: two map to sensor pixel (0, 0), one to (0, 1).
    grid = Grid(
        glt_rows=np.array([[0, 0, 0]]), glt_cols=np.array([[0, 0, 1]]),
        geotransform=(0, 1, 0, 0, 0, -1),
    )
    rows, cols, labels, _ = sensor_pixels(grid, np.array([[0, 1, 2]]))
    assert list(zip(rows, cols, labels)) == [(0, 1, 2)]

    # The same label twice keeps the pixel once.
    rows, cols, labels, _ = sensor_pixels(grid, np.array([[1, 1, 2]]))
    assert list(zip(rows, cols, labels)) == [(0, 0, 1), (0, 1, 2)]


def test_ice_polygons_accept_cirrus_flag_but_liquid_does_not():
    from bandopt.labels import UNLABELLED, class_index, label_grid

    shape = (1, 4)
    grid = Grid(
        glt_rows=np.zeros(shape, dtype=np.int64),
        glt_cols=np.arange(4)[None, :],
        geotransform=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
    )
    ice, liquid = class_index("ice_cloud"), class_index("liquid_cloud")
    polygons = np.array([[ice, ice, liquid, liquid]])
    cloud = np.array([[True, False, True, False]])
    cirrus = np.array([[False, True, False, True]])
    any_flag = cloud | cirrus

    labels = label_grid(grid, {}, cloud, any_flag, polygons, cirrus=cirrus)
    # Ice: cloud-flagged and cirrus-flagged pixels both count.
    # Liquid: only the cloud-flagged pixel counts.
    assert labels.tolist() == [[ice, ice, liquid, UNLABELLED]]

    # Without the cirrus flag the old rule applies: cloud flag only.
    labels = label_grid(grid, {}, cloud, any_flag, polygons)
    assert labels.tolist() == [[ice, UNLABELLED, liquid, UNLABELLED]]
