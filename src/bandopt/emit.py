"""Read EMIT L1B radiance, L1B observation geometry, and L2A mask files.

EMIT products are stored in sensor geometry (downtrack x crosstrack). Each file
also holds a geometry lookup table (GLT): a regular latitude/longitude grid whose
cells give the 1-based crosstrack (glt_x) and downtrack (glt_y) index of the
sensor pixel that covers them, or 0 where there is no data. Labels are built on
that grid, then mapped back to sensor pixels, so spectra are never resampled.
"""

from dataclasses import dataclass

import numpy as np

GLT_NODATA = 0
SUN_ZENITH_BAND = "to-sun zenith"
EARTH_SUN_DISTANCE_BAND = "earth-sun distance"
CLOUD_FLAGS = ("cloud flag", "dilated cloud flag", "cirrus flag", "spacecraft flag")


def _open(path):
    import netCDF4

    dataset = netCDF4.Dataset(path)
    dataset.set_auto_mask(False)
    return dataset


def _variable(group, name, path):
    if name not in group.variables:
        raise KeyError(
            f"{path}: no variable {name!r} in group {group.path!r}. "
            f"Found: {sorted(group.variables)}"
        )
    return group.variables[name]


def _band_names(dataset, variable_name, path):
    """Return lower-case band names stored next to a product's main variable."""
    group = dataset.groups.get("sensor_band_parameters")
    if group is None:
        raise KeyError(f"{path}: no 'sensor_band_parameters' group")
    names = _variable(group, variable_name, path)[:]
    return [str(name).strip().lower() for name in names]


def _find_band(names, wanted, path, exact=False):
    for index, name in enumerate(names):
        if name == wanted or (not exact and name.startswith(wanted)):
            return index
    raise KeyError(f"{path}: no band named {wanted!r}. Found: {names}")


@dataclass
class Grid:
    """The GLT grid of one scene."""

    glt_rows: np.ndarray  # (height, width) 0-based downtrack index, -1 for no data
    glt_cols: np.ndarray  # (height, width) 0-based crosstrack index, -1 for no data
    geotransform: tuple  # GDAL order: (x0, dx, rx, y0, ry, dy), EPSG:4326

    @property
    def shape(self):
        return self.glt_rows.shape

    @property
    def valid(self):
        return self.glt_rows >= 0

    def to_grid(self, raw):
        """Sample a sensor-geometry array (downtrack, crosstrack, ...) onto the grid.

        Cells without data are filled with zeros; use `valid` to mask them.
        """
        out = np.zeros(self.shape + raw.shape[2:], dtype=raw.dtype)
        valid = self.valid
        out[valid] = raw[self.glt_rows[valid], self.glt_cols[valid]]
        return out

    def cell_centres(self):
        """Return longitude and latitude of every cell centre, shape (height, width)."""
        x0, dx, rx, y0, ry, dy = self.geotransform
        rows, cols = np.indices(self.shape) + 0.5
        return x0 + cols * dx + rows * rx, y0 + cols * ry + rows * dy

    def bounds(self):
        """Return (west, south, east, north) of the grid."""
        x0, dx, rx, y0, ry, dy = self.geotransform
        height, width = self.shape
        corners_x = [x0, x0 + width * dx, x0 + height * rx, x0 + width * dx + height * rx]
        corners_y = [y0, y0 + width * ry, y0 + height * dy, y0 + width * ry + height * dy]
        return min(corners_x), min(corners_y), max(corners_x), max(corners_y)


def read_grid(radiance_path):
    """Read the GLT grid from an L1B radiance file."""
    with _open(radiance_path) as dataset:
        location = dataset.groups.get("location")
        if location is None:
            raise KeyError(f"{radiance_path}: no 'location' group")
        glt_x = np.abs(_variable(location, "glt_x", radiance_path)[:].astype(np.int64))
        glt_y = np.abs(_variable(location, "glt_y", radiance_path)[:].astype(np.int64))
        geotransform = tuple(float(v) for v in dataset.getncattr("geotransform"))
    valid = (glt_x != GLT_NODATA) & (glt_y != GLT_NODATA)
    return Grid(
        glt_rows=np.where(valid, glt_y - 1, -1),
        glt_cols=np.where(valid, glt_x - 1, -1),
        geotransform=geotransform,
    )


def read_wavelengths(radiance_path):
    """Return band centres and FWHM in nm from an L1B radiance file."""
    with _open(radiance_path) as dataset:
        group = dataset.groups["sensor_band_parameters"]
        wavelengths = _variable(group, "wavelengths", radiance_path)[:].astype(float)
        fwhm = _variable(group, "fwhm", radiance_path)[:].astype(float)
    return wavelengths, fwhm


def read_radiance(radiance_path):
    """Return radiance (downtrack, crosstrack, bands) in uW/cm^2/sr/nm, float32."""
    with _open(radiance_path) as dataset:
        return _variable(dataset, "radiance", radiance_path)[:].astype(np.float32)


def read_geometry(observation_path):
    """Return solar zenith (deg) and Earth-Sun distance (AU), each (downtrack, crosstrack)."""
    with _open(observation_path) as dataset:
        names = _band_names(dataset, "observation_bands", observation_path)
        obs = _variable(dataset, "obs", observation_path)
        zenith = obs[:, :, _find_band(names, SUN_ZENITH_BAND, observation_path)]
        distance = obs[:, :, _find_band(names, EARTH_SUN_DISTANCE_BAND, observation_path)]
    return zenith.astype(float), distance.astype(float)


def read_cloud_flags(mask_path):
    """Return (cloud, any_flag) boolean arrays (downtrack, crosstrack) from an L2A mask.

    `cloud` is the L2A cloud flag. `any_flag` is True where any of the cloud,
    dilated cloud, cirrus, or spacecraft flags is set, which excludes a pixel
    from the surface classes.
    """
    with _open(mask_path) as dataset:
        names = _band_names(dataset, "mask_bands", mask_path)
        mask = _variable(dataset, "mask", mask_path)
        flags = {
            name: mask[:, :, _find_band(names, name, mask_path, exact=True)] > 0
            for name in CLOUD_FLAGS
        }
    any_flag = np.logical_or.reduce(list(flags.values()))
    return flags["cloud flag"], any_flag


def read_cirrus_flag(mask_path):
    """Return the L2A cirrus flag (downtrack, crosstrack) as a boolean array.

    The cirrus flag comes from the 1.38 um water-vapour band, which the main
    band search excludes, so using it to admit ice-cloud pixels does not tie
    the labels to a candidate band.
    """
    with _open(mask_path) as dataset:
        names = _band_names(dataset, "mask_bands", mask_path)
        mask = _variable(dataset, "mask", mask_path)
        return mask[:, :, _find_band(names, "cirrus flag", mask_path, exact=True)] > 0
