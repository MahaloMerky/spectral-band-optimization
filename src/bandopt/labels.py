"""Class labels on an EMIT GLT grid from WorldCover and hand-drawn cloud polygons."""

import json

import numpy as np

CLASSES = (
    "vegetation",
    "urban",
    "bare_soil",
    "sand",
    "water",
    "liquid_cloud",
    "ice_cloud",
)
SURFACE_CLASSES = CLASSES[:5]
CLOUD_CLASSES = CLASSES[5:]
UNLABELLED = -1

# ESA WorldCover 2021 v200 codes. Shrubland (20) and cropland (40) are left out
# by default: they mix vegetation and soil, and cropland is often bare at the
# acquisition date. WorldCover has one bare class (60), so each scene's config
# decides whether it is bare soil or sand.
DEFAULT_WORLDCOVER_CLASSES = {
    10: "vegetation",  # Tree cover
    30: "vegetation",  # Grassland
    50: "urban",  # Built-up
    60: "bare_soil",  # Bare / sparse vegetation
    80: "water",  # Permanent water bodies
}
WORLDCOVER_NODATA = 0


def class_index(name):
    if name not in CLASSES:
        raise ValueError(f"Unknown class {name!r}. Classes: {list(CLASSES)}")
    return CLASSES.index(name)


def worldcover_fractions(tile_paths, grid, mapping):
    """Return the fraction of each surface class inside every grid cell.

    Each class is reprojected as a 0/1 layer with area averaging, so a value is
    the share of WorldCover 10 m pixels in that ~60 m cell belonging to the class.

    Args:
        tile_paths: WorldCover GeoTIFF tiles that together cover the scene.
        grid: The scene's emit.Grid.
        mapping: Dict from WorldCover code to class name; unmapped codes count
            towards no class, so they lower the purity of a cell.

    Returns:
        Dict from class name to a float array (height, width). Cells outside the
        tiles are NaN.
    """
    import rasterio
    from rasterio.merge import merge
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject

    west, south, east, north = grid.bounds()
    margin = 0.01
    datasets = [rasterio.open(path) for path in tile_paths]
    try:
        overlapping = [
            ds for ds in datasets
            if ds.bounds.left < east + margin and ds.bounds.right > west - margin
            and ds.bounds.bottom < north + margin and ds.bounds.top > south - margin
        ]
        if not overlapping:
            raise ValueError(f"No WorldCover tile overlaps the scene bounds {grid.bounds()}")
        mosaic, transform = merge(
            overlapping,
            bounds=(west - margin, south - margin, east + margin, north + margin),
            nodata=WORLDCOVER_NODATA,
        )
        crs = overlapping[0].crs
    finally:
        for dataset in datasets:
            dataset.close()
    codes = mosaic[0]

    destination_transform = Affine.from_gdal(*grid.geotransform)
    fractions = {}
    for name in sorted(set(mapping.values())):
        member_codes = [code for code, target in mapping.items() if target == name]
        layer = np.isin(codes, member_codes).astype(np.float32)
        # Cells the mosaic does not cover stay NaN rather than counting as 0.
        layer[codes == WORLDCOVER_NODATA] = np.nan
        out = np.full(grid.shape, np.nan, dtype=np.float32)
        reproject(
            source=layer,
            destination=out,
            src_transform=transform,
            src_crs=crs,
            src_nodata=np.nan,
            dst_transform=destination_transform,
            dst_crs="EPSG:4326",
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
        fractions[name] = out
    return fractions


def read_cloud_polygons(path, grid):
    """Rasterise hand-drawn cloud polygons onto the grid.

    The file is GeoJSON in longitude/latitude (EPSG:4326, the GeoJSON default),
    with a "label" property of "liquid_cloud" or "ice_cloud" on every feature.

    Returns:
        Int array (height, width) of class indices, UNLABELLED outside polygons
        and where polygons of different phases overlap.
    """
    from rasterio.features import rasterize
    from rasterio.transform import Affine

    with open(path) as file:
        features = json.load(file)["features"]
    transform = Affine.from_gdal(*grid.geotransform)
    labels = np.full(grid.shape, UNLABELLED, dtype=np.int64)
    covered = np.zeros(grid.shape, dtype=np.int64)
    unknown = {(f.get("properties") or {}).get("label") for f in features} - set(CLOUD_CLASSES)
    if unknown:
        raise ValueError(
            f"{path}: every feature needs a label of {list(CLOUD_CLASSES)}, "
            f"found {sorted(map(str, unknown))}"
        )
    for name in CLOUD_CLASSES:
        shapes = [f["geometry"] for f in features if f["properties"]["label"] == name]
        if not shapes:
            continue
        inside = rasterize(shapes, out_shape=grid.shape, transform=transform, fill=0,
                           default_value=1, dtype="uint8").astype(bool)
        labels[inside] = class_index(name)
        covered += inside
    labels[covered > 1] = UNLABELLED
    return labels


def label_grid(grid, fractions, cloud, any_flag, cloud_polygons=None, purity=0.9, cirrus=None):
    """Combine WorldCover purity, the L2A mask, and cloud polygons into labels.

    Surface classes need a class fraction of at least `purity`, no cloud,
    dilated cloud, cirrus or spacecraft flag, and no cloud polygon. Liquid
    cloud needs a polygon and the L2A cloud flag. Ice cloud needs a polygon and
    the cloud flag or, when `cirrus` is given, the L2A cirrus flag: the cloud
    flag misses most ice anvils that are not very bright.

    Args:
        grid: The scene's emit.Grid.
        fractions: Output of worldcover_fractions.
        cloud: L2A cloud flag on the grid (height, width).
        any_flag: Any L2A cloud-type flag on the grid (height, width).
        cloud_polygons: Output of read_cloud_polygons, or None.
        purity: Minimum class fraction for surface classes.
        cirrus: L2A cirrus flag on the grid (height, width), or None.

    Returns:
        Int array (height, width) of class indices, or UNLABELLED.
    """
    labels = np.full(grid.shape, UNLABELLED, dtype=np.int64)
    in_polygon = np.zeros(grid.shape, dtype=bool)
    if cloud_polygons is not None:
        in_polygon = cloud_polygons != UNLABELLED

    clear = grid.valid & ~any_flag & ~in_polygon
    for name, fraction in fractions.items():
        pure = np.nan_to_num(fraction, nan=0.0) >= purity
        labels[clear & pure] = class_index(name)

    if cloud_polygons is not None:
        flagged = cloud.copy()
        if cirrus is not None:
            flagged |= cirrus & (cloud_polygons == class_index("ice_cloud"))
        is_cloud = grid.valid & flagged & in_polygon
        labels[is_cloud] = cloud_polygons[is_cloud]
    return labels
