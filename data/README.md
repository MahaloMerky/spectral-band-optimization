# Data

Nothing in this folder is committed except this file. Stage large files on cluster
scratch storage and symlink them here.

| Dataset | Source | Local path |
| --- | --- | --- |
| EMIT L2A surface reflectance and masks (`EMITL2ARFL`) | [NASA LP DAAC via Earthdata](https://lpdaac.usgs.gov/products/emitl2arflv001/) | `data/raw/emit/` |
| ESA WorldCover 10 m 2021 v200 | [Zenodo](https://doi.org/10.5281/zenodo.7254221) | `data/raw/worldcover/` |
| OLI and OLI-2 relative spectral responses | USGS | `data/raw/srf/` |
| Landsat Next band edges | [USGS Fact Sheet 2024-3005](https://pubs.usgs.gov/publication/fs20243005) | `data/raw/srf/` |

EMIT downloads need a free [Earthdata account](https://urs.earthdata.nasa.gov/users/new).
Keep the login in `~/.netrc`, never in this repository.

Processed outputs go in `data/processed/`. Class statistics are stored as `.npz` files
with `mu`, `sigma`, and `wavelengths_nm`, using only the candidate bands left after the
water-vapour windows are removed.
