"""Simulate multispectral bands from hyperspectral spectra."""

import torch

# Strong water-vapour absorption, removed from the main analysis.
WATER_VAPOUR_WINDOWS_NM = ((1350.0, 1420.0), (1800.0, 1950.0))
# Cirrus sensitivity run: keeps 1.35-1.42 um, where ice cloud can stay bright while
# the surface is mostly hidden. Almost no signal reaches the sensor at 1.80-1.95 um.
CIRRUS_RUN_WINDOWS_NM = ((1800.0, 1950.0),)

FWHM_PER_SIGMA = 2.355

# Landsat 8/9 OLI band edges in nm (USGS). Bands are approximated as Gaussians with
# the band's centre and a FWHM equal to its edge-to-edge width; the published
# relative spectral responses can replace this approximation later.
OLI_BAND_EDGES_NM = {
    "B1 coastal": (433.0, 453.0),
    "B2 blue": (450.0, 515.0),
    "B3 green": (525.0, 600.0),
    "B4 red": (630.0, 680.0),
    "B5 NIR": (845.0, 885.0),
    "B6 SWIR1": (1560.0, 1660.0),
    "B7 SWIR2": (2100.0, 2300.0),
    "B9 cirrus": (1360.0, 1390.0),
}
OLI_REFLECTIVE = tuple(name for name in OLI_BAND_EDGES_NM if name != "B9 cirrus")


def band_weights(wavelengths_nm, edges_nm):
    """Normalised Gaussian weights of bands on a wavelength grid.

    Args:
        wavelengths_nm: Grid with shape (wavelengths,), for example the candidate
            bands of a statistics file.
        edges_nm: Sequence of (low, high) band edges.

    Returns:
        Weights with shape (bands, wavelengths) whose rows sum to 1, so that band
        values are weights @ spectrum and band statistics are W mu and W Sigma W^T.
    """
    wavelengths_nm = torch.as_tensor(wavelengths_nm, dtype=torch.float64)
    edges = torch.as_tensor(edges_nm, dtype=torch.float64)
    centres = edges.mean(dim=1)
    sigmas = (edges[:, 1] - edges[:, 0]) / FWHM_PER_SIGMA
    responses = gaussian_response(wavelengths_nm, centres, sigmas)
    return responses / responses.sum(dim=1, keepdim=True)


def candidate_band_mask(wavelengths_nm, windows=WATER_VAPOUR_WINDOWS_NM):
    """Return a boolean mask that is True for bands outside the excluded windows."""
    wavelengths_nm = torch.as_tensor(wavelengths_nm)
    mask = torch.ones_like(wavelengths_nm, dtype=torch.bool)
    for low, high in windows:
        mask &= ~((wavelengths_nm >= low) & (wavelengths_nm <= high))
    return mask


def gaussian_response(wavelengths_nm, centres_nm, sigmas_nm):
    """Evaluate Gaussian responses (Eq. 4). Returns shape (bands, wavelengths)."""
    difference = wavelengths_nm[None, :] - centres_nm[:, None]
    return torch.exp(-(difference**2) / (2 * sigmas_nm[:, None] ** 2))


def simulate_bands(reflectance, responses):
    """Apply response functions to spectra (Eq. 1).

    The EMIT spacing is treated as uniform, so the delta-lambda terms cancel.

    Args:
        reflectance: Spectra with shape (..., wavelengths).
        responses: Response functions with shape (bands, wavelengths).

    Returns:
        Band reflectance with shape (..., bands).
    """
    return reflectance @ responses.T / responses.sum(dim=1)
