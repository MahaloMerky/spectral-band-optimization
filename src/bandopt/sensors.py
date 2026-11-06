"""Simulate multispectral bands from hyperspectral spectra."""

import torch

# Strong water-vapour absorption, where EMIT surface reflectance is unreliable.
WATER_VAPOUR_WINDOWS_NM = ((1350.0, 1420.0), (1800.0, 1950.0))

FWHM_PER_SIGMA = 2.355


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
