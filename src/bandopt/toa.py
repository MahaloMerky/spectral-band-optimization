"""Convert at-sensor radiance to top-of-atmosphere (TOA) reflectance."""

import numpy as np

from bandopt.sensors import FWHM_PER_SIGMA

W_PER_M2_TO_UW_PER_CM2 = 100.0
EMIT_FILL_VALUE = -9999.0


def load_solar_spectrum(path):
    """Read a solar spectrum as (wavelengths in nm, irradiance in W/m^2/nm).

    Accepts the TSIS-1 HSRS NetCDF file (variables whose names contain
    "wavelength" and "SSI"), or a text/CSV file of two columns in those units.
    Lines starting with # are skipped.
    """
    path = str(path)
    if path.endswith(".nc"):
        import netCDF4

        with netCDF4.Dataset(path) as dataset:
            names = list(dataset.variables)
            wavelength = [n for n in names if "wavelength" in n.lower()]
            irradiance = [n for n in names if n.lower() == "ssi"]
            if len(wavelength) != 1 or len(irradiance) != 1:
                raise KeyError(
                    f"{path}: expected one wavelength variable and one 'SSI' variable, "
                    f"found {names}"
                )
            wavelengths = np.asarray(dataset[wavelength[0]][:], dtype=float)
            values = np.asarray(dataset[irradiance[0]][:], dtype=float)
    else:
        table = np.loadtxt(path, delimiter="," if path.endswith(".csv") else None, comments="#")
        wavelengths, values = table[:, 0], table[:, 1]
    order = np.argsort(wavelengths)
    return wavelengths[order], values[order]


def band_solar_irradiance(solar_wavelengths_nm, solar_irradiance, centres_nm, fwhm_nm):
    """Convolve a solar spectrum with Gaussian band responses.

    Args:
        solar_wavelengths_nm: Ascending solar wavelengths with shape (samples,).
        solar_irradiance: Solar irradiance at 1 AU with shape (samples,).
        centres_nm: Band centres with shape (bands,).
        fwhm_nm: Band full widths at half maximum with shape (bands,).

    Returns:
        Band-averaged irradiance with shape (bands,), in the input units.
    """
    solar_wavelengths_nm = np.asarray(solar_wavelengths_nm, dtype=float)
    solar_irradiance = np.asarray(solar_irradiance, dtype=float)
    centres_nm = np.asarray(centres_nm, dtype=float)
    sigmas = np.asarray(fwhm_nm, dtype=float) / FWHM_PER_SIGMA

    difference = solar_wavelengths_nm[None, :] - centres_nm[:, None]
    weights = np.exp(-(difference**2) / (2 * sigmas[:, None] ** 2))
    weights = weights * np.gradient(solar_wavelengths_nm)
    return (weights * solar_irradiance).sum(axis=1) / weights.sum(axis=1)


def radiance_to_toa_reflectance(
    radiance, solar_irradiance, sun_zenith_deg, earth_sun_au, fill_value=EMIT_FILL_VALUE
):
    """Compute rho = pi * L * d^2 / (E0 * cos(theta)).

    Args:
        radiance: Radiance with shape (..., bands).
        solar_irradiance: Band irradiance at 1 AU with shape (bands,), in the same
            flux units as the radiance. EMIT radiance is in uW/cm^2/sr/nm, so use
            W_PER_M2_TO_UW_PER_CM2 to convert a spectrum given in W/m^2/nm.
        sun_zenith_deg: Solar zenith angle, a scalar or shape (...).
        earth_sun_au: Earth-Sun distance in AU, a scalar or shape (...).
        fill_value: Radiance marking missing data.

    Returns:
        Reflectance with shape (..., bands). Fill values and pixels with the Sun at
        or below the horizon are NaN.
    """
    radiance = np.asarray(radiance, dtype=float)
    radiance = np.where(radiance == fill_value, np.nan, radiance)
    cos_zenith = np.cos(np.radians(np.asarray(sun_zenith_deg, dtype=float)))
    cos_zenith = np.where(cos_zenith > 0, cos_zenith, np.nan)
    distance = np.asarray(earth_sun_au, dtype=float)
    return (
        np.pi * radiance * distance[..., None] ** 2
        / (np.asarray(solar_irradiance, dtype=float) * cos_zenith[..., None])
    )
