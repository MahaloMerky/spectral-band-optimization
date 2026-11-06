"""Per-class spectral statistics used by the separability score."""

import numpy as np
from sklearn.covariance import LedoitWolf


def class_statistics(spectra, labels, ridge=1e-6):
    """Compute class means and Ledoit-Wolf covariances with a small ridge.

    Adjacent EMIT bands are highly correlated, so raw covariances can be
    ill-conditioned. The ridge is scaled by each covariance's mean diagonal.

    Args:
        spectra: Array with shape (pixels, bands).
        labels: Integer class labels with shape (pixels,).
        ridge: Fraction of the mean variance added to the diagonal.

    Returns:
        A tuple of means (classes, bands), covariances (classes, bands, bands),
        and the sorted class labels (classes,).
    """
    classes = np.unique(labels)
    n_bands = spectra.shape[1]
    means = np.empty((classes.size, n_bands))
    covariances = np.empty((classes.size, n_bands, n_bands))
    for i, label in enumerate(classes):
        pixels = spectra[labels == label]
        estimator = LedoitWolf().fit(pixels)
        covariance = estimator.covariance_
        loading = ridge * np.trace(covariance) / n_bands
        means[i] = estimator.location_
        covariances[i] = covariance + loading * np.eye(n_bands)
    return means, covariances, classes
