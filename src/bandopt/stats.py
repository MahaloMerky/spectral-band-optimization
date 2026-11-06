"""Per-class spectral statistics and scene-based data splits."""

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import GroupKFold


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


def scene_folds(labels, scene_ids, n_folds):
    """Split pixels into folds that keep each scene whole.

    Band selection must use only a fold's training mask, or held-out scores will
    be inflated.

    Args:
        labels: Integer class labels with shape (pixels,).
        scene_ids: Scene identifiers with shape (pixels,).
        n_folds: Number of folds, at most the number of scenes.

    Returns:
        A list of (train_mask, test_mask) boolean arrays.

    Raises:
        ValueError: If a fold's training data would be missing a class. Every class
            must have pixels in enough scenes for the chosen number of folds.
    """
    labels = np.asarray(labels)
    scene_ids = np.asarray(scene_ids)
    all_classes = np.unique(labels)
    folds = []
    for fold, (train, test) in enumerate(
        GroupKFold(n_splits=n_folds).split(labels, groups=scene_ids)
    ):
        missing = np.setdiff1d(all_classes, labels[train])
        if missing.size:
            raise ValueError(
                f"Fold {fold} has no training pixels for classes {missing.tolist()}"
            )
        train_mask = np.zeros(labels.size, dtype=bool)
        train_mask[train] = True
        folds.append((train_mask, ~train_mask))
    return folds
