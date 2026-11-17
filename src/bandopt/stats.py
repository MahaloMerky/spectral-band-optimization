"""Per-class spectral statistics and scene-based data splits."""

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

    Scenes are assigned deterministically, rarest classes first: for each class,
    in order of how few scenes contain it, its not-yet-assigned scenes go to the
    folds holding the fewest scenes of that class (ties: fewest pixels). Scenes of
    a class found in only a few scenes are therefore spread over different folds,
    so that no fold holds out every scene of a class while one remains possible.
    """
    labels = np.asarray(labels)
    scene_ids = np.asarray(scene_ids)
    all_classes = np.unique(labels)
    scenes = np.unique(scene_ids)
    if not 2 <= n_folds <= scenes.size:
        raise ValueError(f"n_folds must be between 2 and the number of scenes ({scenes.size})")
    has_class = {s: set(np.unique(labels[scene_ids == s]).tolist()) for s in scenes}
    size = {s: int((scene_ids == s).sum()) for s in scenes}

    fold_of = {}
    class_count = np.zeros((n_folds, all_classes.size), dtype=int)
    pixels = np.zeros(n_folds, dtype=int)
    order = sorted(range(all_classes.size), key=lambda c: sum(all_classes[c] in has_class[s] for s in scenes))
    for c in order:
        members = sorted((s for s in scenes if all_classes[c] in has_class[s] and s not in fold_of),
                         key=lambda s: -size[s])
        for s in members:
            fold = min(range(n_folds), key=lambda f: (class_count[f, c], pixels[f], f))
            fold_of[s] = fold
            pixels[fold] += size[s]
            for d, cls in enumerate(all_classes):
                class_count[fold, d] += cls in has_class[s]

    assignment = np.array([fold_of[s] for s in scene_ids])
    folds = []
    for fold in range(n_folds):
        test = assignment == fold
        missing = np.setdiff1d(all_classes, labels[~test])
        if missing.size:
            raise ValueError(
                f"Fold {fold} has no training pixels for classes {missing.tolist()}"
            )
        folds.append((~test, test))
    return folds
