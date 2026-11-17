"""Band selection by group-lasso multinomial logistic regression.

Fits softmax regression on standardised bands with the penalty

    lambda * sum_j || W_j ||_2,

where W_j is band j's row of coefficients across all classes, so a band is
either used by every class or dropped entirely. Solved by FISTA (accelerated
proximal gradient) with class-weighted cross-entropy, so rare classes count.

Unlike the other heuristics this optimises classification loss, not the
worst-case separability score; select_bands reports the bands, and the caller
scores them with the same JM objective as everything else.
"""

import numpy as np
import torch


def group_soft_threshold(weights, threshold):
    """Proximal operator of threshold * sum_j ||W_j||: shrink each row towards 0."""
    norms = torch.linalg.vector_norm(weights, dim=1, keepdim=True)
    scale = torch.clamp(1 - threshold / norms.clamp(min=1e-12), min=0)
    return weights * scale


class GroupLassoPath:
    """Fit a decreasing sequence of penalties with warm starts."""

    def __init__(self, features, labels, n_classes, dtype=torch.float64):
        x = torch.as_tensor(features, dtype=dtype)
        self.mean, self.sd = x.mean(dim=0), x.std(dim=0).clamp(min=1e-12)
        self.x = (x - self.mean) / self.sd
        self.y = torch.as_tensor(labels, dtype=torch.int64)
        self.n, self.p = self.x.shape
        self.n_classes = n_classes
        counts = torch.bincount(self.y, minlength=n_classes).to(dtype)
        class_weight = torch.where(counts > 0, counts.sum() / (n_classes * counts.clamp(min=1)),
                                   torch.zeros_like(counts))
        self.sample_weight = class_weight[self.y] / class_weight[self.y].sum()
        self.onehot = torch.nn.functional.one_hot(self.y, n_classes).to(dtype)
        # Lipschitz constant of the weighted softmax loss gradient (Hessian <= 0.5 X^T D X).
        weighted = self.x * self.sample_weight.sqrt()[:, None]
        self.lipschitz = 0.5 * float(torch.linalg.matrix_norm(weighted, ord=2)) ** 2
        self.weights = torch.zeros(self.p, n_classes, dtype=dtype)
        self.bias = torch.zeros(n_classes, dtype=dtype)

    def gradient(self, weights, bias):
        probabilities = torch.softmax(self.x @ weights + bias, dim=1)
        residual = (probabilities - self.onehot) * self.sample_weight[:, None]
        return self.x.T @ residual, residual.sum(dim=0)

    def lambda_max(self):
        """Smallest penalty at which every band is dropped (with the bias fitted)."""
        prior = (self.onehot * self.sample_weight[:, None]).sum(dim=0)
        bias = torch.log(prior.clamp(min=1e-12))
        grad, _ = self.gradient(torch.zeros_like(self.weights), bias)
        return float(torch.linalg.vector_norm(grad, dim=1).max())

    def fit(self, penalty, iterations=300, tolerance=1e-7):
        """FISTA from the current (warm-start) solution; returns the active bands."""
        step = 1.0 / self.lipschitz
        w, b = self.weights.clone(), self.bias.clone()
        w_previous, b_previous, t = w.clone(), b.clone(), 1.0
        for _ in range(iterations):
            t_next = (1 + (1 + 4 * t * t) ** 0.5) / 2
            momentum = (t - 1) / t_next
            vw, vb = w + momentum * (w - w_previous), b + momentum * (b - b_previous)
            grad_w, grad_b = self.gradient(vw, vb)
            w_previous, b_previous = w, b
            w = group_soft_threshold(vw - step * grad_w, step * penalty)
            b = vb - step * grad_b
            t = t_next
            if float(torch.linalg.vector_norm(w - w_previous)) < tolerance * (1 + float(torch.linalg.vector_norm(w))):
                break
        self.weights, self.bias = w, b
        return self.active()

    def active(self):
        return torch.nonzero(torch.linalg.vector_norm(self.weights, dim=1) > 0).flatten().tolist()


def select_bands(features, labels, n_classes, k, n_lambdas=60, ratio=1e-3, iterations=300):
    """Return k band indices from the group-lasso path, and the penalty used.

    Walks the penalty down geometrically from lambda_max until at least k bands
    are active; if more than k became active at that step, keeps the k with the
    largest coefficient norms.
    """
    path = GroupLassoPath(features, labels, n_classes)
    top = path.lambda_max()
    for penalty in top * np.logspace(0, np.log10(ratio), n_lambdas)[1:]:
        active = path.fit(penalty, iterations=iterations)
        if len(active) >= k:
            norms = torch.linalg.vector_norm(path.weights, dim=1)
            chosen = torch.argsort(norms, descending=True)[:k]
            return sorted(chosen.tolist()), float(penalty)
    raise RuntimeError(f"only {len(path.active())} bands became active; lower ratio")
