"""Differentiable sensor design: Gaussian bands with learnable centres and widths.

Each band j has a centre c_j and a FWHM w_j; its response on the EMIT grid is a
Gaussian (Eq. 4) and its value is the response-weighted mean reflectance
(Eq. 1). The band values feed a small classifier, and everything is trained
end to end with

    loss = cross-entropy
           + alpha * sum_j (mean absorption under band j)
           + beta  * sum_j max(0, w_min - w_j)^2

(Eq. 5). During training each band gets Gaussian noise with standard deviation
noise * sqrt(reference_fwhm / w_j), so narrow bands pay in signal-to-noise.

The absorption A(lambda) used here is a proxy: 1 inside the strong water-vapour
windows and 0 elsewhere, with 10 nm cosine edges. A standard-atmosphere
transmittance can replace it through `absorption`.
"""

import math

import numpy as np
import torch
from torch import nn

from bandopt.sensors import FWHM_PER_SIGMA, WATER_VAPOUR_WINDOWS_NM, gaussian_response


def window_absorption(wavelengths_nm, windows=WATER_VAPOUR_WINDOWS_NM, edge_nm=10.0):
    """Absorption proxy: 1 inside the windows, 0 outside, with smooth edges."""
    wavelengths_nm = torch.as_tensor(wavelengths_nm, dtype=torch.float32)
    absorption = torch.zeros_like(wavelengths_nm)
    for low, high in windows:
        distance = torch.maximum(low - wavelengths_nm, wavelengths_nm - high)  # <0 inside
        ramp = 0.5 * (1 - torch.cos(math.pi * torch.clamp(1 - distance / edge_nm, 0, 1)))
        absorption = torch.maximum(absorption, torch.where(distance <= 0, torch.ones_like(ramp), ramp))
    return absorption


class LearnableSensor(nn.Module):
    """k Gaussian bands on a fixed wavelength grid, with learnable centres and FWHMs."""

    def __init__(self, wavelengths_nm, k, init_centres_nm=None, init_fwhm_nm=40.0,
                 noise=0.005, reference_fwhm_nm=30.0):
        super().__init__()
        wavelengths = torch.as_tensor(wavelengths_nm, dtype=torch.float32)
        self.register_buffer("wavelengths", wavelengths)
        self.low, self.high = float(wavelengths.min()), float(wavelengths.max())
        if init_centres_nm is None:
            init_centres_nm = np.linspace(self.low + 50, self.high - 50, k)
        fraction = (torch.as_tensor(init_centres_nm, dtype=torch.float32) - self.low) / (self.high - self.low)
        self.centre_logit = nn.Parameter(torch.logit(fraction.clamp(1e-3, 1 - 1e-3)))
        # FWHM = 1 nm + exp(raw): positive, and a step changes it by a fraction of
        # its size, so wide and narrow bands adapt at the same relative rate.
        self.fwhm_raw = nn.Parameter(torch.full((k,), math.log(init_fwhm_nm - 1.0)))
        self.noise = noise
        self.reference_fwhm = reference_fwhm_nm

    @property
    def centres(self):
        return self.low + (self.high - self.low) * torch.sigmoid(self.centre_logit)

    @property
    def fwhm(self):
        return 1.0 + torch.exp(self.fwhm_raw)

    def responses(self):
        return gaussian_response(self.wavelengths, self.centres, self.fwhm / FWHM_PER_SIGMA)

    def forward(self, spectra):
        responses = self.responses()
        bands = spectra @ responses.T / responses.sum(dim=1)
        if self.training and self.noise > 0:
            sd = self.noise * torch.sqrt(self.reference_fwhm / self.fwhm)
            bands = bands + torch.randn_like(bands) * sd
        return bands


class SensorClassifier(nn.Module):
    def __init__(self, sensor, n_classes, hidden=64):
        super().__init__()
        self.sensor = sensor
        k = sensor.centre_logit.numel()
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(k), nn.Linear(k, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_classes),
        )

    def forward(self, spectra):
        return self.classifier(self.sensor(spectra))


def penalties(sensor, absorption, min_fwhm_nm):
    responses = sensor.responses()
    absorbed = (responses * absorption).sum(dim=1) / responses.sum(dim=1)
    too_narrow = torch.clamp(min_fwhm_nm - sensor.fwhm, min=0) ** 2
    return absorbed.sum(), too_narrow.sum()


def train(spectra, labels, wavelengths_nm, k, n_classes, *, seed=0, epochs=150, batch_size=4096,
          alpha=1.0, beta=1e-3, min_fwhm_nm=10.0, noise=0.005, sensor_lr=0.05, classifier_lr=2e-3,
          init_fwhm_nm=300.0, absorption=None, device="cpu", log_every=0):
    """Train a sensor and classifier; returns the model (in eval mode) and a history dict.

    Classes are weighted by inverse frequency so rare classes (ice cloud) count.
    Initial centres are random, drawn outside the absorption windows. Bands start
    wide (init_fwhm_nm) so that each sees, and gets gradient from, a broad part
    of the spectrum; they can narrow during training.
    """
    generator = np.random.default_rng(seed)
    torch.manual_seed(seed)
    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    if absorption is None:
        absorption = window_absorption(wavelengths)
    absorption = torch.as_tensor(absorption, dtype=torch.float32, device=device)
    allowed = wavelengths[(absorption.cpu().numpy() < 0.01) & (wavelengths > wavelengths.min() + 30)
                          & (wavelengths < wavelengths.max() - 30)]
    init = np.sort(generator.choice(allowed, k, replace=False))
    sensor = LearnableSensor(wavelengths, k, init_centres_nm=init, init_fwhm_nm=init_fwhm_nm, noise=noise)
    model = SensorClassifier(sensor, n_classes).to(device)

    x = torch.as_tensor(spectra, dtype=torch.float32, device=device)
    y = torch.as_tensor(labels, dtype=torch.int64, device=device)
    counts = torch.bincount(y, minlength=n_classes).float()
    weights = torch.where(counts > 0, counts.sum() / (n_classes * counts.clamp(min=1)), torch.zeros_like(counts))
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimiser = torch.optim.Adam([
        {"params": sensor.parameters(), "lr": sensor_lr},
        {"params": model.classifier.parameters(), "lr": classifier_lr},
    ])
    history = {"loss": [], "cross_entropy": []}
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x), device=device)
        total, total_ce = 0.0, 0.0
        for start in range(0, len(x), batch_size):
            batch = order[start:start + batch_size]
            ce = loss_fn(model(x[batch]), y[batch])
            absorbed, narrow = penalties(sensor, absorption, min_fwhm_nm)
            loss = ce + alpha * absorbed + beta * narrow
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(batch)
            total_ce += float(ce.detach()) * len(batch)
        history["loss"].append(total / len(x))
        history["cross_entropy"].append(total_ce / len(x))
        if log_every and (epoch + 1) % log_every == 0:
            print(f"  epoch {epoch + 1}: loss {history['loss'][-1]:.4f}, centres "
                  + " ".join(f"{c:.0f}" for c in sensor.centres.tolist()), flush=True)
    model.eval()
    return model, history


@torch.inference_mode()
def predict(model, spectra, device="cpu", batch_size=65536):
    x = torch.as_tensor(spectra, dtype=torch.float32, device=device)
    return torch.cat([model(x[i:i + batch_size]).argmax(dim=1) for i in range(0, len(x), batch_size)]).cpu().numpy()
