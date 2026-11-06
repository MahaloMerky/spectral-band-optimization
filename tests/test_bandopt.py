import itertools
import math

import numpy as np
import pytest
import torch

from bandopt.combinatorics import chunk_range, unrank_subsets
from bandopt.exact import search_rank_range
from bandopt.sensors import candidate_band_mask, gaussian_response, simulate_bands
from bandopt.separability import pairwise_jm, worst_case_jm
from bandopt.stats import class_statistics


def random_statistics(n_classes=4, n_bands=10, seed=0):
    generator = torch.Generator().manual_seed(seed)
    mu = torch.rand(n_classes, n_bands, generator=generator, dtype=torch.float64)
    factors = torch.randn(n_classes, n_bands, n_bands, generator=generator, dtype=torch.float64)
    sigma = factors @ factors.mT / n_bands + 0.05 * torch.eye(n_bands, dtype=torch.float64)
    return mu, sigma


def reference_jm(mu_a, mu_b, sigma_a, sigma_b):
    """Eq. 2 written directly with dense NumPy linear algebra."""
    mean_sigma = 0.5 * (sigma_a + sigma_b)
    difference = mu_a - mu_b
    bhattacharyya = difference @ np.linalg.solve(mean_sigma, difference) / 8 + 0.5 * np.log(
        np.linalg.det(mean_sigma) / np.sqrt(np.linalg.det(sigma_a) * np.linalg.det(sigma_b))
    )
    return 2 * (1 - np.exp(-bhattacharyya))


def test_pairwise_jm_matches_direct_formula():
    mu, sigma = random_statistics()
    subset = [1, 4, 7]
    result = pairwise_jm(mu, sigma, torch.tensor([subset]))[0].numpy()

    index = np.ix_(subset, subset)
    expected = [
        reference_jm(
            mu[a, subset].numpy(), mu[b, subset].numpy(),
            sigma[a].numpy()[index], sigma[b].numpy()[index],
        )
        for a, b in itertools.combinations(range(mu.shape[0]), 2)
    ]
    np.testing.assert_allclose(result, expected)


def test_jm_analytic_single_band_case():
    mu = torch.tensor([[0.0], [2.0]], dtype=torch.float64)
    sigma = torch.ones(2, 1, 1, dtype=torch.float64)
    result = pairwise_jm(mu, sigma, torch.tensor([[0]])).item()
    assert result == pytest.approx(2 * (1 - math.exp(-0.5)))


def test_worst_case_is_smallest_pair():
    mu, sigma = random_statistics()
    subsets = torch.tensor([[0, 1], [2, 3]])
    scores = worst_case_jm(mu, sigma, subsets)
    assert torch.equal(scores, pairwise_jm(mu, sigma, subsets).amin(dim=1))


def test_non_positive_definite_subset_is_nan():
    mu, sigma = random_statistics(n_bands=3)
    sigma[0] = torch.ones(3, 3, dtype=torch.float64)  # Singular
    assert worst_case_jm(mu, sigma, torch.tensor([[0, 1]])).isnan().all()


@pytest.mark.parametrize("n, k", [(8, 1), (8, 3), (12, 5)])
def test_unrank_visits_every_subset_once(n, k):
    total = math.comb(n, k)
    subsets = unrank_subsets(torch.arange(total), n, k).tolist()
    assert sorted(map(tuple, subsets)) == list(itertools.combinations(range(n), k))


def test_unrank_handles_largest_planned_search():
    n, k = 285, 6
    last = unrank_subsets(torch.tensor([math.comb(n, k) - 1]), n, k)
    assert last.tolist() == [list(range(n - k, n))]


def test_chunks_partition_search_space():
    total, n_tasks = 1003, 7
    ranges = [chunk_range(total, n_tasks, task) for task in range(n_tasks)]
    assert ranges[0][0] == 0 and ranges[-1][1] == total
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))


def test_chunked_search_finds_brute_force_optimum():
    mu, sigma = random_statistics()
    k = 3
    all_subsets = torch.tensor(list(itertools.combinations(range(mu.shape[1]), k)))
    expected = worst_case_jm(mu, sigma, all_subsets).max().item()

    total = all_subsets.shape[0]
    chunks = [
        search_rank_range(mu, sigma, k, *chunk_range(total, 5, task), batch_size=7, top=3)
        for task in range(5)
    ]
    assert max(chunk.scores[0].item() for chunk in chunks) == pytest.approx(expected)
    assert all(chunk.n_failed == 0 for chunk in chunks)
    assert all(torch.all(chunk.scores[:-1] >= chunk.scores[1:]) for chunk in chunks)


def test_search_counts_failed_subsets():
    mu, sigma = random_statistics(n_bands=4)
    sigma[0] = torch.ones(4, 4, dtype=torch.float64)
    result = search_rank_range(mu, sigma, 2, 0, math.comb(4, 2))
    assert result.n_failed == math.comb(4, 2)
    assert result.scores.numel() == 0


def test_simulate_bands_preserves_flat_spectrum_and_averages():
    wavelengths = torch.linspace(400, 2500, 200, dtype=torch.float64)
    responses = gaussian_response(
        wavelengths,
        torch.tensor([650.0, 1600.0], dtype=torch.float64),
        torch.tensor([20.0, 40.0], dtype=torch.float64),
    )
    flat = torch.full((3, 200), 0.3, dtype=torch.float64)
    assert torch.allclose(simulate_bands(flat, responses), torch.tensor(0.3, dtype=torch.float64))

    boxcar = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
    spectrum = torch.tensor([0.2, 0.4, 0.9, 0.9], dtype=torch.float64)
    assert simulate_bands(spectrum, boxcar).item() == pytest.approx(0.3)


def test_candidate_mask_removes_water_vapour_windows():
    wavelengths = torch.tensor([1000.0, 1350.0, 1400.0, 1500.0, 1900.0, 2200.0])
    assert candidate_band_mask(wavelengths).tolist() == [True, False, False, True, False, True]


def test_class_statistics_are_positive_definite_with_correlated_bands():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=(60, 1))
    spectra = np.hstack([signal + 1e-6 * rng.normal(size=(60, 1)) for _ in range(30)])
    labels = np.repeat([3, 8], 30)
    mu, sigma, classes = class_statistics(spectra, labels)
    assert classes.tolist() == [3, 8]
    assert mu.shape == (2, 30)
    assert np.all(np.linalg.eigvalsh(sigma) > 0)
