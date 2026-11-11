import io
import itertools
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from bandopt.sensors import CIRRUS_RUN_WINDOWS_NM, candidate_band_mask
from bandopt.separability import worst_case_bhattacharyya, worst_case_jm
from bandopt.progress import ProgressBar
from bandopt.stats import scene_folds
from bandopt.timing import append_rows, read_rows, summarize_search_log
from bandopt.toa import (
    W_PER_M2_TO_UW_PER_CM2,
    band_solar_irradiance,
    radiance_to_toa_reflectance,
)

SCRIPTS = Path(__file__).parents[1] / "scripts"


def run_script(name, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *map(str, args)],
        capture_output=True, text=True,
    )


@pytest.fixture
def pixel_table(tmp_path):
    """Three classes across four scenes, with bands inside the water-vapour windows."""
    rng = np.random.default_rng(0)
    wavelengths = np.array([500.0, 800.0, 1000.0, 1380.0, 1600.0, 1900.0, 2200.0, 2400.0])
    centres = rng.uniform(0.1, 0.6, size=(3, wavelengths.size))
    spectra, labels, scenes = [], [], []
    for scene in range(4):
        for label, centre in enumerate(centres):
            spectra.append(centre + rng.normal(0, 0.05, size=(40, wavelengths.size)))
            labels.append(np.full(40, label))
            scenes.append(np.full(40, f"scene{scene}"))
    path = tmp_path / "pixels.npz"
    np.savez(
        path,
        spectra=np.concatenate(spectra),
        labels=np.concatenate(labels),
        scene_ids=np.concatenate(scenes),
        wavelengths_nm=wavelengths,
    )
    return path


def test_bhattacharyya_and_jm_rank_subsets_the_same_way():
    generator = torch.Generator().manual_seed(1)
    mu = torch.rand(4, 8, generator=generator, dtype=torch.float64)
    factors = torch.randn(4, 8, 8, generator=generator, dtype=torch.float64)
    sigma = factors @ factors.mT / 8 + 0.05 * torch.eye(8, dtype=torch.float64)
    subsets = torch.tensor(list(itertools.combinations(range(8), 3)))
    by_distance = worst_case_bhattacharyya(mu, sigma, subsets).argsort()
    by_jm = worst_case_jm(mu, sigma, subsets).argsort()
    assert torch.equal(by_distance, by_jm)


def test_bhattacharyya_separates_subsets_that_saturate_jm():
    mu = torch.tensor([[0.0, 0.0], [100.0, 200.0]], dtype=torch.float64)
    sigma = torch.eye(2, dtype=torch.float64).expand(2, 2, 2)
    subsets = torch.tensor([[0], [1]])
    jm = worst_case_jm(mu, sigma, subsets)
    assert jm[0] == jm[1] == 2.0
    distance = worst_case_bhattacharyya(mu, sigma, subsets)
    assert distance[1] > distance[0]


def test_scene_folds_keep_scenes_whole():
    labels = np.tile([0, 1], 30)
    scenes = np.repeat([f"s{i}" for i in range(6)], 10)
    folds = scene_folds(labels, scenes, 3)
    assert np.all(sum(test.astype(int) for _, test in folds) == 1)
    for train, test in folds:
        assert not set(scenes[train]) & set(scenes[test])


def test_scene_folds_reject_fold_that_loses_a_class():
    labels = np.array([0, 0, 0, 0, 1, 1])
    scenes = np.array(["a", "a", "b", "b", "c", "c"])
    with pytest.raises(ValueError, match="classes \\[1\\]"):
        scene_folds(labels, scenes, 3)


def test_toa_reflectance_of_known_radiance():
    irradiance = np.array([150.0, 50.0])
    zenith, distance = 60.0, 1.01
    cos_zenith = np.cos(np.radians(zenith))
    radiance = 0.25 * irradiance * cos_zenith / (np.pi * distance**2)
    result = radiance_to_toa_reflectance(radiance, irradiance, zenith, distance)
    np.testing.assert_allclose(result, 0.25)


def test_toa_handles_image_shapes_fill_values_and_night():
    radiance = np.full((2, 3, 4), 10.0)
    radiance[0, 0, 1] = -9999.0
    zenith = np.full((2, 3), 30.0)
    zenith[1, 2] = 95.0
    result = radiance_to_toa_reflectance(radiance, np.full(4, 100.0), zenith, np.full((2, 3), 1.0))
    assert result.shape == (2, 3, 4)
    assert np.isnan(result[0, 0, 1]) and not np.isnan(result[0, 0, 0])
    assert np.isnan(result[1, 2]).all()


def test_band_solar_irradiance_on_flat_and_uneven_grids():
    wavelengths = np.concatenate([np.arange(400, 800, 1.0), np.arange(800, 1200, 5.0)])
    result = band_solar_irradiance(wavelengths, np.full(wavelengths.size, 1.5), [600, 900], [8, 8])
    np.testing.assert_allclose(result * W_PER_M2_TO_UW_PER_CM2, 150.0)


def test_cirrus_run_keeps_only_the_first_window():
    wavelengths = torch.tensor([1380.0, 1900.0])
    assert candidate_band_mask(wavelengths, CIRRUS_RUN_WINDOWS_NM).tolist() == [True, False]


def test_stats_to_merge_pipeline(pixel_table, tmp_path):
    out = tmp_path / "stats"
    done = run_script("compute_stats.py", "--pixels", pixel_table, "--out", out, "--folds", 2)
    assert done.returncode == 0, done.stderr

    fold = np.load(out / "stats_fold0.npz")
    table = np.load(pixel_table)
    assert fold["wavelengths_nm"].tolist() == [500, 800, 1000, 1600, 2200, 2400]
    assert not set(fold["train_scenes"]) & set(fold["test_scenes"])

    # Statistics must come from training scenes only.
    train = np.isin(table["scene_ids"], fold["train_scenes"]) & (table["labels"] == 0)
    keep = np.isin(table["wavelengths_nm"], fold["wavelengths_nm"])
    np.testing.assert_allclose(fold["mu"][0], table["spectra"][train][:, keep].mean(axis=0))

    stats = out / "stats_fold0.npz"
    chunks = tmp_path / "chunks"
    # Two chunks run as separate calls, as when splitting a search between two PCs.
    for part in ("0", "1"):
        done = run_script(
            "exact_search.py", "--stats", stats, "--k", 2, "--device", "cpu",
            "--chunk-size", 8, "--chunks", part, "--out", chunks,
        )
        assert done.returncode == 0, done.stderr
    assert len(list(chunks.glob("chunk_*.npz"))) == 2

    done = run_script("merge_exact.py", chunks, "--stats", stats)
    assert done.returncode == 0, done.stderr
    assert "All 15 subsets" in done.stdout
    assert "WARNING" not in done.stdout

    lines = (chunks / "top_subsets.csv").read_text().splitlines()
    assert lines[0].startswith("tied,min_bhattacharyya,min_jm,mean_jm")
    assert len(lines) == 16

    # A wrong statistics file must not be merged silently.
    done = run_script("merge_exact.py", chunks, "--stats", out / "stats_fold1.npz")
    assert done.returncode != 0
    assert "statistics other than" in done.stderr


def test_search_resumes_and_refuses_to_mix_searches(pixel_table, tmp_path):
    out = tmp_path / "stats"
    run_script("compute_stats.py", "--pixels", pixel_table, "--out", out, "--folds", 2)
    chunks = tmp_path / "chunks"
    common = ("--k", 2, "--device", "cpu", "--chunk-size", 8, "--out", chunks)

    done = run_script("exact_search.py", "--stats", out / "stats_fold0.npz", "--chunks", "0", *common)
    assert done.returncode == 0, done.stderr
    first = (chunks / "chunk_00000.npz").stat().st_mtime_ns

    # Rerunning everything skips the finished chunk and only adds the missing one.
    done = run_script("exact_search.py", "--stats", out / "stats_fold0.npz", *common)
    assert done.returncode == 0, done.stderr
    assert (chunks / "chunk_00000.npz").stat().st_mtime_ns == first
    assert "chunk 0/" not in done.stdout and "chunk 1/" in done.stdout

    # Different statistics in the same directory must not be resumed into it.
    done = run_script("exact_search.py", "--stats", out / "stats_fold1.npz", *common)
    assert done.returncode != 0
    assert "different search" in done.stderr


def test_merge_warns_when_near_ties_may_be_cut_off(pixel_table, tmp_path):
    out = tmp_path / "stats"
    run_script("compute_stats.py", "--pixels", pixel_table, "--out", out, "--folds", 2)
    stats = out / "stats_all.npz"
    chunks = tmp_path / "chunks"
    run_script("exact_search.py", "--stats", stats, "--k", 2, "--top", 1, "--out", chunks)
    done = run_script("merge_exact.py", chunks, "--stats", stats, "--tolerance", 2)
    assert done.returncode == 0, done.stderr
    assert "WARNING" in done.stdout


def test_timing_summary_counts_each_chunk_once_and_groups_by_k():
    def row(k, chunk, subsets, seconds, device="cpu", fingerprint="a"):
        return {
            "k": str(k), "n_bands": "10", "chunk": str(chunk), "subsets": str(subsets),
            "seconds": str(seconds), "device": device, "stats_fingerprint": fingerprint,
        }

    rows = [
        row(2, 0, 30, 3.0),
        row(2, 0, 30, 1.0),  # Rerun of chunk 0: only this one counts.
        row(2, 1, 15, 0.5),
        row(2, 0, 30, 2.0, fingerprint="b"),  # Another fold is pooled.
        row(3, 0, 120, 4.0),
        row(3, 0, 120, 1.0, device="gpu"),
    ]
    summaries = summarize_search_log(rows)
    assert [(s["k"], s["device"]) for s in summaries] == [(2, "cpu"), (3, "cpu"), (3, "gpu")]

    k2 = summaries[0]
    assert k2["chunks"] == 3 and k2["subsets"] == 75
    assert k2["seconds"] == pytest.approx(3.5)
    assert k2["subsets_per_s"] == pytest.approx(75 / 3.5)
    assert k2["projected_seconds"] == pytest.approx(math.comb(10, 2) / (75 / 3.5))
    assert summaries[2]["subsets_per_s"] == pytest.approx(120)


def test_timing_log_refuses_a_different_layout(tmp_path):
    path = tmp_path / "log.csv"
    append_rows(path, [{"a": 1, "b": 2}], ["a", "b"])
    append_rows(path, [{"a": 3, "b": 4}], ["a", "b"])
    assert len(read_rows(path)) == 2
    with pytest.raises(ValueError, match="columns"):
        append_rows(path, [{"a": 1}], ["a"])


def test_search_logs_timings_and_report_summarises_them(pixel_table, tmp_path):
    out = tmp_path / "stats"
    run_script("compute_stats.py", "--pixels", pixel_table, "--out", out, "--folds", 2)
    stats = out / "stats_all.npz"
    for k in (2, 3):
        done = run_script(
            "exact_search.py", "--stats", stats, "--k", k, "--device", "cpu",
            "--chunk-size", 10, "--out", tmp_path / "exact" / f"k{k}",
        )
        assert done.returncode == 0, done.stderr
        assert "This session:" in done.stdout

    log = tmp_path / "exact" / "timings.csv"
    rows = read_rows(log)
    n_bands = int(rows[0]["n_bands"])
    for k in (2, 3):
        assert sum(int(r["subsets"]) for r in rows if r["k"] == str(k)) == math.comb(n_bands, k)

    done = run_script("merge_exact.py", tmp_path / "exact" / "k3", "--stats", stats)
    assert done.returncode == 0, done.stderr
    assert "Compute time:" in done.stdout

    summary = tmp_path / "summary.csv"
    done = run_script("timing_report.py", log, "--csv", summary)
    assert done.returncode == 0, done.stderr
    assert [r["k"] for r in read_rows(summary)] == ["2", "3"]


def test_progress_bar_counts_resumed_work_and_session_rate():
    stream = io.StringIO()
    bar = ProgressBar(total=1000, done=400, label="k=3", interval=0, stream=stream, enabled=True)
    bar.update(100)
    bar.skip(100)
    assert bar.done == 600 and bar.session_done == 100
    assert "60.0%" in bar.render()
    bar.message("chunk 1/9: done")
    bar.close()
    output = stream.getvalue()
    assert "k=3" in output and "50.0%" in output and output.endswith("\n")

    # Not a terminal: nothing is drawn.
    quiet = io.StringIO()
    bar = ProgressBar(total=10, stream=quiet)
    bar.update(5)
    bar.close()
    assert quiet.getvalue() == ""
