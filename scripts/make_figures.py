"""Make the paper's figures and tables from the pipeline outputs.

Writes vector PDFs to paper/figures/ and LaTeX snippets to paper/generated/,
which main.tex includes. Rerun it whenever the data or results change; any
figure whose inputs are missing is skipped with a message.

    python scripts/make_figures.py --config configs/scenes.toml \
        --pixels D:/bandopt-data/processed/pixels.npz \
        --stats-dir D:/bandopt-data/processed/stats \
        --results-dir D:/bandopt-data/results

Results are read from <results-dir>/exact_<stats name>/k<k>/top_subsets.csv
and timings from <results-dir>/exact_*/timings.csv.
"""

import argparse
import csv
import importlib.util
import json
import math
import tomllib
import urllib.request
from pathlib import Path

import numpy as np

from bandopt import emit
from bandopt.labels import CLASSES
from bandopt.sensors import WATER_VAPOUR_WINDOWS_NM
from bandopt.timing import read_rows, summarize_search_log

ROOT = Path(__file__).resolve().parents[1]
LAND_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_110m_land.geojson"
)
CLASS_COLOURS = {
    "vegetation": "#2e7d32", "urban": "#6d4c41", "bare_soil": "#c17d3c", "sand": "#e0b84c",
    "water": "#1565c0", "liquid_cloud": "#78909c", "ice_cloud": "#d81b60",
}
CLASS_NAMES = {
    "vegetation": "Vegetation", "urban": "Urban", "bare_soil": "Bare soil", "sand": "Sand",
    "water": "Water", "liquid_cloud": "Liquid cloud", "ice_cloud": "Ice cloud",
}
ROLE_COLOURS = {
    "surface": "#455a64", "liquid cloud": "#1e88e5", "ice cloud": "#d81b60",
    "ice and liquid cloud": "#8e24aa",
}


def plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    pyplot.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    })
    return pyplot


def latex_escape(text):
    return str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace(
        "&", r"\&").replace("%", r"\%").replace("#", r"\#")


def shade_windows(ax, unit=1000.0):
    for low, high in WATER_VAPOUR_WINDOWS_NM:
        ax.axvspan(low / unit, high / unit, color="#b0bec5", alpha=0.35, lw=0)


def load_scenes(config_path):
    with open(config_path, "rb") as file:
        config = tomllib.load(file)
    base = Path(config_path).parent
    for scene in config["scene"]:
        for key in ("radiance", "observation", "mask", "clouds"):
            if key in scene:
                scene[key] = (base / scene[key]).resolve()
    return config, config["scene"]


def draw_clouds_module():
    spec = importlib.util.spec_from_file_location("draw_clouds", ROOT / "scripts" / "draw_clouds.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- figures

def scenes_map(scenes, out, cache):
    pyplot = plt()
    land_path = cache / "ne_110m_land.geojson"
    if not land_path.exists():
        land_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LAND_URL, land_path)
    land = json.loads(land_path.read_text())

    fig, ax = pyplot.subplots(figsize=(7.2, 3.4))
    for feature in land["features"]:
        geometry = feature["geometry"]
        polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        for polygon in polygons:
            x, y = np.array(polygon[0]).T
            ax.fill(x, y, color="#eceff1", ec="#b0bec5", lw=0.4)
    seen, placed = set(), []
    for number, scene in enumerate(scenes, 1):
        west, south, east, north = emit.read_grid(scene["radiance"]).bounds()
        lon, lat = (west + east) / 2, (south + north) / 2
        role = scene.get("role", "surface")
        ax.plot(lon, lat, "o", ms=5, mec="white", mew=0.8, color=ROLE_COLOURS[role],
                label=None if role in seen else role.capitalize())
        seen.add(role)
        # The first scene of a crowded group is labelled beside its marker; the
        # others get a leader line to a label stacked off to the west.
        neighbours = [(x, y) for x, y in placed if abs(x - lon) < 12 and abs(y - lat) < 8]
        near = len(neighbours)
        placed.append((lon, lat))
        if near == 0:
            ax.annotate(str(number), (lon, lat), (5, 2), textcoords="offset points",
                        fontsize=6.5, color="#263238", weight="bold")
        else:
            anchor_lon, anchor_lat = neighbours[0]
            ax.annotate(str(number), (lon, lat), (anchor_lon - 20, anchor_lat - 6 * near), fontsize=6.5,
                        color="#263238", weight="bold", ha="right", va="center",
                        arrowprops={"arrowstyle": "-", "color": "#90a4ae", "lw": 0.6})
    ax.set_xlim(-130, 150)
    ax.set_ylim(-45, 58)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)
    fig.savefig(out / "scenes_map.pdf")
    pyplot.close(fig)


def class_spectra(table, out):
    pyplot = plt()
    spectra, labels, wl = table["spectra"], table["labels"], table["wavelengths_nm"] / 1000
    fig, ax = pyplot.subplots(figsize=(7.2, 3.6))
    shade_windows(ax)
    for index, name in enumerate(CLASSES):
        rows = spectra[labels == index]
        if not len(rows):
            continue
        mean, sd = rows.mean(axis=0), rows.std(axis=0)
        ax.plot(wl, mean, color=CLASS_COLOURS[name], lw=1.4, label=f"{CLASS_NAMES[name]} ({len(rows):,})")
        ax.fill_between(wl, mean - sd, mean + sd, color=CLASS_COLOURS[name], alpha=0.12, lw=0)
    ax.set_xlim(wl.min(), wl.max())
    ax.set_ylim(0, None)
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("TOA reflectance")
    ax.legend(ncol=2, fontsize=7, frameon=False, loc="upper right")
    fig.savefig(out / "class_spectra.pdf")
    pyplot.close(fig)


def cloud_phase(config, scenes, table, out, scene_ids=("india_storm", "oklahoma_storm")):
    pyplot = plt()
    dc = draw_clouds_module()
    solar = Path(config["_base"]) / config["solar_spectrum"]
    chosen = [s for s in scenes if s["id"] in scene_ids and "clouds" in s]
    if not chosen:
        print("  no cloud scenes with polygons; skipped")
        return
    fig = pyplot.figure(figsize=(7.2, 2.9))
    width = 0.28
    for i, scene in enumerate(chosen):
        grid = emit.read_grid(scene["radiance"])
        reflectance = dc.read_reflectance_bands(scene["radiance"], scene["observation"], solar, dc.PHASE_NM)
        image = dc.to_display(grid.to_grid(np.nan_to_num(reflectance).astype(np.float32)), 0.9)
        image = image + np.where(grid.valid[..., None], 0, 1.0)
        ax = fig.add_axes([0.0 + i * (width + 0.01), 0.02, width, 0.9])
        height, w = grid.shape
        ax.imshow(np.clip(image, 0, 1), extent=(0, w, height, 0), interpolation="antialiased")
        polygons = dc.geojson_to_polygons(json.loads(Path(scene["clouds"]).read_text()), grid.geotransform)
        for label, corners in polygons:
            xs, ys = zip(*(list(corners) + [corners[0]]))
            ax.plot(xs, ys, color="#ffd600" if label == "ice_cloud" else "#00b0ff", lw=1.2)
        ax.set_title(f"{scene.get('site', scene['id'])}, {scene.get('date', '')}", fontsize=8)
        ax.axis("off")
    ax = fig.add_axes([0.0 + len(chosen) * (width + 0.01) + 0.09, 0.16, 0.33, 0.72])
    shade_windows(ax)
    wl = table["wavelengths_nm"] / 1000
    for name in ("liquid_cloud", "ice_cloud"):
        rows = table["spectra"][table["labels"] == CLASSES.index(name)]
        if len(rows):
            ax.plot(wl, rows.mean(axis=0), color=CLASS_COLOURS[name], lw=1.4, label=CLASS_NAMES[name])
            ax.fill_between(wl, *np.percentile(rows, [25, 75], axis=0), color=CLASS_COLOURS[name], alpha=0.15, lw=0)
    for nm in dc.PHASE_NM:
        ax.axvline(nm / 1000, color="#90a4ae", ls="--", lw=0.7)
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("TOA reflectance")
    ax.set_xlim(wl.min(), wl.max())
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(out / "cloud_phase.pdf")
    pyplot.close(fig)


def label_quality(table, out):
    pyplot = plt()
    spectra, labels, scenes, wl = table["spectra"], table["labels"], table["scene_ids"], table["wavelengths_nm"]
    band = {nm: int(np.abs(wl - nm).argmin()) for nm in (650, 860, 1650)}
    fig, axes = pyplot.subplots(1, 2, figsize=(7.2, 3.0))
    panels = [
        ("water", spectra[:, band[1650]], "Water: TOA reflectance at 1.65 µm", 0.05),
        ("vegetation", (spectra[:, band[860]] - spectra[:, band[650]])
         / (spectra[:, band[860]] + spectra[:, band[650]]), "Vegetation: NDVI", 0.4),
    ]
    for ax, (name, value, title, line) in zip(axes, panels):
        member = labels == CLASSES.index(name)
        ids = [s for s in np.unique(scenes[member]) if (member & (scenes == s)).sum() >= 50]
        data = [value[member & (scenes == s)] for s in ids]
        ax.boxplot(data, orientation="horizontal", showfliers=False, widths=0.6,
                   medianprops={"color": CLASS_COLOURS[name]})
        ax.set_yticks(range(1, len(ids) + 1), [s.replace("_", " ") for s in ids], fontsize=6.5)
        ax.axvline(line, color="#e53935", ls="--", lw=0.8)
        ax.set_title(title, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "label_quality.pdf")
    pyplot.close(fig)


def read_selection(results_dir, stats_name, k):
    path = results_dir / f"exact_{stats_name}" / f"k{k}" / "top_subsets.csv"
    if not path.exists():
        return None
    with path.open() as file:
        row = next(csv.DictReader(file))
    return {
        "wavelengths": [float(v) for v in row["wavelengths_nm"].split()],
        "min_jm": float(row["min_jm"]),
        "mean_jm": float(row["mean_jm"]),
    }


def selected_bands(table, results_dir, stats_names, ks, out):
    pyplot = plt()
    rows = [(name, k, read_selection(results_dir, name, k)) for k in ks for name in stats_names]
    rows = [r for r in rows if r[2]]
    if not rows:
        print("  no search results; skipped")
        return rows
    wl = table["wavelengths_nm"] / 1000
    fig, (top, bottom) = pyplot.subplots(
        2, 1, figsize=(7.2, 3.2 + 0.18 * len(rows)), sharex=True,
        gridspec_kw={"height_ratios": [3, 0.6 + 0.25 * len(rows)], "hspace": 0.08},
    )
    for ax in (top, bottom):
        shade_windows(ax)
    for index, name in enumerate(CLASSES):
        member = table["labels"] == index
        if member.any():
            top.plot(wl, table["spectra"][member].mean(axis=0), color=CLASS_COLOURS[name], lw=1.0,
                     label=CLASS_NAMES[name])
    top.set_ylim(0, None)
    top.set_ylabel("TOA reflectance")
    top.legend(ncol=4, fontsize=6.5, frameon=False, loc="upper right")
    for row_index, (name, k, selection) in enumerate(rows):
        bottom.plot([w / 1000 for w in selection["wavelengths"]], [row_index] * k, "|", ms=8, mew=2,
                    color="#212121" if name == "all" else "#78909c")
    bottom.set_yticks(range(len(rows)), [f"k={k}, {name}" for name, k, _ in rows], fontsize=6.5)
    bottom.set_ylim(len(rows) - 0.5, -0.5)
    bottom.spines["left"].set_visible(False)
    bottom.tick_params(axis="y", length=0)
    bottom.set_xlim(wl.min(), wl.max())
    bottom.set_xlabel("Wavelength (µm)")
    fig.savefig(out / "selected_bands.pdf")
    pyplot.close(fig)
    return rows


def timing(results_dir, out, n_bands):
    logs = sorted(results_dir.glob("exact_*/timings.csv"))
    if not logs:
        print("  no timing logs; skipped")
        return []
    summaries = summarize_search_log([row for log in logs for row in read_rows(log)])
    pyplot = plt()
    fig, ax = pyplot.subplots(figsize=(4.2, 2.8))
    measured = {}
    for s in summaries:
        measured.setdefault(s["device"], {})[s["k"]] = s
    for device, by_k in measured.items():
        ks = sorted(by_k)
        ax.plot(ks, [by_k[k]["projected_seconds"] for k in ks], "o-", label=f"{device} (measured)")
        last = by_k[ks[-1]]
        extra = list(range(ks[-1] + 1, 7))
        projected = [math.comb(n_bands, k) / last["subsets_per_s"] for k in extra]
        if extra:
            ax.plot([ks[-1]] + extra, [last["projected_seconds"]] + projected, "o--", color=ax.lines[-1].get_color(),
                    mfc="white", label="extrapolated at the k=%d rate" % ks[-1])
    for seconds, text in ((60, "1 min"), (3600, "1 h"), (86400, "1 day")):
        ax.axhline(seconds, color="#cfd8dc", lw=0.6)
        ax.text(3.0, seconds * 1.15, text, fontsize=6.5, color="#78909c")
    ax.set_yscale("log")
    ax.set_xlabel("Bands k")
    ax.set_ylabel("Full exact search (s)")
    ax.set_xticks(range(3, 7))
    ax.legend(fontsize=6.5, frameon=False)
    fig.savefig(out / "timing.pdf")
    pyplot.close(fig)
    return summaries


def read_csv(path):
    if not path.exists():
        return None
    with path.open() as file:
        return list(csv.DictReader(file))


METHOD_COLOURS = {"SFS": "#8d6e63", "SFFS": "#6d4c41", "mRMR": "#9e9e9e", "GL": "#fb8c00", "SA": "#1e88e5", "GA": "#43a047"}


def heuristics_figure(results_dir, out):
    rows = read_csv(results_dir / "heuristics_all.csv")
    if not rows:
        print("  no heuristics_all.csv; skipped")
        return None
    pyplot = plt()
    fig, ax = pyplot.subplots(figsize=(6.0, 3.2))
    ks = sorted({int(r["k"]) for r in rows})
    methods = [m for m in METHOD_COLOURS if any(r["method"] == m for r in rows)]
    width = 0.8 / len(methods)
    for i, method in enumerate(methods):
        for k in ks:
            values = [float(r["min_jm"]) for r in rows if r["method"] == method and int(r["k"]) == k]
            x = k + (i - (len(methods) - 1) / 2) * width
            if len(values) > 1:
                ax.boxplot([values], positions=[x], widths=width * 0.8, showfliers=False, patch_artist=True,
                           boxprops={"facecolor": METHOD_COLOURS[method], "alpha": 0.5},
                           medianprops={"color": METHOD_COLOURS[method]})
            ax.plot([x], [max(values)], "o", ms=3.5, color=METHOD_COLOURS[method],
                    label=method if k == ks[0] else None)
    exact = {int(r["k"]): float(r["exact_min_jm"]) for r in rows if r["exact_min_jm"]}
    if exact:
        ax.plot(list(exact), list(exact.values()), "k*", ms=9, label="Exact optimum")
    ax.set_xticks(ks)
    ax.set_xlabel("Bands k")
    ax.set_ylabel("Worst-case JM")
    ax.legend(fontsize=7, frameon=False, ncol=3)
    fig.savefig(out / "heuristics.pdf")
    pyplot.close(fig)
    return rows


def method_tables(results_dir, generated):
    """Write the heuristics, OLI, classifier and differentiable tables when their results exist."""
    rows = read_csv(results_dir / "heuristics_all.csv")
    if rows:
        lines = [r"\begin{tabular}{rlrrrr}", r"\toprule",
                 r"$k$ & Method & Best JM & Median JM & Gap of best & Time/run (s) \\", r"\midrule"]
        for k in sorted({int(r["k"]) for r in rows}):
            for method in METHOD_COLOURS:
                runs = [r for r in rows if int(r["k"]) == k and r["method"] == method]
                if not runs:
                    continue
                jm = [float(r["min_jm"]) for r in runs]
                best = max(runs, key=lambda r: float(r["min_jm"]))
                gap = f"{float(best['gap']):.3f}" if best["gap"] else "--"
                seconds = np.mean([float(r["seconds"]) for r in runs])
                lines.append(f"{k} & {method} & {max(jm):.3f} & {np.median(jm):.3f} & {gap} & {seconds:.1f} \\\\")
            lines.append(r"\addlinespace")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (generated / "heuristics_table.tex").write_text("\n".join(lines) + "\n")

    rows = read_csv(results_dir / "sensors_all.csv")
    if rows:
        lines = [r"\begin{tabular}{rlrrr}", r"\toprule",
                 r"$k$ & Best OLI subset & OLI JM & EMIT optimum & Gap \\", r"\midrule"]
        for r in rows:
            emit_jm = f"{float(r['emit_min_jm']):.3f}" if r["emit_min_jm"] else "--"
            gap = f"{float(r['gap']):.3f}" if r["gap"] else "--"
            bands = latex_escape(r["oli_bands"].replace(" + ", ", "))
            lines.append(f"{r['k']} & {bands} & {float(r['oli_min_jm']):.3f} & {emit_jm} & {gap} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (generated / "oli_table.tex").write_text("\n".join(lines) + "\n")

    rows = read_csv(results_dir / "classifiers.csv")
    if rows:
        features = list(dict.fromkeys(r["features"] for r in rows))
        classifiers = list(dict.fromkeys(r["classifier"] for r in rows))
        lines = [r"\begin{tabular}{l" + "r" * len(classifiers) + "}", r"\toprule",
                 "Features & " + " & ".join(classifiers) + r" \\", r"\midrule"]
        for feature in features:
            cells = []
            for classifier in classifiers:
                values = [float(r["macro_f1"]) for r in rows if r["features"] == feature and r["classifier"] == classifier]
                cells.append(f"{np.mean(values):.3f} $\\pm$ {np.std(values):.3f}" if values else "--")
            lines.append(latex_escape(feature) + " & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (generated / "classifier_table.tex").write_text("\n".join(lines) + "\n")

    rows = read_csv(results_dir / "differentiable.csv")
    if rows:
        lines = [r"\begin{tabular}{rrllrr}", r"\toprule",
                 r"Fold & $k$ & Centres (nm) & FWHM (nm) & Macro-F1 & Worst-case JM \\", r"\midrule"]
        for r in rows:
            if r.get("chosen", "1") != "1":
                continue
            centres = ", ".join(f"{float(c):.0f}" for c in r["centres_nm"].split())
            fwhm = ", ".join(f"{float(w):.0f}" for w in r["fwhm_nm"].split())
            lines.append(f"{r['fold']} & {r['k']} & {centres} & {fwhm} & {float(r['macro_f1']):.3f} & "
                         f"{float(r['min_jm']):.3f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (generated / "differentiable_table.tex").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------- tables

def write_tables(scenes, table, results_rows, timings, n_bands, generated):
    counts = {}
    for scene_id, label in zip(table["scene_ids"], table["labels"]):
        counts.setdefault(str(scene_id), np.zeros(len(CLASSES), int))[label] += 1

    lines = [r"\begin{tabular}{rllrrl" + "r" * len(CLASSES) + "}", r"\toprule",
             r"\# & Scene & Site & Date & Cloud & Role & " + " & ".join(
                 r"\rotatebox{70}{" + CLASS_NAMES[c] + "}" for c in CLASSES) + r" \\", r"\midrule"]
    for number, scene in enumerate(scenes, 1):
        c = counts.get(scene["id"], np.zeros(len(CLASSES), int))
        lines.append(" & ".join([
            str(number), r"\texttt{" + latex_escape(scene["id"]) + "}", latex_escape(scene.get("site", "")),
            scene.get("date", ""), f"{scene.get('cloud_cover', '')}\\%", scene.get("role", ""),
            *[f"{v:,}" if v else "--" for v in c],
        ]) + r" \\")
    total = table["labels"]
    lines += [r"\midrule", r"\multicolumn{6}{l}{Total} & " + " & ".join(
        f"{(total == i).sum():,}" for i in range(len(CLASSES))) + r" \\", r"\bottomrule", r"\end{tabular}"]
    (generated / "scenes_table.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{rrl}", r"\toprule", r"$k$ & $\binom{n}{k}$ & Approach \\", r"\midrule"]
    approach = {3: "Exact", 4: "Exact", 5: "Exact, chunked and resumable (GPU)", 6: "Stretch goal",
                10: "Heuristics only"}
    for k in (3, 4, 5, 6, 10):
        mantissa, exponent = f"{math.comb(n_bands, k):.1e}".split("e")
        lines.append(f"{k} & ${mantissa}\\times10^{{{int(exponent)}}}$ & {approach[k]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (generated / "search_space.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{rllrr}", r"\toprule",
             r"$k$ & Statistics & Selected bands (nm) & Worst-case JM & Mean JM \\", r"\midrule"]
    for name, k, sel in results_rows:
        lines.append(f"{k} & {latex_escape(name)} & " + ", ".join(f"{w:.0f}" for w in sel["wavelengths"])
                     + f" & {sel['min_jm']:.3f} & {sel['mean_jm']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (generated / "results_table.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{rrrrl}", r"\toprule",
             r"$k$ & Subsets & Compute time & Subsets/s & Device \\", r"\midrule"]
    for s in timings:
        lines.append(f"{s['k']} & {s['subsets']:,} & {s['seconds']:.1f} s & {s['subsets_per_s']:,.0f} & "
                     f"{latex_escape(s['device'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (generated / "timing_table.tex").write_text("\n".join(lines) + "\n")

    macros = {
        "nCandidates": n_bands, "nScenes": len(scenes), "nPixels": f"{len(table['labels']):,}",
        "nBandsEMIT": len(table["wavelengths_nm"]),
    }
    (generated / "numbers.tex").write_text(
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "scenes.toml")
    parser.add_argument("--pixels", type=Path, required=True)
    parser.add_argument("--stats-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "paper")
    args = parser.parse_args()

    figures, generated = args.out / "figures", args.out / "generated"
    figures.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    config, scenes = load_scenes(args.config)
    config["_base"] = args.config.parent
    table = np.load(args.pixels, allow_pickle=True)
    n_bands = int(np.load(args.stats_dir / "stats_all.npz")["mu"].shape[1])
    stats_names = ["all"] + sorted(p.stem.replace("stats_", "") for p in args.stats_dir.glob("stats_fold*.npz"))

    print("scene map");        scenes_map(scenes, figures, ROOT / "data" / "raw" / "naturalearth")
    print("class spectra");    class_spectra(table, figures)
    print("cloud phase");      cloud_phase(config, scenes, table, figures)
    print("label quality");    label_quality(table, figures)
    print("selected bands");   rows = selected_bands(table, args.results_dir, stats_names, (3, 4), figures)
    print("timing");           summaries = timing(args.results_dir, figures, n_bands)
    print("heuristics");       heuristics_figure(args.results_dir, figures)
    print("tables");           write_tables(scenes, table, rows, summaries, n_bands, generated)
    method_tables(args.results_dir, generated)
    print(f"Wrote figures to {figures} and tables to {generated}")


if __name__ == "__main__":
    main()
