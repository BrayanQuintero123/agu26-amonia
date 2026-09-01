# Semi-automatic quantification of CH₄ plumes in Tanager

Detecting a methane plume is already routine; turning it into a defensible kg/h is not.
This repo is an open pipeline that goes from a **Tanager** radiance cube to a per-plume flow
rate, leaving the human exactly where judgment is needed and nowhere else.

**The algorithm delineates, the operator confirms.**

```
L1 radiance  →  matched filter (ppm·m)  →  wavelet anomaly  →  k·σ candidates
                                                                     ↓
                             Q (kg/h)  ←  IME · Ueff / L  ←  connected-component mask
```

## Start here

A single command, which picks the path on its own based on what each scene provides:

```bash
python auto_pipeline/quantify_tanager.py tanager/*_ortho_radiance_hdf5.h5
```

| If the scene includes… | Path | What happens |
|---|---|---|
| `ql_ch4_json` **with** plumes | automatic | Seeded from the official sources, no clicks needed. |
| no GeoJSON, or empty | picker | Proposes candidates on the wavelet map; you pick with a click. |

`--dry-run` reports what it would do without running anything. `--mode gt` / `--mode pick` force one path or the other.

### The picker, step by step

```bash
python auto_pipeline/pick_plumes_wavelet.py <scene_ortho_radiance_hdf5.h5>
```

1. Runs **both** matched filters and produces 4 panels: wavelet anomaly and physical enhancement, for each.
2. Proposes as a candidate **every cluster** above `k·σ` with ≥ `--min-pix` pixels, taking the
   **union of both** MFs so no plume is missed because one retrieval didn't see it.
   Each candidate comes numbered, color-coded by which one detected it (yellow = column-wise MF only,
   blue = group-wise MF only, green = both), with **its Google Maps link**, so you can rule out
   false positives before quantifying anything.
3. One click grows the mask via connected component. You accept, adjust σ, or reject.
4. Quantifies with both MFs on the **same** mask and the **same** wind (ERA5 by default,
   one per plume at its own position) and saves CSV + PNG.

## Results

`results/` contains the 20 Carbon Mapper plumes from two Tanager scenes, quantified by
this pipeline and checked against the official product.

### Why comparing Q against Q doesn't mean what it seems to

Solving `Q = IME·3600·Ueff/L` over the 20 plumes in the GeoJSON gives, **exactly and without
exception**, `Ueff = u10` and `L = fetch`. The repo's classic parameterization
(`Ueff = a·u10 + b`, Guanter et al. 2021 / Roger et al. 2024) is 1.8–2.4× lower at those wind speeds.

Re-deriving Q under four defensible combinations of mask and length scale:

| mask | L | LARS/official | ADV/official | ADV/LARS |
|---|---|---|---|---|
| quicklook footprint | fetch | 2.01 | 2.43 | 1.26 |
| quicklook footprint | √area | 2.07 | 2.37 | 1.14 |
| own k·σ growth | √area | 2.07 | 2.06 | 1.00 |
| own k·σ growth | fetch | 1.46 | 0.92 | 0.63 |

The ratio against the official value ranges between **0.92× and 2.43×** without touching a single
data point. That's not measurement error, it's convention — and no comparison between inventories
means anything unless the mask and the Ueff are stated.

### The two matched filters

Both share the target `t = mu · k` (with `k` from the libRadtran LUT), so their outputs are
in ppm·m and comparable. Only the way the **background** is estimated changes:

| | Background |
|---|---|
| **column-wise** (`Retrieval_methods.AT_MF`) | one covariance per column, `pinv`, unregularized |
| **group-wise** (`scripts/mf_advanced.py`) | columns grouped (10–30) according to striping; within each group, PCA(3) + k-means, one covariance per cluster with shrinkage |

With the mask held fixed, the group-wise one recovers 14–26 % more mass, but:

- SNR is **worse** in both scenes (0.71 vs 1.12, and 3.10 vs 3.97)
- background tail 4–9× heavier (p99.9 = 3251 and 8171 vs 804 and 882 ppm·m)
- **fails to delineate 3 of 20 plumes** at the standard 2σ threshold
- negative Q on one real plume

Grouping columns to get more samples per covariance doesn't make up for the noise it introduces.

```bash
python auto_pipeline/compare_mf.py <scenes.h5> --per-plume --slim
```

### Quantification convention

| Flag | Options | Default |
|---|---|---|
| `--ueff-mode` | `cm` (Ueff = u10, Carbon Mapper's) · `lars` (a·u10 + b) | `cm` |
| `--l-mode` | `sqrt-area` · `fetch` · `hull` | `sqrt-area` |
| `--mask-mode` | `footprint` (isolates the retrieval) · `grow` (full pipeline) | `footprint` |

`sqrt-area` = `√(N·gsd²)`, which is what `extract_Q` already used, and against the official
`fetch` of the 20 plumes gives a median of **0.92**. The convex hull gives 1.73 and spikes up to
5× on elongated plumes, so it isn't comparable with the official product.

## Why Tanager

- **30 m** versus EMIT's 60 m: 4× the pixels per plume. Mask quality feeds directly into Q
  twice over, via IME and via L.
- **Coverage.** EMIT only observes between 52°N and 52°S — that's the ISS's inclination.
  This repo's Yamal scene is at **66.5°N**, 14° outside that limit: its 8 plumes
  are unobservable from the ISS, not for scheduling reasons but because of orbit.

  > "During its planned one-year mission, EMIT will collect data over Earth's dust-source
  > regions […] between 52° north and south latitude."
  > — Smith, J. M. (2023), *Meet EMIT, the Newest Imaging Spectrometer*, NASA Earthdata.

## Structure

| | |
|---|---|
| `auto_pipeline/quantify_tanager.py` | **entry point**; routes GT vs picker |
| `auto_pipeline/pick_plumes_wavelet.py` | interactive picker with both MFs |
| `auto_pipeline/compare_mf.py` | comparison of both MFs against GT |
| `auto_pipeline/plumes_from_json.py` | seeds plumes from `ql_ch4_json` |
| `auto_pipeline/wavelet_den.py` | wavelet anomaly |
| `auto_pipeline/auto_plume.py` | semi-automatic delineation (EMIT and generic) |
| `scripts/mf_advanced.py` | matched filter by column groups |
| `scripts/Retrieval_methods.py` | column-wise matched filter |
| `scripts/Tanager_reader.py` | Tanager reader (ortho and basic) |
| `scripts/quant_func_v2.py` | IME, Ueff, conversion to kg |
| `results/` | the 20 quantified plumes, 4 configurations |

Other missions (EMIT, EnMAP, PRISMA, GF5, AVIRIS-NG) are still supported via
`auto_pipeline/pipeline_auto.py`; see the section at the end.

Quantifiable gases: `ch4`, `co2`, `c2h4`, `c2h2`, `nh3`.

## Installation

Conda environment with GDAL, rasterio, h5py, pyproj, scikit-image, scikit-learn, PyWavelets,
scipy, matplotlib. On Windows you have to **activate the environment**, invoking the
`python.exe` directly isn't enough: without the environment's `PATH`, LAPACK can't load its
DLLs and `np.linalg.lstsq` crashes with `0xc06d007f`.

```bash
conda activate <your-environment>
python auto_pipeline/quantify_tanager.py --help
```

The libRadtran LUTs are also required (path set in `scripts/variable_definition.py`).

## Other missions (EMIT, EnMAP, PRISMA, GF5, AVIRIS-NG)

### Main pipeline (one plume per gas)

```bash
python auto_pipeline/pipeline_auto.py "<EMIT_L1B_RAD...nc>" --gas ch4
python auto_pipeline/pipeline_auto.py "<...ortho_radiance_hdf5.h5>" --gas ch4 --mission Tanager --wind era5
python auto_pipeline/pipeline_auto.py "<RAD.nc>" --gas ch4,nh3,c2h4 -s "Carrasco"
```

Useful flags:

| Flag | Description |
|---|---|
| `--gas` | Gas or comma-separated list. |
| `--mission` | `EMIT`, `Tanager`, `EnMAP`, `PRISMA`, `GF5`, `AVIRIS-NG`. If omitted, auto-detected from the input filename. |
| `--wind {geos,era5}` | Wind source. `--era5` is shorthand for `--wind era5` and looks for the ERA5 `.nc` next to the input file. |
| `--wind-value` | Manual wind in m/s, for that run only. |
| `--sigma` | Plume threshold sigma multiplier (default 2.0). |

Plume map over a basemap image:

```bash
python auto_pipeline/plot_plume_map.py --lat <lat> --lon <lon> --gas ch4 --sigma 2.0
```

### Interactive multi-plume workflow (Tanager, CH4)

When a scene has **multiple plumes**, this script detects all of them and lets you choose
which ones to quantify:

```bash
python scripts/quantify_ch4_tanager_multi.py
```

What it does:

1. Applies the wavelet denoiser to the enhancement map.
2. Keeps local maxima above the **P99.8** percentile that also exceed **2.5σ** in the raw
   enhancement, and suppresses candidates within **1000 m** of a stronger one — so a single
   plume doesn't generate multiple points.
3. Prints a numbered table with a **Google Maps link per candidate**, so you can review
   each one and rule out false positives before marking anything.
4. Opens a window with the *enhancement* + *wavelet* panels and the marked candidates.
5. **A left click inside a plume** grows its mask automatically
   (`grow_plume_from_source`: 3×3 median, threshold at `sigma·std`, connected component).
   No need to draw polygons. Right click undoes, Enter finishes.
6. Quantifies each selected plume (IME + Ueff + wind) and shows a summary table;
   saves the results to CSV and the figure to `out_tanager/plumas_seleccionadas.png`.

Adjustments: `--pctl 99.9` (stricter), `--n-sigma 3`, `--min-dist 1500`, `--top 15`,
`--sigma 2.0`, `--wind-value 3.2` (avoids the GEOS-FP download).

## Note on Tanager

`Tanager_reader.py` auto-detects the product type:

- `ortho_radiance_hdf5` → `HDFEOS/GRIDS/HYP`, geotransform and EPSG read from the `StructMetadata`.
- `basic_radiance_hdf5` → `HDFEOS/SWATHS/HYP`, with embedded `Latitude`/`Longitude`;
  the grid is derived from the lat/lon *bounding box*.

Tanager uses a **30 m** pixel, with the same Ueff coefficients as EnMAP/PRISMA/GF5.

## Data

Data files are **not** version-controlled: Tanager's `.h5` files weigh 580–682 MB each
and EMIT's `.nc` is ~1.8 GB, well above GitHub's 100 MB limit. The `.gitignore`
excludes `*.nc`, `*.h5`, `*.tif`, `*.npy`, `TanagerScene*/`, `out_tanager/` and `output/`.

Running it also requires the original HS_tool repo and the libRadtran LUTs.
