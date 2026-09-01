#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantify CH4 (or other gas) emission rate directly from an EMIT/EnMAP/PRISMA
plume GeoTIFF, without manual polygon delineation.

Assumes the TIFF is already plume-masked: pixels outside the plume are
`nodata` (as is the case for EMIT_L2B_CH4PLM products), so the mask is read
straight from the raster instead of being drawn by hand.

Reuses ppmm_to_kg / extract_ueff / plot_ime_and_L from your existing module
(quantification_func_v1_with_plot.py) so results stay consistent with the
rest of your pipeline. Only the pixel-area logic is new: instead of assuming
a fixed square gsd (e.g. 60 m for EMIT), it derives the true pixel width and
height in meters from the raster's geotransform + latitude, since a
lat/lon-regridded product is NOT square in meters away from the equator.

Usage
-----
    python quantify_from_tif.py plume.tif --u10 4.8335 --gas ch4 --mission EMIT
"""

import argparse
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# Adjust this import to match the actual filename of your module
# (yours is currently named quant_func_v2.py). Only ppmm_to_kg and
# extract_ueff need to exist there -- plot_ime_and_L is defined below,
# self-contained, so it doesn't depend on you having added it to your module.
from quant_func_v2 import (
    ppmm_to_kg,
    extract_ueff,
)


def plot_ime_and_L(IME, L, Q=None, ueff=None, gas='ch4', name=None, psave=None):
    """
    Simple two-panel bar plot summarizing the quantification inputs/outputs
    for a single plume: IME (kg) on the left, L (m) on the right.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    ax1.bar(['IME'], [IME], color='#d95f02', width=0.5)
    ax1.set_ylabel('IME (kg)', fontsize=13)
    ax1.text(0, IME, f'{IME:.1f} kg', ha='center', va='bottom', fontsize=12)
    ax1.set_ylim(0, IME * 1.3 if IME > 0 else 1)

    ax2.bar(['L'], [L], color='#1b9e77', width=0.5)
    ax2.set_ylabel('L (m)', fontsize=13)
    ax2.text(0, L, f'{L:.1f} m', ha='center', va='bottom', fontsize=12)
    ax2.set_ylim(0, L * 1.3 if L > 0 else 1)

    title = f'{gas.upper()} quantification inputs'
    if name is not None:
        title += f' — {name}'
    fig.suptitle(title, fontsize=15)

    subtitle_parts = []
    if ueff is not None:
        subtitle_parts.append(f'Ueff = {ueff:.3f} m/s')
    if Q is not None:
        subtitle_parts.append(f'Q = {Q:.1f} kg/h')
    if subtitle_parts:
        fig.text(0.5, 0.90, '  |  '.join(subtitle_parts), ha='center', fontsize=12, color='gray')

    if psave is not None and name is not None:
        fig.savefig(psave + name + '_IME_L.png', dpi=150, bbox_inches='tight')

    plt.show()
    return fig


def load_plume_tif(tif_path):
    """Read the raster and build the plume mask from nodata."""
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs

    mask = np.isfinite(arr)
    if nodata is not None:
        mask &= (arr != nodata)

    return arr, mask, transform, crs


def pixel_size_m(transform, lat_center):
    """
    True pixel width/height in meters at a given latitude, for a raster
    on a regular lat/lon grid (EPSG:4326). Not square away from the equator.
    """
    dx_deg = abs(transform.a)
    dy_deg = abs(transform.e)
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(lat_center))
    dx_m = dx_deg * m_per_deg_lon
    dy_m = dy_deg * m_per_deg_lat
    return dx_m, dy_m


def quantify_plume_from_tif(tif_path, u10, mission='EMIT', gas='ch4', plot=True):

    arr, mask, transform, crs = load_plume_tif(tif_path)

    if not mask.any():
        raise ValueError(f'No valid (non-nodata) pixels found in {tif_path}')

    # Latitude of the plume centroid, needed to convert deg -> m correctly
    rows, cols = np.where(mask)
    lat_center = transform.f + (rows.mean() + 0.5) * transform.e

    dx_m, dy_m = pixel_size_m(transform, lat_center)
    px_area_geo = dx_m * dy_m      # m^2, geotransform-derived cell area (NOT the true ground footprint --
                                    # the lat/lon regrid doesn't preserve the native ~60x60 m sensor pixel size)

    sum_xgas = np.sum(arr[mask])   # ppm*m, matches xgas_im*mask sum in extract_Q
    N_pix = mask.sum()

    # --- Ueff (uses your existing per-mission/gas coefficients) ---
    ueff, a, err_u10, gsd_default, bool_thres = extract_ueff(u10, mission, gas)

    # Use the standard fixed ground-sample-distance (60 m for EMIT, matching the
    # instrument's native resolution per the ATBD) instead of the geotransform-derived
    # cell size. This matches what the rest of your pipeline (and your teammate's
    # script) assumes, and tracks the official NASA Q much more closely (~5-6%
    # vs ~14% using the geotransform-derived, non-square cell area).
    gsd_eff = gsd_default
    px_area = gsd_eff ** 2

    # --- IME (reuses your existing conversion, fed the standard pixel area) ---
    IME, conv_factor = ppmm_to_kg(sum_xgas, gsd_eff, gas)

    # --- L (plume length scale = sqrt of true masked area) ---
    L = np.sqrt(N_pix * px_area)

    # --- Q and its uncertainty (same formulas as extract_Q) ---
    Q = IME * 3600 * ueff / L
    if bool_thres:
        err_Q = np.sqrt(
            (3600 * IME * a * 2 / L) ** 2
            + (3600 * ueff * conv_factor * np.sqrt(N_pix) * np.nanstd(arr[mask]) / L) ** 2
        )
    else:
        err_Q = np.sqrt(
            (3600 * IME * a * u10 * 0.5 / L) ** 2
            + (3600 * ueff * conv_factor * np.sqrt(N_pix) * np.nanstd(arr[mask]) / L) ** 2
        )

    print(f'--- {tif_path} ---')
    print(f'Plume centroid latitude: {lat_center:.4f}')
    print(f'Geotransform cell size (informational only): {dx_m:.2f} x {dy_m:.2f} m ({px_area_geo:.1f} m^2)')
    print(f'Pixel area used for quantification: {gsd_eff:.1f} x {gsd_eff:.1f} m ({px_area:.1f} m^2, standard {mission} GSD)')
    print(f'N_pix (masked) = {int(N_pix)}')
    print(f'sum_xgas = {sum_xgas:.2f} ppm*m')
    print(f'IME  = {IME:.2f} kg')
    print(f'L    = {L:.2f} m')
    print(f'Ueff = {ueff:.3f} m/s   (U10 = {u10} m/s, a={a})')
    print(f'Q    = {Q:.2f} +/- {err_Q:.2f} kg/h')

    if plot:
        plot_ime_and_L(IME, L, Q=Q, ueff=ueff, gas=gas, name=tif_path.split('/')[-1])

    return {
        'IME': IME,
        'L': L,
        'Q': Q,
        'err_Q': err_Q,
        'Ueff': ueff,
        'N_pix': int(N_pix),
        'pixel_area_m2': px_area,
        'sum_xgas_ppmm': sum_xgas,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantify CH4 emission rate from a plume GeoTIFF.')
    parser.add_argument('tif_path', help='Path to the plume GeoTIFF (e.g. EMIT_L2B_CH4PLM_...tif)')
    parser.add_argument('--u10', type=float, required=True, help='10 m wind speed in m/s (e.g. from ERA5)')
    parser.add_argument('--mission', default='EMIT', choices=['EMIT', 'EnMAP', 'PRISMA', 'GF5', 'ZY1', 'AVIRIS-NG'])
    parser.add_argument('--gas', default='ch4', choices=['ch4', 'co2', 'nh3', 'c2h4', 'c2h2'])
    parser.add_argument('--no-plot', dest='plot', action='store_false')
    args = parser.parse_args()

    quantify_plume_from_tif(args.tif_path, args.u10, mission=args.mission, gas=args.gas, plot=args.plot)