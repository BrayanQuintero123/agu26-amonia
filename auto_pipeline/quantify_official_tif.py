#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validacion inversa: cuantifica una pluma YA delineada (un GeoTIFF oficial de EMIT
L2B CH4PLM, en ppm.m) usando el codigo de LARS (extract_Q, SIN modificar) con un
viento dado, y lo compara contra el valor reportado por el producto oficial.

La mascara = todos los pixeles validos (no-nodata) del tif. El viento y los
valores oficiales se pasan por linea de comandos (de la metadata del producto).

Uso:
  python auto_pipeline/quantify_official_tif.py --tif "C:/.../EMIT_L2B_CH4PLM_....tif" --u10 4.8335 \
         --official-q 3615.84 --official-uncert 194.96 --wind-std 0.11

  # minimo:
  python auto_pipeline/quantify_official_tif.py --tif "<tif>" --u10 4.8335
"""

import os
import sys
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(os.path.dirname(_HERE), 'scripts')
for _p in (_SCRIPTS, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import rasterio

from quant_func_v2 import extract_Q, extract_ueff, ppmm_to_kg


def read_plume_tif(tif_path):
    with rasterio.open(tif_path) as ds:
        arr = ds.read(1).astype(float)
        nd = ds.nodata
        res = ds.res
        crs = ds.crs
        # true ground pixel size (lat/lon grids get compressed in lon by cos(lat))
        lat_c = ds.transform.f + ds.transform.e * ds.height / 2.0
    valid = np.isfinite(arr)
    if nd is not None:
        valid &= (arr != nd)
    xgas = np.where(valid, arr, 0.0)        # nodata -> 0 (so sum only counts the plume)
    return xgas, valid, res, crs, lat_c


def errQ_with_wind_std(xgas, mask, u10, mission, gas, wind_std):
    """Recompute LARS's err_Q formula but substituting err_u10 = wind_std (for CH4-type gases).
    Transparent re-implementation of the exact formula in quant_func_v2.extract_Q — the LARS code
    itself is NOT modified."""
    ueff, a, _err_u10, gsd, _bt = extract_ueff(u10, mission, gas)
    sum_xgas = np.sum(xgas * mask)
    N = np.sum(mask)
    IME, conv = ppmm_to_kg(sum_xgas, gsd, gas)
    L = np.sqrt(N * gsd ** 2)
    std = np.std(xgas)
    term_wind = 3600 * IME * a * wind_std / L
    term_noise = 3600 * ueff * conv * np.sqrt(N) * std / L
    return np.sqrt(term_wind ** 2 + term_noise ** 2), term_wind, term_noise


def main():
    ap = argparse.ArgumentParser(description='Quantify an official EMIT L2B plume tif with LARS (extract_Q) and compare.')
    ap.add_argument('--tif', required=True, help='Path to the official plume GeoTIFF (ppm.m).')
    ap.add_argument('--u10', type=float, required=True, help='Wind speed (m/s) from the product metadata.')
    ap.add_argument('--gas', default='ch4', help='ch4 / co2 / c2h4 / c2h2 / nh3')
    ap.add_argument('--wind-std', type=float, default=None, help='Optional: substitute this wind error (m/s) into LARS err_Q, e.g. 0.11 (ERA5 std).')
    ap.add_argument('--official-q', type=float, default=None, help='Official Emissions Rate (kg/hr) for comparison.')
    ap.add_argument('--official-uncert', type=float, default=None, help='Official uncertainty (kg/hr) for comparison.')
    a = ap.parse_args()

    gas = a.gas.strip().lower()
    xgas, mask, res, crs, lat_c = read_plume_tif(a.tif)
    N = int(mask.sum())

    print('=== Tif oficial ===')
    print(f'  {os.path.basename(a.tif)}')
    print(f'  CRS={crs} | pixel={res[0]:.6g} x {res[1]:.6g} | pixeles de pluma (validos)={N}')
    print(f'  max enh = {np.nanmax(xgas):.1f} ppm.m | u10 usado = {a.u10} m/s | gas = {gas}')
    if crs and crs.to_epsg() == 4326:
        px_lat = res[1] * 111320.0
        px_lon = res[0] * 111320.0 * np.cos(np.radians(lat_c))
        print(f'  (nota: pixel real ~ {px_lat:.0f} m lat x {px_lon:.0f} m lon; LARS asume gsd=60 m para EMIT)')
    print()

    print('=== LARS extract_Q (codigo sin modificar) ===')
    if gas == 'nh3':
        Q1, e1, Q2, e2, u10o, err_u10 = extract_Q(xgas, mask, a.u10, 'EMIT', gas)
        print(f'  Q(tau->inf) = {Q1:.1f} +/- {e1:.1f} kg/h')
        print(f'  Q(tau->1h)  = {Q2:.1f} +/- {e2:.1f} kg/h')
        Q, errQ = Q1, e1
    else:
        Q, errQ, u10o, err_u10 = extract_Q(xgas, mask, a.u10, 'EMIT', gas)
        print(f'  Q = {Q:.1f} +/- {errQ:.1f} kg/h  ({100*errQ/Q:.0f}%)')

    # decomposition
    if gas == 'nh3':
        ueff, aa, _, _, gsd, _bt = (*extract_ueff(a.u10, 'EMIT', gas)[:2], None, None,
                                    extract_ueff(a.u10, 'EMIT', gas)[5], None)
    else:
        ueff, aa, _err_u10, gsd, _bt = extract_ueff(a.u10, 'EMIT', gas)
    sum_xgas = np.sum(xgas * mask)
    IME, conv = ppmm_to_kg(sum_xgas, gsd, gas)
    L = np.sqrt(N * gsd ** 2)
    print(f'    piezas -> Ueff={ueff:.3f} m/s | IME={IME:.1f} kg | L=sqrt(N*60^2)={L:.0f} m | sum={sum_xgas:.0f} ppm.m')
    print()

    if a.wind_std is not None and gas != 'nh3':
        e_ov, tw, tn = errQ_with_wind_std(xgas, mask, a.u10, 'EMIT', gas, a.wind_std)
        print(f'=== override: err_u10 = {a.wind_std} m/s (misma formula, sin tocar LARS) ===')
        print(f'  err_Q = {e_ov:.1f} kg/h ({100*e_ov/Q:.0f}%)   [term viento={tw:.1f} | term ruido={tn:.1f}]')
        print()

    if a.official_q is not None:
        print('=== Comparacion con el oficial ===')
        dq = 100 * (Q - a.official_q) / a.official_q
        print(f'  Q:          LARS={Q:.1f}  vs  oficial={a.official_q:.1f} kg/h   (diff {dq:+.1f}%)')
        if a.official_uncert is not None:
            print(f'  incertid.:  LARS={errQ:.1f}  vs  oficial={a.official_uncert:.1f} kg/h')


if __name__ == '__main__':
    main()
