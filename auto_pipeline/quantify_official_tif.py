#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validacion inversa: cuantifica una pluma YA delineada (GeoTIFF oficial de EMIT L2B
CH4PLM, en ppm.m) con el codigo de LARS (extract_Q, SIN modificar), y compara
contra el valor del producto oficial.

Dos modos de viento (pueden usarse juntos, en una sola corrida):
  A) --u10 X       : viento pasado a mano (el que reporta la metadata del oficial).
                     Aisla el METODO usando inputs identicos a los del oficial.
  B) --fetch-era5  : descargamos NOSOTROS el ERA5 para la fecha/lugar del tif
                     (fecha del nombre del archivo, centro de la pluma) y sacamos u10.
                     Mezcla metodo + nuestra extraccion de viento (test del flujo real).

La mascara = todos los pixeles validos (no-nodata) del tif.

Uso:
  # solo metodo (viento del oficial):
  python auto_pipeline/quantify_official_tif.py --tif "<tif>" --u10 4.8335 --official-q 3615.84
  # ademas nuestro ERA5 auto:
  python auto_pipeline/quantify_official_tif.py --tif "<tif>" --u10 4.8335 --fetch-era5 --official-q 3615.84
"""

import os
import sys
import re
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
        T = ds.transform
        lat_c = T.f + T.e * ds.height / 2.0
    valid = np.isfinite(arr)
    if nd is not None:
        valid &= (arr != nd)
    xgas = np.where(valid, arr, 0.0)
    return xgas, valid, res, crs, lat_c, T


def scene_time_from_name(tif_path):
    """Extract YYYYMMDDhhmmss from an EMIT filename like ..._20220815T074645_..."""
    m = re.search(r'(\d{8})T(\d{6})', os.path.basename(tif_path))
    if not m:
        raise ValueError('no encontre el timestamp YYYYMMDDThhmmss en el nombre del tif')
    return m.group(1) + m.group(2)


def plume_centroid_latlon(valid, T):
    rows, cols = np.where(valid)
    lons = T.c + T.a * (cols + 0.5) + T.b * (rows + 0.5)
    lats = T.f + T.d * (cols + 0.5) + T.e * (rows + 0.5)
    return float(lats.mean()), float(lons.mean())


def errQ_with_wind_std(xgas, mask, u10, gas, wind_std):
    ueff, a, _e, gsd, _bt = extract_ueff(u10, 'EMIT', gas)
    IME, conv = ppmm_to_kg(np.sum(xgas * mask), gsd, gas)
    L = np.sqrt(np.sum(mask) * gsd ** 2)
    std = np.std(xgas)
    tw = 3600 * IME * a * wind_std / L
    tn = 3600 * ueff * conv * np.sqrt(np.sum(mask)) * std / L
    return np.sqrt(tw ** 2 + tn ** 2)


def quantify(xgas, mask, u10, gas, official_q, wind_std):
    Q, errQ, _, _ = extract_Q(xgas, mask, u10, 'EMIT', gas)
    ueff, a, _e, gsd, _bt = extract_ueff(u10, 'EMIT', gas)
    IME, conv = ppmm_to_kg(np.sum(xgas * mask), gsd, gas)
    L = np.sqrt(np.sum(mask) * gsd ** 2)
    print('    Q = %.1f +/- %.1f kg/h  (%.0f%%)   [Ueff=%.3f | IME=%.0f kg | L=sqrt(area)=%.0f m]'
          % (Q, errQ, 100 * errQ / Q, ueff, IME, L))
    if wind_std is not None:
        print('      (con err_viento=%.2f -> incert = %.1f kg/h)' % (wind_std, errQ_with_wind_std(xgas, mask, u10, gas, wind_std)))
    if official_q is not None:
        print('      vs oficial %.1f kg/h  ->  diff %+.0f%%' % (official_q, 100 * (Q - official_q) / official_q))


def main():
    ap = argparse.ArgumentParser(description='Quantify an official EMIT L2B plume tif with LARS; wind from metadata (--u10) and/or our own ERA5 (--fetch-era5).')
    ap.add_argument('--tif', required=True)
    ap.add_argument('--u10', type=float, default=None, help='Wind (m/s) from the product metadata (manual).')
    ap.add_argument('--fetch-era5', action='store_true', help='Also download OUR ERA5 for the scene (from the filename date + plume centroid) and quantify with it.')
    ap.add_argument('--gas', default='ch4')
    ap.add_argument('--wind-std', type=float, default=None, help='Optional wind error (m/s) for the alt uncertainty calc, e.g. 0.11.')
    ap.add_argument('--official-q', type=float, default=None)
    a = ap.parse_args()

    if a.u10 is None and not a.fetch_era5:
        raise SystemExit('da al menos --u10 (viento metadata) o --fetch-era5 (nuestro ERA5).')

    gas = a.gas.strip().lower()
    xgas, valid, res, crs, lat_c, T = read_plume_tif(a.tif)
    N = int(valid.sum())

    print('=== Tif oficial ===')
    print('  %s' % os.path.basename(a.tif))
    print('  pixeles de pluma=%d | max=%.1f ppm.m | gas=%s' % (N, np.nanmax(xgas), gas))
    if a.official_q is not None:
        print('  OFICIAL: Q = %.1f kg/h' % a.official_q)
    print()

    if a.u10 is not None:
        print('--- A) viento de la metadata (manual) = %.3f m/s ---' % a.u10)
        quantify(xgas, valid, a.u10, gas, a.official_q, a.wind_std)
        print()

    if a.fetch_era5:
        print('--- B) NUESTRO ERA5 (auto: fecha del nombre + centro de la pluma) ---')
        ts = scene_time_from_name(a.tif)
        clat, clon = plume_centroid_latlon(valid, T)
        psave = os.path.dirname(os.path.abspath(a.tif)) + '/'
        name = os.path.splitext(os.path.basename(a.tif))[0]
        print('    escena %s | centro pluma (%.4f, %.4f)' % (ts, clat, clon))
        try:
            from wind_era5 import ensure_era5_file, wind_speed_era5
            f = ensure_era5_file(ts, clat, clon, psave, name)
            u10_e, ue, ve = wind_speed_era5(ts, clat, clon, f, psave, name)
            print('    ERA5 nuestro: u10 = %.2f m/s' % u10_e)
            quantify(xgas, valid, u10_e, gas, a.official_q, a.wind_std)
        except Exception as e:
            print('    ERA5 fallo (%s); revisa tu ~/.cdsapirc' % e)


if __name__ == '__main__':
    main()
