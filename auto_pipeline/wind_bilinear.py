#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variante de scripts/wind_v2.py con interpolacion ESPACIAL BILINEAL (en vez de
vecino mas cercano) sobre la grilla GEOS-FP. Todo lo demas (modelo GEOS-FP,
producto tavg1_2d_slv_Nx, interpolacion TEMPORAL al minuto de adquisicion,
Ueff calibrado aguas abajo, cache .npy) se conserva identico al original.

El original vive en scripts/wind_v2.py y queda intacto.
Tecnica bilineal tomada de wavelet_plume_finder_public/emit_ime.py::read_wind_era5.
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(os.path.dirname(_HERE), 'scripts')
for _p in (_SCRIPTS, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from netCDF4 import Dataset
import requests
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator


def format_dw_url(year, month, day, hour):
    template = "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/Y{y}/M{m:02d}/D{d:02d}/GEOS.fp.asm.tavg1_2d_slv_Nx.{y}{m:02d}{d:02d}_{h:02d}30.V01.nc4"
    return template.format(y=year, m=month, d=day, h=hour)


def _download_with_retry(url, tmp_file, max_retries=5, timeout=120):
    """
    Streams `url` to `tmp_file`, retrying with exponential backoff on dropped
    connections or truncated transfers (IncompleteRead / ChunkedEncodingError).
    Verifies the number of bytes written matches Content-Length when the
    server provides it, so a partial file never silently gets passed to
    netCDF4.Dataset().
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                expected = r.headers.get("Content-Length")
                expected = int(expected) if expected is not None else None

                written = 0
                with open(tmp_file, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            written += len(chunk)

                if expected is not None and written != expected:
                    raise IOError(
                        f"Incomplete download: got {written} bytes, expected {expected}"
                    )
            return  # success

        except (requests.exceptions.RequestException, IOError) as e:
            last_exc = e
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
            if attempt == max_retries:
                break
            wait = min(2 ** attempt, 30)
            print(f"  Download attempt {attempt}/{max_retries} failed ({e}); retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts") from last_exc


def get_geos_u10(date: str, hour: int, coords: tuple, pbase): #stores the file, reads it, then removes it
    year, month, day = date.split("-")
    lat_c, lon_c = coords

    dw_url = format_dw_url(year, int(month), int(day), hour)

    tmp_path = os.path.join(pbase, "tmp")
    os.makedirs(tmp_path, exist_ok=True)

    tmp_file = os.path.join(pbase + 'tmp/', "wind.nc")
    _download_with_retry(dw_url, tmp_file)

    try:
        winds = Dataset(tmp_file)

        lat = np.asarray(winds["lat"][:], dtype=float)
        lon = np.asarray(winds["lon"][:], dtype=float)
        u_grid = np.asarray(winds["U10M"][:][0], dtype=float) #(nlat, nlon)
        v_grid = np.asarray(winds["V10M"][:][0], dtype=float)

        winds.close()
    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

    #RegularGridInterpolator requires strictly-ascending axes
    if lat[0] > lat[-1]:
        lat = lat[::-1]; u_grid = u_grid[::-1, :]; v_grid = v_grid[::-1, :]
    if lon[0] > lon[-1]:
        lon = lon[::-1]; u_grid = u_grid[:, ::-1]; v_grid = v_grid[:, ::-1]

    interp_u = RegularGridInterpolator((lat, lon), u_grid, method='linear', bounds_error=False, fill_value=None)
    interp_v = RegularGridInterpolator((lat, lon), v_grid, method='linear', bounds_error=False, fill_value=None)

    u = float(interp_u([[lat_c, lon_c]])[0]) #bilinear at the exact source lat/lon
    v = float(interp_v([[lat_c, lon_c]])[0])

    return u, v


def wind_speed_bilinear(ts, lat, lon, pbase, psave, name): #same signature/behaviour as wind_v2.wind_speed_manual, only spatial interp changed
    #NOTE: cache uses a distinct suffix so it never collides with the original nearest-neighbor cache
    content_match = os.listdir(psave)
    key = name + '_u_arr_bilinear.npy'; len_key = len(key)
    bool_match = False
    for c in content_match:
        if c[:len_key] == key:
            bool_match = True

    if bool_match == True:
        u_arr_final = np.load(psave + name + '_u_arr_bilinear.npy')
        u10 = u_arr_final[0]
        print('Wind data uploaded! (bilinear)')

    else:
        ts = str(ts)
        u_array, i = np.zeros((1, 9)), 0
        year, month, day, hour, minute = ts[:4], ts[4:6], ts[6:8], int(ts[8:10]), int(ts[10:12])

        if hour == 0:

            if int(day) < 11:
                day_new = int(day) - 1
                day_new = '0' + str(day_new)
            else:
                day_new = int(day) - 1
                day_new = str(day_new)

            date = year + '-' + month + '-' + day_new
            hour_new = 23

            print('Starting GEOS-FP wind request (bilinear)')
            u_array[i, 0], u_array[i, 3] = get_geos_u10(date, hour_new, (lat, lon), pbase)
            print('1/3')
            date = year + '-' + month + '-' + day
            u_array[i, 1], u_array[i, 4] = get_geos_u10(date, hour, (lat, lon), pbase)
            print('2/3')
            u_array[i, 2], u_array[i, 5] = get_geos_u10(date, hour + 1, (lat, lon), pbase)
            print('3/3')

        elif hour == 23:

            if int(day) < 9:
                day_new = int(day) + 1
                day_new = '0' + str(day_new)
            else:
                day_new = int(day) + 1
                day_new = str(day_new)

            hour_new = 0

            date = year + '-' + month + '-' + day
            print('Starting GEOS-FP wind request (bilinear)')
            u_array[i, 0], u_array[i, 3] = get_geos_u10(date, hour - 1, (lat, lon), pbase)
            print('1/3')
            u_array[i, 1], u_array[i, 4] = get_geos_u10(date, hour, (lat, lon), pbase)
            print('2/3')
            date = year + '-' + month + '-' + day_new
            u_array[i, 2], u_array[i, 5] = get_geos_u10(date, hour_new, (lat, lon), pbase)
            print('3/3')
        else:

            date = year + '-' + month + '-' + day
            print('Starting GEOS-FP wind request (bilinear)')
            u_array[i, 0], u_array[i, 3] = get_geos_u10(date, hour - 1, (lat, lon), pbase)
            print('1/3')
            u_array[i, 1], u_array[i, 4] = get_geos_u10(date, hour, (lat, lon), pbase)
            print('2/3')
            u_array[i, 2], u_array[i, 5] = get_geos_u10(date, hour + 1, (lat, lon), pbase)
            print('3/3')

        #Temporal interpolation to the acquisition instant (tavg1 nodes centered at HH:30 -> h-0.5, h+0.5, h+1.5)
        arr_x, arr_y = np.array([hour - 0.5, hour + 0.5, hour + 1.5]), np.array([u_array[i, 0], u_array[i, 1], u_array[i, 2]])
        fu = interpolate.interp1d(arr_x, arr_y, fill_value="extrapolate")
        arr_x, arr_y = np.array([hour - 0.5, hour + 0.5, hour + 1.5]), np.array([u_array[i, 3], u_array[i, 4], u_array[i, 5]])
        fv = interpolate.interp1d(arr_x, arr_y, fill_value="extrapolate")
        final_t = hour + (minute - 10) / 60
        u_array[i, 6], u_array[i, 7] = fu(final_t), fv(final_t)
        u_array[i, 8] = np.sqrt((u_array[i, 6]**2) + (u_array[i, 7]**2))
        added_time = (minute - 10) / 60
        if added_time >= 0:
            u10, ux_hour, uy_hour, ux_add, uy_add = u_array[i, 8], u_array[i, 1], u_array[i, 4], u_array[i, 2], u_array[i, 5]
        else:
            u10, ux_hour, uy_hour, ux_add, uy_add = u_array[i, 8], u_array[i, 1], u_array[i, 4], u_array[i, 0], u_array[i, 3]
        u_arr_final = np.array([u10, ux_hour, uy_hour, ux_add, uy_add])
        np.save(psave + name + '_u_arr_bilinear.npy', u_arr_final)

        print('(u_0, v_0) = (' + str(round(u_arr_final[1], 2)) + ', ' + str(round(u_arr_final[2], 2)) + ') m/s (point 0 for interpolation)')
        print('(u_1, v_1) = (' + str(round(u_arr_final[3], 2)) + ', ' + str(round(u_arr_final[4], 2)) + ') m/s (point 1 for interpolation)')
        print('u10 = ' + str(round(u_arr_final[0], 2)) + ' m/s (used in IME-quantification)')

        u10 = u_arr_final[0]

    return u10