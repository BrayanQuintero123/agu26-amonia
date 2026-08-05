#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Viento ERA5 (ECMWF) como fuente alternativa a GEOS-FP, para consistencia con
Carbon Mapper / MARS (que usan ERA5). Adaptado de
wavelet_plume_finder_public/emit_ime.py::read_wind_era5.

ERA5 NO es de acceso libre como GEOS-FP: hay que tener el .nc descargado (del CDS
de Copernicus, con u10/v10). Este modulo solo LEE ese archivo e interpola
bilinealmente al punto de la fuente y a la hora de adquisicion.

Para descargar el ERA5 ver download_era5.py (requiere API key del CDS).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(os.path.dirname(_HERE), 'scripts')
for _p in (_SCRIPTS, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np


def ensure_era5_file(ts, lat, lon, psave, name, pad=0.5):
    """Return a local ERA5 .nc for this scene, downloading it from the CDS (cdsapi) if it isn't
    cached yet. Needs ~/.cdsapirc with a valid key. First download can be slow (CDS queue);
    afterwards it's cached as <name>_era5.nc."""
    path = psave + name + '_era5.nc'
    if os.path.exists(path):
        return path

    import cdsapi
    ts = str(ts)
    year, month, day, hour = ts[:4], ts[4:6], ts[6:8], ts[8:10]
    north, south = lat + pad, lat - pad
    west, east = lon - pad, lon + pad
    print('  [wind] descargando ERA5 del CDS (la cola del CDS puede tardar minutos-horas; se cachea)...')
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind'],
            'year': year, 'month': month, 'day': day,
            'time': f'{hour}:00',
            'area': [north, west, south, east],
            'format': 'netcdf',
        },
        path,
    )
    print('  [wind] ERA5 descargado:', path)
    return path


def wind_speed_era5(ts, lat, lon, era5_file, psave, name):
    """Bilinear u10/v10 from a local ERA5 .nc at (lat,lon) for the scene time.
    Returns (u10, u, v). Same idea/signature spirit as wind_bilinear.wind_speed_bilinear.
    Caches to <name>_u_arr_era5.npy."""
    import xarray as xr
    import pandas as pd
    from scipy.interpolate import RegularGridInterpolator

    cache = psave + name + '_u_arr_era5.npy'
    if os.path.exists(cache):
        wa = np.load(cache)
        print('ERA5 wind uploaded!')
        return float(wa[0]), float(wa[1]), float(wa[2])

    ts = str(ts)
    scene_time = pd.Timestamp(f'{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[8:10]}:{ts[10:12]}:{ts[12:14]}')

    ds = xr.open_dataset(era5_file)
    lat_dim = 'latitude' if 'latitude' in ds.dims else 'lat'
    lon_dim = 'longitude' if 'longitude' in ds.dims else 'lon'

    for tdim in ('valid_time', 'time'):
        if tdim in ds.dims:
            ds = ds.sel({tdim: scene_time}, method='nearest')  # nearest hour to the acquisition
            break

    lat_vals = ds[lat_dim].values.astype(float)
    lon_vals = ds[lon_dim].values.astype(float)
    u = np.asarray(ds['u10'].values, dtype=float)
    v = np.asarray(ds['v10'].values, dtype=float)

    #ERA5 lon may be 0..360; convert the query lon to match if needed
    lon_q = lon
    if lon_vals.min() >= 0 and lon < 0:
        lon_q = lon + 360.0

    if lat_vals[0] > lat_vals[-1]:
        lat_vals = lat_vals[::-1]; u = u[::-1, :]; v = v[::-1, :]
    if lon_vals[0] > lon_vals[-1]:
        lon_vals = lon_vals[::-1]; u = u[:, ::-1]; v = v[:, ::-1]

    iu = RegularGridInterpolator((lat_vals, lon_vals), u, method='linear', bounds_error=False, fill_value=None)
    iv = RegularGridInterpolator((lat_vals, lon_vals), v, method='linear', bounds_error=False, fill_value=None)
    ue = float(iu([[lat, lon_q]])[0])
    ve = float(iv([[lat, lon_q]])[0])
    u10 = float(np.hypot(ue, ve))

    np.save(cache, np.array([u10, ue, ve, ue, ve]))
    print('ERA5 wind: u10=%.2f m/s (u=%.2f, v=%.2f)' % (u10, ue, ve))
    return u10, ue, ve
