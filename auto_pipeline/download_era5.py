#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descarga ERA5 (u10/v10) para la fecha/hora/zona de una escena EMIT, via el CDS de
Copernicus (cdsapi). Genera un .nc que luego se pasa al pipeline con --era5-file.

REQUISITO: cuenta gratuita en https://cds.climate.copernicus.eu + archivo
~/.cdsapirc con tu API key (ver https://cds.climate.copernicus.eu/how-to-api).
La cola del CDS puede tardar minutos-horas; por eso es un paso aparte, no dentro
del pipeline interactivo.

Uso:
  python auto_pipeline/download_era5.py --lat 9.5369 --lon -73.499 --time 20230223173139 --out era5_pribbenow.nc
"""

import argparse


def main():
    ap = argparse.ArgumentParser(description='Download ERA5 10m wind (u10/v10) around a scene, via CDS.')
    ap.add_argument('--lat', type=float, required=True)
    ap.add_argument('--lon', type=float, required=True)
    ap.add_argument('--time', required=True, help='Scene UTC time YYYYMMDDhhmmss (from the EMIT filename / metadata).')
    ap.add_argument('--out', required=True, help='Output .nc path.')
    ap.add_argument('--pad', type=float, default=0.5, help='Half-size of the lat/lon box (deg) around the source.')
    a = ap.parse_args()

    import cdsapi
    ts = a.time
    year, month, day, hour = ts[:4], ts[4:6], ts[6:8], ts[8:10]
    north, south = a.lat + a.pad, a.lat - a.pad
    west, east = a.lon - a.pad, a.lon + a.pad

    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind'],
            'year': year, 'month': month, 'day': day,
            'time': f'{hour}:00',
            'area': [north, west, south, east],   # N, W, S, E
            'format': 'netcdf',
        },
        a.out,
    )
    print('saved', a.out)


if __name__ == '__main__':
    main()
