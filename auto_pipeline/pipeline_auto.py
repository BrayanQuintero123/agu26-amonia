#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline semi-automatico GENERAL: corre el mismo flujo que ch4 (den + clic en el
hotspot + crecimiento automatico + double-check) para CUALQUIER gas cuantificable,
o para varios en una sola corrida.

El retrieval L2 calcula TODOS los gases una sola vez; luego cada gas pedido se
delimita/cuantifica de forma interactiva por turnos. Reusa el nucleo de LARS
(retrieval, IME, Ueff, georreferenciacion) y la delineacion semi-automatica.
Los archivos originales quedan intactos.

Gases cuantificables (tienen calibracion Ueff + masa molar): ch4, co2, c2h4, c2h2, nh3.
(h2o NO se cuantifica: es vapor de agua, se usa de otra forma.)

Uso:
  python auto_pipeline/pipeline_auto.py "<RAD.nc>"  --gas ch4                          # EMIT (auto-detectado)
  python auto_pipeline/pipeline_auto.py "<RAD.h5>"  --gas ch4 --mission Tanager        # Tanager
  python auto_pipeline/pipeline_auto.py "<RAD.nc>"  --gas ch4 --wind era5              # con ERA5
  python auto_pipeline/pipeline_auto.py "<RAD.nc>"  --gas ch4,nh3,c2h4 -s "Carrasco"
"""

import os
import sys
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(os.path.dirname(_HERE), 'scripts')
for _p in (_SCRIPTS, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hs_tool4_deltaxgas import deltax_rets            # original L1->L2 retrieval (all gases at once)
from store_results import excel_info                   # original CSV writer
from auto_plume import emission_quantification_auto     # semi-automatic quantification
from gas_bands import GAS_BANDS                          # gas -> (show_band, quan_band)


def _run_one_gas(cube, mission, gas, p, n, psave, site, sigma, wind_source='geos', era5_file=None,
                 wind_value=None, wind_from=None):
    show_b, quan_b = GAS_BANDS[gas]
    dxgas_show = cube[:, :, show_b]
    dxgas_quan = cube[:, :, quan_b]
    rad_ref = cube[:, :, 0]

    if gas == 'nh3':
        Q_1, err_Q_1, Q_2, err_Q_2, u10, err_u10, lat_s, lon_s, ts, bool_det = emission_quantification_auto(
            dxgas_show, dxgas_quan, rad_ref, mission, gas, p, n, psave, sigma, wind_source, era5_file, wind_value, wind_from)
        if bool_det:
            fields = ['Site', 'Mission', 'Timestamp\nYYYYMMDDhhmmss', 'Source-lat(º)', 'Source-lon(º)',
                      'u10 (m/s)', 'err(u10)', 'Q_tau=inf (kg/h)', 'err(Q_tau=inf)', 'Q_tau=1h (kg/h)', 'err(Q_tau=1h)']
            info = [site, mission, ts, round(lat_s, 4), round(lon_s, 4), round(u10, 2), round(err_u10, 2),
                    round(Q_1, 2), round(err_Q_1, 2), round(Q_2, 2), round(err_Q_2, 2)]
            excel_info(fields, info, psave, gas)
    else:
        Q, err_Q, u10, err_u10, lat_s, lon_s, ts, bool_det = emission_quantification_auto(
            dxgas_show, dxgas_quan, rad_ref, mission, gas, p, n, psave, sigma, wind_source, era5_file, wind_value, wind_from)
        if bool_det:
            fields = ['Site', 'Mission', 'Timestamp\nYYYYMMDDhhmmss', 'Source-lat(º)', 'Source-lon(º)',
                      'u10 (m/s)', 'err(u10)', 'Q (kg/h)', 'err(Q)']
            info = [site, mission, ts, round(lat_s, 4), round(lon_s, 4), round(u10, 2), round(err_u10, 2),
                    round(Q, 2), round(err_Q, 2)]
            excel_info(fields, info, psave, gas)


def run_gases(p, n, psave, gases, site, sigma=2.0, wind_source='geos', era5_file=None,
              wind_value=None, wind_from=None, mission=None):
    cube, mission = deltax_rets(p, n, psave, mission=mission)
    for gas in gases:
        print(f'\n===================  {gas.upper()}  ===================')
        _run_one_gas(cube, mission, gas, p, n, psave, site, sigma, wind_source, era5_file, wind_value, wind_from)


if __name__ == '__main__':

    ap = argparse.ArgumentParser(
        description='Semi-automatic plume delineation + IME quantification for one or several gases.')
    ap.add_argument('rad_file', help='Ruta al archivo de radiancia: .nc para EMIT/EnMAP/PRISMA, .h5 para Tanager.')
    ap.add_argument('--gas', default='ch4',
                    help='Gas or comma-separated list. Options: ch4, co2, c2h4, c2h2, nh3. Example: --gas ch4,nh3')
    ap.add_argument('-o', '--output-dir', default=None, help='Folder for L2/L4 outputs. Default: an "output" subfolder next to the RAD file.')
    ap.add_argument('-s', '--site', default='site', help='Site name, only labels the output csv rows.')
    ap.add_argument('--sigma', type=float, default=2.0, help='Initial sigma multiplier for the plume threshold (adjustable interactively).')
    ap.add_argument('--wind', choices=['geos', 'era5'], default='geos', help='Wind source to quantify with. Both are printed for comparison if --era5-file is given.')
    ap.add_argument('--era5', action='store_true', help='Shorthand for --wind era5. Searches for an ERA5 .nc in the same folder as the input file.')
    ap.add_argument('--era5-file', default=None, help='Path to a local ERA5 .nc (u10/v10). Required if --wind era5 and not using --era5 auto-search.')
    ap.add_argument('--wind-value', type=float, default=None, help='Manual wind speed (m/s) override for THIS run only (e.g. 1.1 to match a reference). GEOS/ERA5 still printed for comparison.')
    ap.add_argument('--wind-from', type=float, default=None, help='Optional wind FROM-direction (met. degrees) for the manual override, only affects the map arrow.')
    ap.add_argument('--mission', default=None,
                    choices=['EMIT', 'Tanager', 'EnMAP', 'PRISMA', 'GF5', 'AVIRIS-NG'],
                    help='Mision/satelite. Si no se da, se auto-detecta del nombre del archivo.')
    args = ap.parse_args()

    # --era5 flag: shorthand for --wind era5, auto-finds the ERA5 file next to the input
    if args.era5:
        args.wind = 'era5'
        if args.era5_file is None:
            import glob
            input_dir = os.path.dirname(os.path.abspath(args.rad_file))
            candidates = glob.glob(os.path.join(input_dir, '*.nc')) + glob.glob(os.path.join(input_dir, '*era5*.nc4'))
            era5_candidates = [f for f in candidates if 'era5' in os.path.basename(f).lower()]
            if era5_candidates:
                args.era5_file = era5_candidates[0]
                print(f'[ERA5] Auto-detected: {args.era5_file}')
            else:
                raise SystemExit('--era5: no ERA5 .nc found next to the input file. Use --era5-file <path>.')

    gases = [g.strip().lower() for g in args.gas.split(',') if g.strip()]
    invalid = [g for g in gases if g not in GAS_BANDS]
    if invalid:
        raise SystemExit(f'Not quantifiable / unknown gas(es): {invalid}. Valid: {list(GAS_BANDS)}')

    path_img = os.path.dirname(os.path.abspath(args.rad_file)) + '/'
    name_img = os.path.splitext(os.path.basename(args.rad_file))[0]
    psave = args.output_dir if args.output_dir else os.path.join(path_img, 'output') + '/'
    os.makedirs(psave, exist_ok=True)

    run_gases(path_img, name_img, psave, gases, args.site, sigma=args.sigma,
              wind_source=args.wind, era5_file=args.era5_file,
              wind_value=args.wind_value, wind_from=args.wind_from,
              mission=args.mission)
