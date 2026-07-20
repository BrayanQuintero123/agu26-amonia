#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Punto de entrada del pipeline CH4 SEMI-AUTOMATICO.

Igual que ch4_pipeline.py (raiz) pero la delineacion de pluma es automatica
(clic en la fuente -> crecimiento por umbral+componente conexa -> double-check)
y el viento usa interpolacion bilineal. Todo el nucleo de LARS (retrieval,
IME, Ueff, conversion) se reutiliza sin cambios; los archivos originales quedan
intactos.

Uso:
  python auto_pipeline/ch4_pipeline_auto.py "<ruta al EMIT L1B RAD .nc>"
  python auto_pipeline/ch4_pipeline_auto.py "<...RAD.nc>" -o "<carpeta salida>" -s "Sitio" --sigma 2.0
"""

import os
import sys
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(os.path.dirname(_HERE), 'scripts')
for _p in (_SCRIPTS, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from main_auto import main_auto


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Semi-automatic CH4 pipeline: LARS L1->L2 retrieval + auto plume delineation '
                    '(source click + threshold growth + operator double-check) + bilinear GEOS-FP wind + IME.')
    parser.add_argument('rad_file', help='Full path to the EMIT L1B RAD .nc (its matching OBS .nc must be in the same folder).')
    parser.add_argument('-o', '--output-dir', default=None, help='Folder for L2/L4 outputs. Default: an "output" subfolder next to the RAD file.')
    parser.add_argument('-s', '--site', default='site', help='Site name, only labels the output csv row.')
    parser.add_argument('--sigma', type=float, default=2.0, help='Initial sigma multiplier for the plume threshold (default 2.0; adjustable interactively).')
    args = parser.parse_args()

    path_img = os.path.dirname(os.path.abspath(args.rad_file)) + '/'
    name_img = os.path.splitext(os.path.basename(args.rad_file))[0]

    psave = args.output_dir if args.output_dir else os.path.join(path_img, 'output') + '/'
    os.makedirs(psave, exist_ok=True)

    main_auto(path_img, name_img, psave, 'ch4', args.site, sigma_mult=args.sigma)
