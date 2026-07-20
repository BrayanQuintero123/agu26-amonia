#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copia de scripts/main.py que usa la cuantificacion SEMI-AUTOMATICA
(emission_quantification_auto) en lugar de la manual. El retrieval L2
(deltax_rets) y el guardado CSV (excel_info) son los ORIGINALES, reutilizados
sin cambios. scripts/main.py queda intacto.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(os.path.dirname(_HERE), 'scripts')
for _p in (_SCRIPTS, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hs_tool4_deltaxgas import deltax_rets   # original L1->L2 retrieval
from store_results import excel_info          # original CSV writer
from auto_plume import emission_quantification_auto  # NEW semi-automatic quantification


def main_auto(p, n, psave, gas, site, sigma_mult=2.0):

    cube, mission = deltax_rets(p, n, psave)  # retrieval (or reload) — unchanged LARS core

    if gas in ('ch4', 'co2', 'c2h4', 'c2h2'):
        if gas == 'ch4':
            dxgas_show = cube[:, :, 1]; dxgas_quan = dxgas_show.copy()
        elif gas == 'co2':
            dxgas_show = cube[:, :, 3]; dxgas_quan = dxgas_show.copy()
        elif gas == 'c2h4':
            dxgas_show = cube[:, :, 11]; dxgas_quan = cube[:, :, 7]
        elif gas == 'c2h2':
            dxgas_show = cube[:, :, 25]; dxgas_quan = cube[:, :, 23]

        rad_ref = cube[:, :, 0]

        Q, err_Q, u10, err_u10, lat_s, lon_s, ts, bool_det = emission_quantification_auto(
            dxgas_show, dxgas_quan, rad_ref, mission, gas, p, n, psave, sigma_mult)

        if bool_det:
            fields = ['Site', 'Mission', 'Timestamp\nYYYYMMDDhhmmss', 'Source-lat(º)', 'Source-lon(º)',
                      'u10 (m/s)', 'err(u10)', 'Q (kg/h)', 'err(Q)']
            info = [site, mission, ts, round(lat_s, 4), round(lon_s, 4), round(u10, 2), round(err_u10, 2),
                    round(Q, 2), round(err_Q, 2)]

    elif gas == 'nh3':
        dxgas_show = cube[:, :, 19]; dxgas_quan = cube[:, :, 13]
        rad_ref = cube[:, :, 0]

        Q_1, err_Q_1, Q_2, err_Q_2, u10, err_u10, lat_s, lon_s, ts, bool_det = emission_quantification_auto(
            dxgas_show, dxgas_quan, rad_ref, mission, gas, p, n, psave, sigma_mult)

        if bool_det:
            fields = ['Site', 'Mission', 'Timestamp\nYYYYMMDDhhmmss', 'Source-lat(º)', 'Source-lon(º)',
                      'u10 (m/s)', 'err(u10)', 'Q_tau=inf (kg/h)', 'err(Q_tau=inf)', 'Q_tau=1h (kg/h)', 'err(Q_tau=1h)']
            info = [site, mission, ts, round(lat_s, 4), round(lon_s, 4), round(u10, 2), round(err_u10, 2),
                    round(Q_1, 2), round(err_Q_1, 2), round(Q_2, 2), round(err_Q_2, 2)]

    else:
        bool_det = False

    if bool_det:
        excel_info(fields, info, psave, gas)

    return
